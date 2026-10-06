"""Adversarial checks of the trajectory hypothesis, time measure and lifecycle."""
import copy
import json

import numpy as np
import pytest
import torch

from scripts.configure_trajectories import VARIANTS, SEEDS
from src.experiments import training
from src.experiments.features import encode_images
from src.experiments.losses import distribution_loss
from src.experiments.rare_sampling import sampling_probabilities, corrected_kl
from src.experiments.runtime import load_config, digest_file, cache_spec
from src.experiments.taxonomy import REACTION_CLASSES
from src.experiments.trajectory import (TrajectoryPredictor, TrajectoryVideos, aggregate_moments,
    temporal_weights, interval_packet, load_prototypes, permute_intervals)


@pytest.fixture(autouse=True)
def one_cpu_thread():
    torch.set_num_threads(1)


def coordinates():
    return torch.rand(21, 3, generator=torch.Generator().manual_seed(20)) * 2 - 1


@pytest.fixture
def prototype_asset(tmp_path, monkeypatch):
    path = tmp_path / 'vad.json'
    asset = {'class_order': REACTION_CLASSES, 'scale': [-1, 1], 'source': {'version': '2.1'},
        'labels': {c: dict(zip(('valence', 'arousal', 'dominance'), v.tolist()), source_word=c, approximation=False)
                   for c, v in zip(REACTION_CLASSES, coordinates())}}
    path.write_text(json.dumps(asset))
    monkeypatch.setenv('V2R_VAD_V2_FILE', str(path))
    return path


def config(name, asset):
    cfg = load_config(f'configs/experiments/{name}.yaml')
    cfg['model'].update(input_dim=8, hidden_dim=16, layers=1)
    cfg['training'].update(epochs=3, batch_size=2)
    cfg['trajectory']['prototypes']['asset_sha256'] = digest_file(asset)
    return cfg


class Visual:
    def __init__(self):
        self.ids = ['a', 'b', 'c']
        self.counts = np.array([3, 5, 4])
        g = torch.Generator().manual_seed(22)
        self.frames = [torch.randn(int(n), 8, generator=g) for n in self.counts]
        self.targets = torch.randn(3, 21, generator=g).softmax(-1).numpy()
        self.frame_records = [{'sample_id': vid, 'frames': [
            {'index': j, 'start_time': str(j * 2), 'end_time': str((j+1) * 2)} for j in range(n)]}
            for vid, n in zip(self.ids, self.counts)]
    def __len__(self):
        return 3
    def __getitem__(self, i):
        return self.frames[i], torch.from_numpy(self.targets[i]), i


def test_real_timestamp_format_positive_durations_and_no_gap_invention():
    value = interval_packet([{'start_time': '00:00:00.000', 'end_time': '00:00:01.560'},
                             {'start_time': '00:00:03.040', 'end_time': '00:00:06.000'}])
    torch.testing.assert_close(value[:, 0], torch.tensor([1.56, 2.96]))
    for records in ([{'start_time': 'nan', 'end_time': '4'}],
                    [{'start_time': '3', 'end_time': '2'}],
                    [{'start_time': '0', 'end_time': '3'}, {'start_time': '2', 'end_time': '4'}]):
        with pytest.raises(ValueError):
            interval_packet(records)


@pytest.mark.parametrize('mode', ['duration', 'soft', 'sparse'])
def test_density_is_invariant_to_interval_splitting_and_padding(mode):
    score = torch.tensor([[.1, 2., -4.]], dtype=torch.double, requires_grad=True)
    dt = torch.tensor([[6., 3., 7.]], dtype=torch.double)
    mask = torch.ones_like(score, dtype=torch.bool)
    w, density = temporal_weights(score, dt, mask, mode)
    split_score = score[:, [0, 0, 1, 2]]
    split_dt = torch.tensor([[2., 4., 3., 7.]], dtype=torch.double)
    other, _ = temporal_weights(split_score, split_dt, torch.ones_like(split_score, dtype=torch.bool), mode)
    torch.testing.assert_close(w, torch.stack([other[:, :2].sum(1), other[:, 2], other[:, 3]], dim=1))
    torch.testing.assert_close(w.sum(1), torch.ones(1, dtype=torch.double))
    if mode == 'sparse':
        assert (w == 0).any()
    if mode != 'duration':
        assert torch.autograd.gradcheck(lambda s: temporal_weights(s, dt, mask, mode)[0], (score,))
        torch.testing.assert_close(w, temporal_weights(score + 50, dt, mask, mode)[0])
    padded, _ = temporal_weights(torch.cat([score, score.new_tensor([[1e8]])], 1),
                                torch.cat([dt, dt.new_tensor([[-1.]])], 1),
                                torch.tensor([[True, True, True, False]]), mode)
    torch.testing.assert_close(w, padded[:, :3])
    assert padded[0, -1] == 0


def test_probability_mixture_matches_duration_example_and_relevance_can_reverse_it():
    q = torch.tensor([[[.9, .1], [.4, .6]]], dtype=torch.double)
    dt = torch.tensor([[6., 3.]], dtype=torch.double)
    mask = torch.ones(1, 2, dtype=torch.bool)
    score = torch.tensor([[0., 5.]], dtype=torch.double)
    p, attention, details = aggregate_moments(q.log(), q.log(), dt, mask, score, 'duration')
    torch.testing.assert_close(p.exp(), torch.tensor([[.7333333333333333, .2666666666666667]], dtype=torch.double))
    salient, _, _ = aggregate_moments(q.log(), q.log(), dt, mask, score, 'sparse')
    assert salient.exp()[0, 1] > salient.exp()[0, 0]
    torch.testing.assert_close(details['contributions'].sum(1), p.exp())


def test_absolute_proximity_is_not_local_softmax_and_peak_definitions_differ():
    affinity = torch.tensor([[[.9, .8], [.02, .1]]], dtype=torch.double)
    log_q = affinity.log().log_softmax(-1)
    dt = torch.ones(1, 2, dtype=torch.double)
    mask = torch.ones(1, 2, dtype=torch.bool)
    score = torch.zeros_like(dt)
    mean = aggregate_moments(log_q, affinity.log(), dt, mask, score, 'duration')[0].exp()
    raw = aggregate_moments(log_q, affinity.log(), dt, mask, score, 'sparse_proximity')[0].exp()
    torch.testing.assert_close(raw, affinity.sum(1) / affinity.sum())
    assert not torch.allclose(mean, raw)
    peak = aggregate_moments(log_q, affinity.log(), dt, mask, score, 'peak')[0].exp()
    torch.testing.assert_close(peak, log_q[:, 0].exp())
    # Maximum relative fear confidence occurs at the second, more distant moment.
    assert log_q.exp()[0, :, 1].argmax() == 1
    class_peak = aggregate_moments(log_q, affinity.log(), dt, mask, score, 'class_peak')[0].exp()
    torch.testing.assert_close(class_peak, torch.tensor([[.9/1.7, .8/1.7]], dtype=torch.double))


def test_power_zero_is_mean_and_one_uses_both_duration_weighted_sums():
    q = torch.tensor([[[.9, .1], [.4, .6]]], dtype=torch.double)
    dt = torch.tensor([[6., 3.]], dtype=torch.double)
    mask = torch.ones(1, 2, dtype=torch.bool)
    score = torch.zeros_like(dt)
    mean = aggregate_moments(q.log(), q.log(), dt, mask, score, 'duration')[0]
    zero = aggregate_moments(q.log(), q.log(), dt, mask, score, 'power', torch.tensor(0.))[0]
    torch.testing.assert_close(zero, mean)
    actual = aggregate_moments(q.log(), q.log(), dt, mask, score, 'power', torch.tensor(1.))[0].exp()
    expected = (dt[:, :, None] * q.square()).sum(1) / (dt[:, :, None] * q).sum(1)
    torch.testing.assert_close(actual, expected / expected.sum(-1, keepdim=True))


@pytest.mark.parametrize('pooling,decoder', [(p, 'vad') for p in ('duration','peak','class_peak','soft','sparse','power','sparse_proximity')] + [('soft','free'), ('sparse','free')])
def test_models_have_finite_gradients_normalized_contributions_and_ignore_padding(pooling, decoder):
    torch.manual_seed(42)
    model = TrajectoryPredictor(8, 16, layers=1, prototypes=coordinates(), pooling=pooling, decoder=decoder)
    dataset = TrajectoryVideos(Visual())
    x, mask, target, _ = next(iter(training.loader(dataset, {'seed': 42, 'training': {'batch_size': 3}})))
    logits, attention, details = model(x, mask, return_details=True)
    torch.testing.assert_close(logits.exp().sum(1), torch.ones(3))
    torch.testing.assert_close(attention.sum(-1), torch.ones(3, 21))
    torch.testing.assert_close(details['contributions'].sum(1), logits.exp())
    assert not attention.masked_select(~mask[:, None]).any()
    changed = x.clone(); changed[~mask] = float('nan')
    torch.testing.assert_close(logits, model(changed, mask)[0])
    distribution_loss(logits, target).backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert model.output.weight.grad.abs().sum() > 0
    if pooling in {'soft', 'sparse', 'sparse_proximity'}:
        assert sum(p.grad.abs().sum() for p in model.relevance.parameters()) > 0
    if decoder == 'vad':
        assert model.raw_temperature.grad.abs() > 0 and model.prototypes.requires_grad is False


def test_sparse_and_soft_share_initialization_and_only_time_free_control_is_permutation_invariant():
    torch.manual_seed(42)
    sparse = TrajectoryPredictor(8, 16, layers=1, prototypes=coordinates(), pooling='sparse')
    torch.manual_seed(42)
    soft = TrajectoryPredictor(8, 16, layers=1, prototypes=coordinates(), pooling='soft')
    for key in sparse.state_dict():
        torch.testing.assert_close(sparse.state_dict()[key], soft.state_dict()[key], rtol=0, atol=0)
    data = TrajectoryVideos(Visual())
    x, mask, _, _ = next(iter(training.loader(data, {'seed':42,'training':{'batch_size':3}})))
    changed = x.clone()
    for i, count in enumerate(data.counts):
        # Unequal durations must travel with their original visual observations.
        dt = torch.arange(1, int(count)+1).float()
        x[i, :count, -3] = dt
        x[i, :count, -2] = dt.cumsum(0) - dt
        x[i, :count, -1] = dt.cumsum(0)
        changed[i, :count] = permute_intervals(x[i, :count], torch.arange(int(count)-1,-1,-1))
    sparse.eval()
    assert not torch.allclose(sparse(x, mask)[0], sparse(changed, mask)[0])
    sparse.positional_encoding = False
    torch.testing.assert_close(sparse(x, mask)[0], sparse(changed, mask)[0], atol=2e-6, rtol=2e-6)


def test_prototype_scale_order_hash_and_permutation(prototype_asset):
    options = config('traj_vad_sparse', prototype_asset)['trajectory']['prototypes']
    original, info = load_prototypes(options)
    torch.testing.assert_close(original, coordinates())
    permuted, other = load_prototypes(dict(options, mapping='permuted'))
    assert not torch.equal(original, permuted)
    assert sorted(other['vad_word_for_each_class']) == sorted(REACTION_CLASSES)
    assert info['uses_text_features'] is False
    asset = json.loads(prototype_asset.read_text()); asset['class_order'].reverse()
    prototype_asset.write_text(json.dumps(asset))
    with pytest.raises(ValueError, match='checksum'):
        load_prototypes(options)
    options['asset_sha256'] = digest_file(prototype_asset)
    with pytest.raises(ValueError, match='class order'):
        load_prototypes(options)


@pytest.mark.parametrize('forbidden', ['description', 'evidence', 'vad_objective', 'pretrained_highlight'])
def test_pure_visual_guard_rejects_extra_input_paths(prototype_asset, forbidden):
    cfg = config('traj_vad_sparse', prototype_asset)
    cfg[forbidden] = {}
    with pytest.raises(ValueError, match='pure visual'):
        training.build_model(cfg, 'cpu')


def test_importance_correction_recovers_natural_objective_exactly():
    targets = torch.tensor([[.99,.01], [.9,.1], [.2,.8]])
    options = {'exponent':.5,'cap':3.}
    p = sampling_probabilities(targets.numpy(), options)
    assert p[2] > p[0]
    logits = torch.tensor([[.2,.4], [.1,.6], [.8,.1]], requires_grad=True)
    terms = torch.stack([corrected_kl(logits[i:i+1], targets[i:i+1], p, torch.tensor([i])) for i in range(3)])
    expected = distribution_loss(logits, targets)
    actual = (torch.tensor(p, dtype=terms.dtype) * terms).sum()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(torch.autograd.grad(actual, logits, retain_graph=True)[0], torch.autograd.grad(expected, logits)[0])


def test_trajectory_reload_resume_and_contribution_export(tmp_path, prototype_asset, monkeypatch):
    cfg = config('traj_vad_sparse_rare', prototype_asset)
    dataset = TrajectoryVideos(Visual())
    model, info = training.fit(cfg, dataset, dataset, tmp_path/'reference', 'cpu', 'source')
    assert info['checkpoint_reload_exact']
    prediction = training.predict(model, dataset, cfg, 'cpu', attention_output=tmp_path/'export')
    with np.load(tmp_path/'export/trajectory.npz') as saved:
        np.testing.assert_allclose(saved['prediction'], prediction, atol=1e-7)
        for i,(lo,hi) in enumerate(zip(saved['offsets'][:-1],saved['offsets'][1:])):
            np.testing.assert_allclose(saved['contributions'][lo:hi].sum(0), prediction[i], atol=2e-6)
    original = training.atomic_torch
    def interrupt(path, state):
        original(path, state)
        if path.name == 'last.pt' and state['epoch'] == 1:
            raise RuntimeError('synthetic interruption')
    monkeypatch.setattr(training, 'atomic_torch', interrupt)
    with pytest.raises(RuntimeError, match='synthetic interruption'):
        training.fit(cfg, dataset, dataset, tmp_path/'interrupted', 'cpu', 'source')
    monkeypatch.setattr(training, 'atomic_torch', original)
    training.fit(cfg, dataset, dataset, tmp_path/'resumed', 'cpu', 'source', tmp_path/'interrupted/last.pt')
    expected = torch.load(tmp_path/'reference/last.pt', weights_only=False)
    actual = torch.load(tmp_path/'resumed/last.pt', weights_only=False)
    assert actual['history'] == expected['history']
    for key in actual['model']:
        torch.testing.assert_close(actual['model'][key], expected['model'][key], rtol=0, atol=0)


def test_dino_features_use_cls_without_calling_text_aligned_method():
    class Encoder:
        def __call__(self, **inputs):
            class Output:
                last_hidden_state = torch.arange(24).reshape(2,3,4)
            return Output()
        def get_image_features(self, **inputs):
            raise AssertionError('Wrong encoder path')
    actual = encode_images(Encoder(), {}, {'kind':'dinov2'})
    torch.testing.assert_close(actual, torch.tensor([[0,1,2,3], [12,13,14,15]]))


def test_all_batch_configs_share_visual_cache_splits_and_optimization():
    from scripts import submit_trajectories
    assert list(VARIANTS) == submit_trajectories.VARIANTS
    assert SEEDS == submit_trajectories.SEEDS
    reference = load_config('configs/experiments/traj_vad_sparse.yaml')
    spec = cache_spec(reference)
    for name in VARIANTS:
        for seed in SEEDS:
            cfg = load_config(f'configs/experiments/{name}_s{seed}.yaml')
            assert cfg['seed'] == seed
            assert cache_spec(cfg) == spec
            options = copy.deepcopy(cfg['training']); options.pop('sampling', None)
            assert options == reference['training']
            assert cfg['encoder']['kind'] == 'dinov2'
            assert not any(k in cfg for k in ('description','evidence','vad_objective','pretrained_highlight'))
    for left,right in [('traj_vad_soft','traj_vad_sparse'), ('traj_free_soft','traj_free_sparse')]:
        a,b = [load_config(f'configs/experiments/{n}.yaml') for n in (left,right)]
        for item in (a,b):
            item.pop('experiment');item['trajectory'].pop('pooling')
        assert a == b


def test_launch_gate_checks_exact_source_variant_coverage_and_contributions(tmp_path, monkeypatch):
    from scripts import submit_trajectories as submit
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('V2R_ROOT', str(tmp_path))
    monkeypatch.setattr(submit, 'require_completed_cache', lambda job: None)
    out = tmp_path/'outputs/smoke/smoke_trajectory_3videos_123'
    out.mkdir(parents=True)
    source = {'source_sha256':'source','dirty':False}
    (tmp_path/'code_version.json').write_text(json.dumps(source))
    (out/'code_version.json').write_text(json.dumps(source))
    variants = {name: {'loss_decreased':True,'checkpoint_reload_exact':True,'optimizer_resume_passed':True,
                      'contribution_export_verified':name != 'dino_meanpool'} for name in VARIANTS}
    result = {'status':'completed','benchmark_result':False,'visual_only':True,
              'video_count':3,'frame_count':24,'variants':variants}
    (out/'metrics.json').write_text(json.dumps(result))
    submit.require_trajectory_smoke('123')
    result['variants']['traj_vad_sparse']['contribution_export_verified'] = False
    (out/'metrics.json').write_text(json.dumps(result))
    with pytest.raises(RuntimeError, match='contribution'):
        submit.require_trajectory_smoke('123')
    source['source_sha256'] = 'different'
    (tmp_path/'code_version.json').write_text(json.dumps(source))
    with pytest.raises(RuntimeError, match='exact source'):
        submit.require_trajectory_smoke('123')


def test_rare_diagnostics_groups_use_training_mass_only(tmp_path):
    from src.experiments.diagnostics import save_rare_diagnostics
    train = np.tile(np.arange(1,22,dtype=float)/231, (3,1))
    target = train[:, ::-1].copy()
    save_rare_diagnostics(tmp_path, train, target, target)
    result = json.loads((tmp_path/'rare_diagnostics.json').read_text())
    assert result['groups']['rare']['labels'] == REACTION_CLASSES[:6]
    assert result['brier_sum'] == 0 and result['class_probability_mae'] == [0] * 21
