"""Check sourced coordinate assignment, objective scaling, and the intervention's gradients."""
import copy
import json

import numpy as np
import pytest
import torch

from src.experiments.models import ReactionPredictor
from src.experiments.runtime import digest_file, load_config
from src.experiments.taxonomy import REACTION_CLASSES
from src.experiments.vad import expected_vad_loss, geometry_loss, reaction_vad
from src.experiments import training


def test_auxiliary_vad_is_expected_coordinates_and_squared_euclidean_loss():
    matrix = torch.tensor([[1., 0., .5], [0., 1., .5]])
    target = torch.tensor([[.75, .25], [.5, .5]])
    expected = torch.tensor([[.75, .25, .5], [.5, .5, .5]])
    assert expected_vad_loss(expected, target, matrix).item() == 0
    prediction = torch.zeros(2, 3, requires_grad=True)
    loss = expected_vad_loss(prediction, target, matrix)
    assert loss.item() == pytest.approx((.75**2 + .25**2 + .5**2 + 3*.5**2) / 2)
    loss.backward()
    torch.testing.assert_close(prediction.grad, -expected)


def test_geometry_uses_distinct_pairs_and_moves_cosine_toward_vad_similarity():
    weights = torch.eye(2, requires_grad=True)
    matrix = torch.tensor([[0., 0., 0.], [1., 0., 0.]])
    loss = geometry_loss(weights, matrix, gamma=2)
    assert loss.item() == pytest.approx(np.exp(-4))
    loss.backward()
    assert weights.grad[0, 1] < 0 and weights.grad[1, 0] < 0
    changed = weights.detach() - .01 * weights.grad
    assert geometry_loss(changed, matrix, gamma=2).item() < loss.item()


def test_auxiliary_head_preserves_initial_reaction_model_and_updates_shared_projection():
    torch.manual_seed(42)
    base = ReactionPredictor(16, 32)
    torch.manual_seed(42)
    augmented = ReactionPredictor(16, 32, auxiliary_vad=True)
    for name, value in base.state_dict().items():
        torch.testing.assert_close(value, augmented.state_dict()[name], rtol=0, atol=0)
    x, mask = torch.randn(3, 4, 16), torch.ones(3, 4, dtype=torch.bool)
    logits, _, predicted = augmented(x, mask, return_vad=True)
    torch.testing.assert_close(logits, base(x, mask)[0], rtol=0, atol=0)
    target = torch.ones(3, 21) / 21
    expected_vad_loss(predicted, target, torch.ones(21, 3) / 2).backward()
    assert augmented.projection[1].weight.grad.norm() > 0
    assert augmented.vad_head.weight.grad.norm() > 0
    assert augmented.head[-1].weight.grad is None


def test_vad_mapping_control_preserves_coordinates_and_records_label_assignment(tmp_path, monkeypatch):
    values = np.random.default_rng(17).uniform(size=(21, 3)).astype(np.float32)
    asset = {'source': {'synthetic_fixture': True}, 'labels': {c: dict(
        zip(['valence', 'arousal', 'dominance'], map(float, point)), source_word=c, approximation=False)
        for c, point in zip(REACTION_CLASSES, values)}}
    path = tmp_path / 'synthetic_vad.json'
    path.write_text(json.dumps(asset))
    monkeypatch.setenv('V2R_REACTION_VAD_FILE', str(path))
    options = dict(mapping='sourced', permutation_seed=271828, asset_sha256=digest_file(path))
    actual, provenance = reaction_vad(options, 'cpu')
    np.testing.assert_array_equal(actual.numpy(), values)
    assert provenance['vad_word_for_each_class'] == REACTION_CLASSES
    options['mapping'] = 'permuted'
    permuted, provenance = reaction_vad(options, 'cpu')
    assert provenance['vad_word_for_each_class'] != REACTION_CLASSES
    assert sorted(map(tuple, permuted.tolist())) == sorted(map(tuple, actual.tolist()))
    for i, word in enumerate(provenance['vad_word_for_each_class']):
        np.testing.assert_array_equal(permuted[i].numpy(), values[REACTION_CLASSES.index(word)])
    path.write_text(path.read_text() + ' ')
    with pytest.raises(ValueError, match='checksum'):
        reaction_vad(options, 'cpu')


@pytest.mark.parametrize('name', ['b_vad_aux', 'b_vad_geometry'])
def test_vad_configs_change_only_semantic_mapping_for_the_control(name):
    source = load_config(f'configs/experiments/{name}.yaml')
    control = load_config(f'configs/experiments/{name}_permuted.yaml')
    source.pop('experiment'); control.pop('experiment')
    assert source['vad_objective'].pop('mapping') == 'sourced'
    assert control['vad_objective'].pop('mapping') == 'permuted'
    assert source == control


@pytest.mark.parametrize('name', ['b_vad_aux', 'b_vad_geometry'])
def test_training_consumes_vad_objective_and_reloads_checkpoint(tmp_path, monkeypatch, name):
    cfg = load_config(f'configs/experiments/{name}.yaml')
    cfg['model'].update(input_dim=16, hidden_dim=32)
    cfg['training'].update(epochs=2, batch_size=3)
    class Videos:
        frames = torch.randn(3, 4, 16)
        targets = torch.eye(21)[:3].numpy()
        def __len__(self):
            return 3
        def __getitem__(self, i):
            return self.frames[i], torch.from_numpy(self.targets[i]), i
    matrix = torch.rand(21, 3)
    monkeypatch.setattr(training, 'reaction_vad', lambda options, device: (matrix, {'synthetic': True}))
    model, info = training.fit(cfg, Videos(), Videos(), tmp_path, 'cpu', 'synthetic-source')
    history = json.loads((tmp_path / 'history.json').read_text())
    assert info['checkpoint_reload_exact']
    for epoch in history:
        assert epoch['train_vad_loss_unweighted'] > 0
        assert epoch['train_objective'] == pytest.approx(epoch['train_reaction_loss'] +
            cfg['vad_objective']['weight'] * epoch['train_vad_loss_unweighted'], abs=1e-6)
