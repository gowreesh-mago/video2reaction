import copy
import json
from pathlib import Path

import pytest
import torch

from scripts.configure_trajectory_overnight import variants
from scripts import submit_trajectory_overnight as submit
from src.experiments import training
from src.experiments.losses import distribution_loss
from src.experiments.runtime import cache_spec, load_config


def test_overnight_matrix_is_distinct_visual_and_preserves_splits():
    manifest = json.loads(Path('configs/trajectory_overnight.json').read_text())
    assert manifest['variants'] == list(variants())
    assert len(manifest['variants']) * len(manifest['seeds']) == manifest['predictor_count'] == 216
    reference = load_config('configs/experiments/traj_vad_sparse.yaml')
    identities = set()
    for name in manifest['variants']:
        for seed in manifest['seeds']:
            cfg = load_config(f'configs/experiments/{name}_s{seed}.yaml')
            assert cfg['seed'] == seed and cache_spec(cfg) == cache_spec(reference)
            assert not any(k in cfg for k in ('description', 'evidence', 'vad_objective', 'pretrained_highlight'))
            assert cfg['training']['learning_rate'] == .0003
            assert cfg['training']['epochs'] == 50 and cfg['training']['selection_metric'] == 'kl'
            cfg.pop('experiment')
            identity = json.dumps(cfg, sort_keys=True)
            assert identity not in identities
            identities.add(identity)


@pytest.mark.parametrize('name', list(variants()))
def test_each_overnight_condition_has_finite_trainable_visual_path(name, monkeypatch):
    torch.set_num_threads(1)
    torch.manual_seed(42)
    monkeypatch.setattr(training, 'load_prototypes', lambda options: (torch.rand(21, 3) * 2 - 1, {}))
    cfg = load_config(f'configs/experiments/{name}.yaml')
    cfg['model']['input_dim'] = 8
    cfg['model']['hidden_dim'] = 16
    model = training.build_model(cfg, 'cpu')
    x = torch.randn(3, 4, 8)
    mask = torch.ones(3, 4, dtype=torch.bool)
    if 'trajectory' in cfg:
        times = torch.tensor([[1., 0., 1.], [2., 1., 3.], [3., 3., 6.], [1., 6., 7.]])
        x = torch.cat((x, times.expand(3, -1, -1)), dim=-1)
        logits, _, details = model(x, mask, return_details=True)
        torch.testing.assert_close(details['contributions'].sum(1), logits.exp())
    else:
        logits, _ = model(x, mask)
    loss = distribution_loss(logits, torch.randn(3, 21).softmax(-1))
    loss.backward()
    assert torch.isfinite(loss)
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert sum(g.abs().sum() for g in grads) > 0


def test_overnight_gate_requires_all_distinct_exact_source_shards(tmp_path, monkeypatch):
    manifest = json.loads(Path('configs/trajectory_overnight.json').read_text())
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('V2R_ROOT', str(tmp_path))
    monkeypatch.setattr(submit, 'require_completed_cache', lambda job: None)
    source = {'source_sha256': 'source', 'dirty': False}
    Path('code_version.json').write_text(json.dumps(source))
    jobs = ['100', '101', '102', '103']
    for index, job in enumerate(jobs):
        out = tmp_path / 'outputs/smoke' / f'smoke_trajectory_overnight_3videos_{job}'
        out.mkdir(parents=True)
        (out / 'code_version.json').write_text(json.dumps(source))
        result = {'status': 'completed', 'benchmark_result': False, 'visual_only': True,
                  'video_count': 3, 'frame_count': 24, 'batch': 'overnight', 'shard_index': index,
                  'shard_count': 4, 'epochs': 20,
                  'variants': {name: {'loss_decreased': True, 'checkpoint_reload_exact': True,
                      'optimizer_resume_passed': True, 'contribution_export_verified': 'dino_meanpool' not in name}
                      for name in manifest['variants'][index::4]}}
        (out / 'metrics.json').write_text(json.dumps(result))
    submit.require_smokes(jobs, manifest)
    with pytest.raises(ValueError):
        submit.require_smokes(jobs[:-1] + [jobs[0]], manifest)
    result['variants'].pop(next(iter(result['variants'])))
    (out / 'metrics.json').write_text(json.dumps(result))
    with pytest.raises(RuntimeError, match='coverage'):
        submit.require_smokes(jobs, manifest)
    source['source_sha256'] = 'changed'
    Path('code_version.json').write_text(json.dumps(source))
    with pytest.raises(RuntimeError, match='source'):
        submit.require_smokes(jobs, manifest)
