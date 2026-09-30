"""Adversarial synthetic checks for the reviewed experiment controls and lifecycle."""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.freeze_release import freeze
from src.experiments.features import CachedVideos
from src.experiments.losses import distribution_loss
from src.experiments.models import ReactionPredictor
from src.experiments.registry import update_registry
from src.experiments.runtime import cache_directory, cache_spec, digest_file, fingerprint, load_config
from src.experiments.taxonomy import REACTION_CLASSES
from src.experiments import training
from src.experiments import features as feature_module


def test_temporal_control_matches_parameters_and_initialization():
    torch.manual_seed(42)
    ordered = ReactionPredictor(16, 32, aggregation='temporal').eval()
    torch.manual_seed(42)
    unordered = ReactionPredictor(16, 32, aggregation='temporal', positional_encoding=False).eval()
    for name, tensor in ordered.state_dict().items():
        torch.testing.assert_close(tensor, unordered.state_dict()[name], rtol=0, atol=0)
    x = torch.randn(3, 7, 16)
    mask = torch.ones(3, 7, dtype=torch.bool)
    permutation = torch.tensor([4, 0, 6, 2, 3, 5, 1])
    with torch.no_grad():
        torch.testing.assert_close(unordered(x, mask)[0], unordered(x[:, permutation], mask)[0], atol=1e-6, rtol=1e-6)
        assert not torch.allclose(ordered(x, mask)[0], ordered(x[:, permutation], mask)[0], atol=1e-5, rtol=1e-5)


def test_composite_ranking_direction_and_normalization():
    target = torch.tensor([[.6, .3, .1]])
    logits = torch.zeros(1, 3, requires_grad=True)
    ranking = distribution_loss(logits, target, 'composite', cosine_weight=0, ranking_weight=1) - distribution_loss(logits, target, 'kl')
    assert ranking.item() == pytest.approx(.1)
    torch.testing.assert_close(torch.autograd.grad(ranking, logits)[0], torch.tensor([[-2/3, 0., 2/3]]))
    uniform = torch.ones(1, 3) / 3
    no_pairs = distribution_loss(logits, uniform, 'composite', cosine_weight=0, ranking_weight=1)
    torch.testing.assert_close(no_pairs, distribution_loss(logits, uniform, 'kl'))


def test_completed_registry_cannot_be_corrupted_by_retry(tmp_path):
    registry = tmp_path / 'registry.json'
    update_registry(registry, 'run', status='completed', job_id='123')
    with pytest.raises(ValueError, match='immutable'):
        update_registry(registry, 'run', status='failed', job_id='123')
    with pytest.raises(ValueError, match='another SLURM'):
        update_registry(registry, 'run', status='submitted', job_id='999')
    update_registry(registry, 'run', status='submitted', job_id=None)
    saved = json.loads(registry.read_text())['experiments']['run']
    assert saved['status'] == 'completed' and saved['job_id'] == '123'


def test_frozen_release_survives_source_sync_and_rejects_tampering(tmp_path):
    source = tmp_path / 'code'
    source.mkdir()
    (source / '.venv').mkdir()
    (source / 'train.py').write_text('original source\n')
    manifest = {'dirty': False, 'git_commit': 'a' * 40, 'source_sha256': 'b' * 64,
                'files': {'train.py': digest_file(source / 'train.py')}}
    (source / 'code_version.json').write_text(json.dumps(manifest))
    release = freeze(source, tmp_path / 'releases')
    (source / 'train.py').write_text('new working copy\n')
    assert (release / 'train.py').read_text() == 'original source\n'
    (release / 'train.py').write_text('tampered\n')
    with pytest.raises(ValueError, match='modified'):
        freeze(source, tmp_path / 'releases')


class MockVideos:
    def __init__(self, seed, n):
        generator = torch.Generator().manual_seed(seed)
        self.frames = [torch.randn(2 + i % 3, 16, generator=generator) for i in range(n)]
        self.targets = torch.randn(n, 21, generator=generator).softmax(-1).numpy()

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        return self.frames[index], torch.from_numpy(self.targets[index]), index


def mock_config():
    cfg = load_config('configs/experiments/b1_meanpool.yaml')
    cfg['model'].update(input_dim=16, hidden_dim=32)
    cfg['training'].update(epochs=3, batch_size=4, patience=8)
    return cfg


def test_resume_matches_uninterrupted_training(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    cfg = mock_config()
    train, val = MockVideos(10, 9), MockVideos(11, 5)
    training.fit(cfg, train, val, tmp_path / 'reference', 'cpu', 'source')
    original_save = training.atomic_torch
    class Interruption(Exception):
        pass
    def interrupted_save(path, state):
        original_save(path, state)
        if Path(path).name == 'last.pt' and state['epoch'] == 1:
            raise Interruption()
    monkeypatch.setattr(training, 'atomic_torch', interrupted_save)
    with pytest.raises(Interruption):
        training.fit(cfg, train, val, tmp_path / 'interrupted', 'cpu', 'source')
    monkeypatch.setattr(training, 'atomic_torch', original_save)
    training.fit(cfg, train, val, tmp_path / 'resumed', 'cpu', 'source', tmp_path / 'interrupted' / 'last.pt')
    expected = torch.load(tmp_path / 'reference' / 'last.pt', weights_only=False)
    actual = torch.load(tmp_path / 'resumed' / 'last.pt', weights_only=False)
    assert expected['history'] == actual['history']
    for key in expected['model']:
        torch.testing.assert_close(expected['model'][key], actual['model'][key], atol=0, rtol=0)
    changed = copy.deepcopy(cfg)
    changed['loss']['kind'] = 'composite'
    with pytest.raises(ValueError, match='same resolved config'):
        training.fit(changed, train, val, tmp_path / 'mismatch', 'cpu', 'source', tmp_path / 'interrupted' / 'last.pt')


def test_cache_refuses_incomplete_wrong_ids_and_corruption(tmp_path, monkeypatch):
    monkeypatch.setenv('V2R_FEATURE_DIR', str(tmp_path))
    cfg = load_config('configs/experiments/b1_meanpool.yaml')
    root = cache_directory(cfg)
    directory = root / 'train'
    directory.mkdir(parents=True)
    np.save(directory / 'features.npy', np.zeros((2, 1152), dtype=np.float32))
    np.save(directory / 'image_sha256.npy', np.asarray(['x', 'y'], dtype='S64'))
    index = {'videos': [{'sample_id': 'mock'}], 'offsets': [0, 2], 'split_sha256': cfg['data']['split_sha256']['train']}
    (directory / 'index.json').write_text(json.dumps(index))
    manifest = {'status': 'completed', 'spec_sha256': fingerprint(cache_spec(cfg)), 'splits': {'train': {
        'features_sha256': digest_file(directory / 'features.npy'),
        'image_hashes_sha256': digest_file(directory / 'image_sha256.npy'),
        'index_sha256': digest_file(directory / 'index.json')}}}
    (root / 'manifest.json').write_text(json.dumps(manifest))
    rows = {'mock': {'reaction_outcome': {'reaction_distribution': {REACTION_CLASSES[0]: 1.}}}}
    assert len(CachedVideos(cfg, 'train', rows)) == 1
    with pytest.raises(ValueError, match='sample IDs'):
        CachedVideos(cfg, 'train', {'other': rows['mock']})
    with (directory / 'features.npy').open('ab') as stream:
        stream.write(b'corruption')
    with pytest.raises(ValueError, match='corrupted'):
        CachedVideos(cfg, 'train', rows)
    manifest['status'] = 'running'
    (root / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='incomplete'):
        CachedVideos(cfg, 'train', rows)


def test_declared_control_configs_change_only_the_intended_component():
    mean = load_config('configs/experiments/b1_meanpool.yaml')
    composite = load_config('configs/experiments/a5_distribution.yaml')
    assert mean['model'] == composite['model']
    for key in ('training', 'data', 'encoder', 'seed'):
        assert mean[key] == composite[key]
    ordered = load_config('configs/experiments/b2_temporal.yaml')
    unordered = load_config('configs/experiments/b2_set_control.yaml')
    ordered.pop('experiment'); unordered.pop('experiment')
    assert ordered['model'].pop('positional_encoding') is True
    assert unordered['model'].pop('positional_encoding') is False
    assert ordered == unordered


def test_feature_extraction_roundtrip_with_synthetic_images(tmp_path, monkeypatch):
    from PIL import Image
    monkeypatch.setenv('V2R_FEATURE_DIR', str(tmp_path / 'cache'))
    monkeypatch.setenv('V2R_FRAME_DIR', str(tmp_path / 'frames'))
    folder = tmp_path / 'frames' / 'mock'
    folder.mkdir(parents=True)
    (folder / 'index.csv').write_text('scene_number,start_frame\n0,0\n1,10\n2,20\n')
    for i in range(3):
        Image.new('RGB', (4, 4), (16 * (i + 1), 32, 64)).save(folder / f'{i + 1:03d}.jpg')
    class Inputs(dict):
        def to(self, device):
            return self
    class Processor:
        def to_dict(self):
            return {'synthetic_test_processor': True}
        def __call__(self, images, return_tensors):
            values = np.stack([np.asarray(image).mean((0, 1)) for image in images])
            return Inputs(pixel_values=torch.tensor(values, dtype=torch.float32))
    class Encoder:
        def get_image_features(self, pixel_values):
            return pixel_values
    monkeypatch.setattr(feature_module, 'load_frozen_encoder', lambda *args: (Processor(), Encoder()))
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'is_bf16_supported', lambda: True)
    cfg = load_config('configs/experiments/feature_cache.yaml')
    cfg['encoder'].update(feature_dim=3, batch_size=2, workers=0)
    rows = {'mock': {'reaction_outcome': {'reaction_distribution': {REACTION_CLASSES[0]: 1.}}}}
    out = tmp_path / 'output'
    out.mkdir()
    result = feature_module.prepare_features(cfg, {'train': rows}, out)
    assert result['splits']['train']['frames'] == 3
    dataset = CachedVideos(cfg, 'train', rows)
    values, target, index = dataset[0]
    assert values.shape == (3, 3) and target.shape == (21,) and index == 0
    assert values[0, 0] < values[1, 0] < values[2, 0]
    hashes = np.load(cache_directory(cfg) / 'train' / 'image_sha256.npy')
    assert hashes[0].decode() == digest_file(folder / '001.jpg')
    # A completed cache can be revisited without changing its data or identity.
    second = feature_module.prepare_features(cfg, {'train': rows}, out)
    assert result == second
