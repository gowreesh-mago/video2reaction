"""Synthetic checks for image/cache identity, target-independent selection, and controls."""
import copy
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

from src.experiments import emotion, evidence
from src.experiments.data import frame_index
from src.experiments.features import CachedVideos
from src.experiments.models import ReactionPredictor
from src.experiments.registry import atomic_json
from src.experiments.runtime import cache_directory, cache_spec, digest_file, fingerprint, load_config
from src.experiments.taxonomy import REACTION_CLASSES


@pytest.fixture
def synthetic_cache(tmp_path, monkeypatch):
    cfg = load_config('configs/experiments/emotion_cache.yaml')
    cfg['encoder']['feature_dim'] = cfg['model']['input_dim'] = 16
    cfg['emotion_encoder'].update(batch_size=2, workers=0, flush_batches=1)
    monkeypatch.setenv('V2R_FEATURE_DIR', str(tmp_path / 'visual'))
    monkeypatch.setenv('V2R_EMOTION_FEATURE_DIR', str(tmp_path / 'emotion'))
    monkeypatch.setenv('V2R_FRAME_DIR', str(tmp_path / 'images'))
    rows, records, hashes, offsets = {}, [], [], [0]
    for i, count in enumerate([3, 5]):
        vid = f'synthetic_{i}'
        rows[vid] = {'reaction_outcome': {'reaction_distribution': {REACTION_CLASSES[i]: 1.}}}
        folder = tmp_path / 'images' / vid
        folder.mkdir(parents=True)
        (folder / 'index.csv').write_text('scene_number,start_frame,start_time\n' +
            ''.join(f'{j},{j*10},{j*.5}\n' for j in range(count)))
        for j in range(count):
            image_path = folder / f'{j+1:03d}.jpg'
            Image.new('RGB', (8, 8), (10 + j*25, 40 + i*30, 80)).save(image_path)
            hashes.append(digest_file(image_path))
        frames, _ = frame_index(tmp_path / 'images', vid)
        records.append(dict(sample_id=vid, total_frames=count,
            frames=[{k: v for k, v in row.items() if k != 'path'} for row in frames],
            index_sha256=digest_file(folder / 'index.csv')))
        offsets.append(offsets[-1] + count)
    root = cache_directory(cfg)
    directory = root / 'train'
    directory.mkdir(parents=True)
    values = np.random.default_rng(7).normal(size=(offsets[-1], 16)).astype(np.float32)
    np.save(directory / 'features.npy', values)
    np.save(directory / 'image_sha256.npy', np.asarray(hashes, dtype='S64'))
    atomic_json(directory / 'index.json', dict(videos=records, offsets=offsets,
                split_sha256=cfg['data']['split_sha256']['train']))
    complete = {field: digest_file(directory / filename) for filename, field in [
        ('features.npy', 'features_sha256'), ('image_sha256.npy', 'image_hashes_sha256'), ('index.json', 'index_sha256')]}
    atomic_json(root / 'manifest.json', dict(status='completed', spec_sha256=fingerprint(cache_spec(cfg)),
                                           splits={'train': complete}))
    class Encoder(torch.nn.Module):
        def forward(self, x):
            r, g, b = x.mean((-1, -2)).unbind(-1)
            return torch.stack([r, g, b, -r, -g, -b, r + b, g + b], -1)
    def transform(image):
        return torch.tensor(np.array(image), dtype=torch.float32).permute(2, 0, 1) / 255
    monkeypatch.setattr(emotion, 'load_emotion_encoder', lambda *args: (transform, Encoder()))
    # Explicitly synthetic VAD fixture, never used by an actual-data entrypoint.
    vad = np.random.default_rng(2).uniform(size=(8, 3))
    monkeypatch.setattr(evidence, 'read_coarse_vad', lambda cfg: vad)
    output = tmp_path / 'out'
    output.mkdir()
    return cfg, {'train': rows}, output


def test_emotion_cache_resumes_after_committed_prefix_and_rejects_corruption(synthetic_cache, monkeypatch):
    cfg, splits, out = synthetic_cache
    original = emotion.atomic_json
    class Interrupted(Exception):
        pass
    def stop(path, value):
        original(path, value)
        if Path(path).name == 'progress.json' and value['next_frame'] == 2:
            raise Interrupted()
    monkeypatch.setattr(emotion, 'atomic_json', stop)
    with pytest.raises(Interrupted):
        emotion.prepare_emotion_cache(cfg, splits, out, device='cpu')
    path = emotion.emotion_directory(cfg) / 'train' / 'logits.npy'
    partial = np.load(path, mmap_mode='r+')
    prefix = partial[:2].copy()
    partial[2:] = 999  # Simulates uncommitted writes after the durable resume cursor.
    partial.flush()
    monkeypatch.setattr(emotion, 'atomic_json', original)
    result = emotion.prepare_emotion_cache(cfg, splits, out, device='cpu')
    assert result['splits']['train']['frames'] == 8
    visual = CachedVideos(cfg, 'train', splits['train'])
    logits = emotion.load_emotion_logits(cfg, visual)
    np.testing.assert_array_equal(logits[:2], prefix)
    assert logits.max() < 2
    np.testing.assert_allclose(emotion.probabilities(logits).sum(1), 1, atol=1e-12)
    monkeypatch.setattr(emotion, 'load_emotion_encoder', lambda *a: pytest.fail('Completed cache must not reload weights'))
    assert emotion.prepare_emotion_cache(cfg, splits, out, device='cpu') == result
    with path.open('ab') as stream:
        stream.write(b'corrupted')
    with pytest.raises(ValueError, match='Corrupted'):
        emotion.load_emotion_logits(cfg, visual)


def test_emotion_cache_rejects_images_that_differ_from_visual_cache(synthetic_cache, monkeypatch):
    cfg, splits, out = synthetic_cache
    import os
    Image.new('RGB', (8, 8), 'red').save(Path(os.environ['V2R_FRAME_DIR']) / 'synthetic_0' / '001.jpg')
    with pytest.raises(ValueError, match='Image content differs'):
        emotion.prepare_emotion_cache(cfg, splits, out, device='cpu')


def test_scores_and_subset_controls_have_the_declared_behavior():
    vad = np.full((8, 3), .5)
    vad[0] = [1., .9, .5]
    q = np.stack([np.eye(8)[0], np.ones(8) / 8])
    expected, scores = evidence.peak_scores(q, vad, [.5, .5, .5])
    np.testing.assert_array_equal(expected[0], vad[0])
    assert scores['arousal'][0] == .9
    assert scores['distance'][0] == pytest.approx(np.sqrt(.5**2 + .4**2))
    assert scores['confidence'][0] == scores['distance'][0]
    assert scores['confidence'][1] == pytest.approx(0, abs=1e-14)
    scores = {key: np.array([0., 5., 5., 3., 1., 2., 4.]) for key in scores}
    for method in ['arousal', 'distance', 'confidence']:
        np.testing.assert_array_equal(evidence.select_frames(7, method, 1, scores, 'id', 42), [1])
        np.testing.assert_array_equal(evidence.select_frames(7, method, 3, scores, 'id', 42), [1, 2, 6])
    np.testing.assert_array_equal(evidence.select_frames(7, 'uniform', 1, {}, 'id', 42), [3])
    for method in ['arousal', 'distance', 'confidence', 'uniform', 'random']:
        np.testing.assert_array_equal(evidence.select_frames(7, method, 8, scores, 'id', 42), np.arange(7))
        for k in [1, 2, 4]:
            chosen = evidence.select_frames(7, method, k, scores, 'id', 42)
            assert len(chosen) == len(np.unique(chosen)) == k
            assert (np.diff(chosen) > 0).all()
            np.random.seed(999)
            np.testing.assert_array_equal(chosen, evidence.select_frames(7, method, k, scores, 'id', 42))


def test_selection_is_target_independent_and_export_keeps_original_frame_identity(synthetic_cache):
    cfg, splits, out = synthetic_cache
    emotion.prepare_emotion_cache(cfg, splits, out, device='cpu')
    cfg['evidence'].update(selector='arousal', k=2)
    visual = CachedVideos(cfg, 'train', splits['train'])
    dataset = evidence.EvidenceVideos(cfg, visual)
    original = [x.copy() for x in dataset.selected]
    visual.targets[:] = np.roll(visual.targets, 5, axis=1)
    changed = evidence.EvidenceVideos(cfg, visual)
    for before, after in zip(original, changed.selected):
        np.testing.assert_array_equal(before, after)
    assert dataset.counts.tolist() == [2, 2]
    assert dataset.stratum_counts.tolist() == [3, 5]
    dataset.save_evidence(out / 'diagnostics')
    records = json.loads((out / 'diagnostics' / 'selected_frames.json').read_text())['videos']
    saved = np.load(out / 'diagnostics' / 'frame_evidence.npz', allow_pickle=False)
    for i, record in enumerate(records):
        frames, _, _ = dataset[i]
        begin, end = visual.offsets[i:i + 2]
        np.testing.assert_array_equal(frames.numpy(), visual.features[begin:end][original[i]])
        assert record['selected_frames'] == [visual.frame_records[i]['frames'][j] for j in original[i]]
        assert saved['peak_weights'][begin:end].sum() == pytest.approx(1)


def test_zero_evidence_retains_baseline_predictions_and_common_gradients():
    torch.manual_seed(42)
    baseline = ReactionPredictor(16, 32)
    torch.manual_seed(42)
    augmented = ReactionPredictor(16, 32, evidence_dim=11)
    x = torch.randn(2, 4, 16)
    mask = torch.ones(2, 4, dtype=torch.bool)
    p = baseline(x, mask)[0]
    q = augmented(torch.cat([x, torch.zeros(2, 4, 11)], -1), mask)[0]
    torch.testing.assert_close(p, q, rtol=0, atol=0)
    p.sum().backward(); q.sum().backward()
    for name, parameter in baseline.named_parameters():
        torch.testing.assert_close(parameter.grad, dict(augmented.named_parameters())[name].grad, rtol=0, atol=0)


def test_fusion_controls_share_parameters_but_respond_to_their_own_evidence():
    models = {}
    for fusion in ['global_peak', 'global_global', 'peak_peak']:
        torch.manual_seed(42)
        models[fusion] = ReactionPredictor(16, 32, fusion=fusion).eval()
    reference = models['global_peak'].state_dict()
    for model in models.values():
        for name, value in model.state_dict().items():
            torch.testing.assert_close(reference[name], value, rtol=0, atol=0)
    x = torch.randn(2, 5, 17)
    x[..., -1] = 0
    x[:, 0, -1] = 1
    mask = torch.ones(2, 5, dtype=torch.bool)
    changed = x.clone()
    changed[:, 2:, 0] += 9
    with torch.no_grad():
        for name, model in models.items():
            before, after = model(x, mask)[0], model(changed, mask)[0]
            if name == 'peak_peak':
                torch.testing.assert_close(before, after, rtol=0, atol=0)
            else:
                assert not torch.allclose(before, after)
        x[..., -1] = 1  # When every frame is selected all three conditions coincide.
        for model in models.values():
            torch.testing.assert_close(model(x, mask)[0], models['global_peak'](x, mask)[0], rtol=0, atol=0)


def test_experiment_configs_preserve_controls_and_visual_cache_identity():
    from scripts.submit_emotion import PRIMARY, CURVE
    baseline = load_config('configs/experiments/b1_meanpool.yaml')
    configs = {name: load_config(f'configs/experiments/{name}.yaml') for name in PRIMARY + CURVE}
    assert len(configs) == 26
    for name, cfg in configs.items():
        for key in ['seed', 'data', 'encoder', 'training', 'loss', 'evaluation']:
            assert cfg[key] == baseline[key], (name, key)
        assert cfg['experiment']['name'] == name
        assert Path(f'slurm/{name}.sbatch').is_file()
        assert cache_spec(cfg) == cache_spec(baseline)
    for k in [1, 2, 4, 8]:
        peers = [configs[f'd_{method}_k{k}'] for method in ['arousal', 'distance', 'confidence', 'uniform', 'random']]
        assert all(cfg['model'] == baseline['model'] for cfg in peers)
        cleaned = []
        for cfg in peers:
            cfg = copy.deepcopy(cfg)
            cfg.pop('experiment')
            cfg['evidence'].pop('selector')
            cleaned.append(cfg)
        assert all(cfg == cleaned[0] for cfg in cleaned)
