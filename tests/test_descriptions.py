"""Synthetic checks for text retention, cache integrity, and modality controls."""
import copy
import json

import numpy as np
import pytest
import torch

from src.experiments import descriptions as d
from src.experiments.models import ReactionPredictor
from src.experiments.runtime import load_config, seed_everything
from src.experiments.training import fit, predict


class Tokenizer:
    def num_special_tokens_to_add(self, pair=False):
        return 1

    def encode(self, text, add_special_tokens=False):
        assert text == text.lower() and not add_special_tokens
        return [ord(c) - 96 for c in text if c != ' ']

    def build_inputs_with_special_tokens(self, tokens):
        return tokens + [100]

    def pad(self, chunks, padding, max_length, return_attention_mask, return_tensors):
        assert padding == 'max_length' and not return_attention_mask and return_tensors == 'pt'
        return {'input_ids': torch.tensor([r['input_ids'] + [0] * (max_length-len(r['input_ids'])) for r in chunks])}


class Encoder:
    def __init__(self):
        self.calls = []

    def get_text_features(self, input_ids):
        self.calls.append(input_ids.clone())
        content = input_ids.masked_fill(input_ids == 100, 0).float()
        return torch.stack([content.sum(1), content.max(1).values], dim=-1)


def options():
    return {'max_length': 4, 'feature_dim': 2, 'chunk_batch_size': 2, 'video_batch_size': 1}


def test_all_tokens_retained_and_weighted_without_padding_or_eos_weight():
    model = Encoder()
    actual, records = d.encode_descriptions(['ABCDEFG', 'B'], Tokenizer(), model, options(), 'cpu')
    np.testing.assert_allclose(actual, [[(3*6+3*15+7)/7, (3*3+3*6+7)/7], [2, 2]])
    assert records == [{'content_tokens': 7, 'chunks': 3}, {'content_tokens': 1, 'chunks': 1}]
    assert torch.cat(model.calls).tolist() == [[1, 2, 3, 100], [4, 5, 6, 100], [7, 100, 0, 0], [2, 100, 0, 0]]


def cache_fixture(tmp_path, monkeypatch):
    cfg = {'description_encoder': options(), 'data': {'split_sha256': {'train': 'synthetic'}}}
    rows = {'a': {'clip_description': 'ABCD'}, 'b': {'clip_description': 'EF'}}
    monkeypatch.setenv('V2R_DESCRIPTION_FEATURE_DIR', str(tmp_path / 'cache'))
    monkeypatch.setattr(d, 'load_text_encoder', lambda *args: (Tokenizer(), Encoder()))
    return cfg, rows


def test_cache_recovers_only_uncommitted_rows_and_completed_reuse_needs_no_model(tmp_path, monkeypatch):
    cfg, rows = cache_fixture(tmp_path, monkeypatch)
    real = d.encode_descriptions
    calls = []
    def fail_second(texts, *args):
        calls.extend(texts)
        if len(calls) == 2:
            raise RuntimeError('synthetic interruption')
        return real(texts, *args)
    monkeypatch.setattr(d, 'encode_descriptions', fail_second)
    with pytest.raises(RuntimeError, match='interruption'):
        d.prepare_description_cache(cfg, {'train': rows}, tmp_path, 'cpu')
    root = d.description_directory(cfg)
    progress = json.loads((root / 'train/progress.json').read_text())
    assert progress['next_video'] == 1
    array = np.load(root / 'train/features.npy', mmap_mode='r+')
    array[1] = 999  # an uncommitted write must be recomputed
    array.flush()
    resumed = []
    def record(texts, *args):
        resumed.extend(texts)
        return real(texts, *args)
    monkeypatch.setattr(d, 'encode_descriptions', record)
    d.prepare_description_cache(cfg, {'train': rows}, tmp_path, 'cpu')
    assert resumed == ['EF']
    values = d.load_description_features(cfg, 'train', rows)
    np.testing.assert_allclose(values, [[(3*6+4)/4, (3*3+4)/4], [11, 6]])
    monkeypatch.setattr(d, 'load_text_encoder', lambda *args: pytest.fail('Completed cache must not load weights'))
    d.prepare_description_cache(cfg, {'train': rows}, tmp_path, 'cpu')
    changed = copy.deepcopy(rows)
    changed['b']['clip_description'] = 'different content'
    with pytest.raises(ValueError, match='changed'):
        d.load_description_features(cfg, 'train', changed)
    array[0] = -999
    array.flush()
    with pytest.raises(ValueError, match='Corrupted'):
        d.load_description_features(cfg, 'train', rows)


def test_cache_rejects_changed_committed_prefix(tmp_path, monkeypatch):
    cfg, rows = cache_fixture(tmp_path, monkeypatch)
    real = d.encode_descriptions
    def fail(texts, *args):
        if texts == ['EF']:
            raise RuntimeError('interrupted')
        return real(texts, *args)
    monkeypatch.setattr(d, 'encode_descriptions', fail)
    with pytest.raises(RuntimeError):
        d.prepare_description_cache(cfg, {'train': rows}, tmp_path, 'cpu')
    values = np.load(d.description_directory(cfg) / 'train/features.npy', mmap_mode='r+')
    values[0] = 0
    values.flush()
    with pytest.raises(ValueError, match='prefix'):
        d.prepare_description_cache(cfg, {'train': rows}, tmp_path, 'cpu')


@pytest.mark.parametrize('value', [None, '', '   '])
def test_missing_descriptions_are_not_silently_replaced(value):
    with pytest.raises(ValueError, match='empty'):
        d.description_inventory({'a': {'clip_description': value}}, 'synthetic')


class Visual:
    ids, split = ['a', 'b', 'c'], 'train'
    counts = np.array([2, 3, 4])
    targets = np.eye(21, dtype=np.float32)[:3]
    def __init__(self):
        self.frames = [torch.randn(int(n), 16) for n in self.counts]
        self.reads = []
    def __len__(self):
        return 3
    def __getitem__(self, i):
        self.reads.append(i)
        return self.frames[i], torch.from_numpy(self.targets[i]), i


@pytest.mark.parametrize('mode', ['text', 'visual_text', 'visual_visual', 'text_text'])
def test_description_modality_isolation_and_original_frame_strata(monkeypatch, mode):
    visual = Visual()
    text = np.random.default_rng(7).normal(size=(3, 16)).astype(np.float32)
    monkeypatch.setattr(d, 'load_description_features', lambda *args: text)
    dataset = d.DescriptionVideos({'description': {'mode': mode}}, visual, dict.fromkeys(visual.ids))
    frames, target, index = dataset[1]
    assert index == 1
    torch.testing.assert_close(target, torch.from_numpy(visual.targets[1]))
    np.testing.assert_array_equal(dataset.stratum_counts, visual.counts)
    if mode in {'text', 'text_text'}:
        assert visual.reads == []
        torch.testing.assert_close(frames[:, :16], torch.from_numpy(text[1:2]))
        assert dataset.counts.tolist() == [1, 1, 1]
    else:
        torch.testing.assert_close(frames[:, :16], visual.frames[1])
    if mode == 'visual_text':
        torch.testing.assert_close(frames[:, 16:], torch.from_numpy(text[1:2]).expand(3, -1))
    elif mode in {'visual_visual', 'text_text'}:
        torch.testing.assert_close(frames[:, :16], frames[:, 16:])


def test_fusion_both_branches_have_gradients_and_ignore_padding():
    model = ReactionPredictor(16, 32, description_dim=16)
    frames = torch.randn(2, 3, 32, requires_grad=True)
    mask = torch.tensor([[True, True, False], [True, True, True]])
    logits = model(frames, mask)[0]
    logits.square().sum().backward()
    assert frames.grad[..., :16].norm() > 0 and frames.grad[..., 16:].norm() > 0
    assert frames.grad[0, 2].norm() == 0
    changed = frames.detach().clone()
    changed[0, 2] = 123
    torch.testing.assert_close(logits, model(changed, mask)[0], rtol=0, atol=0)


def test_configs_preserve_b1_settings_and_match_fusion_capacity():
    baseline = load_config('configs/experiments/b1_meanpool.yaml')
    names = ['description_only', 'visual_description', 'description_visual_control', 'description_text_control']
    states = []
    for name in names:
        cfg = load_config(f'configs/experiments/{name}.yaml')
        for key in ('data', 'encoder', 'training', 'loss', 'evaluation', 'seed'):
            assert cfg[key] == baseline[key]
        seed_everything(42)
        model = ReactionPredictor(**cfg['model'])
        if name != 'description_only':
            states.append(model.state_dict())
        else:
            assert cfg['model'] == baseline['model']
    for state in states[1:]:
        assert state.keys() == states[0].keys()
        for key in state:
            torch.testing.assert_close(state[key], states[0][key], rtol=0, atol=0)


@pytest.mark.parametrize('name', ['description_only', 'visual_description', 'description_visual_control', 'description_text_control'])
def test_production_fit_and_reload_with_description_inputs(tmp_path, monkeypatch, name):
    cfg = load_config(f'configs/experiments/{name}.yaml')
    cfg['model'].update(input_dim=16, hidden_dim=32)
    if cfg['model'].get('description_dim'):
        cfg['model']['description_dim'] = 16
    cfg['training'].update(epochs=2, batch_size=3)
    text = np.random.default_rng(9).normal(size=(3, 16)).astype(np.float32)
    monkeypatch.setattr(d, 'load_description_features', lambda *args: text)
    visual = Visual()
    dataset = d.DescriptionVideos(cfg, visual, dict.fromkeys(visual.ids))
    model, info = fit(cfg, dataset, dataset, tmp_path, 'cpu', 'synthetic')
    assert info['checkpoint_reload_exact']
    predicted = predict(model, dataset, cfg, 'cpu')
    shuffled = predict(model, dataset, cfg, 'cpu', shuffle_frames=True)
    np.testing.assert_allclose(predicted.sum(1), 1, atol=1e-6)
    np.testing.assert_allclose(predicted, shuffled, atol=2e-6)
