"""Check that E's intervention and exported frame identities match its claim."""
import json

import numpy as np
import pytest
import torch

from src.experiments.attention import AttentionRecorder
from src.experiments.models import ReactionPredictor
from src.experiments.runtime import load_config
from src.experiments.training import predict


def test_shared_queries_match_capacity_initialization_and_use_all_queries():
    torch.manual_seed(42)
    separate = ReactionPredictor(16, 32, aggregation='query')
    torch.manual_seed(42)
    shared = ReactionPredictor(16, 32, aggregation='shared_query')
    for name, value in separate.state_dict().items():
        torch.testing.assert_close(value, shared.state_dict()[name], rtol=0, atol=0)
    x = torch.randn(2, 5, 16)
    mask = torch.tensor([[True] * 3 + [False] * 2, [True] * 5])
    separate_logits, separate_maps = separate(x, mask)
    shared_logits, shared_maps = shared(x, mask)
    expected = separate_maps.mean(1, keepdim=True).expand_as(separate_maps)
    torch.testing.assert_close(shared_maps, expected, rtol=0, atol=0)
    assert not torch.allclose(separate_logits, shared_logits)
    shared_logits.square().sum().backward()
    assert (shared.queries.grad.norm(dim=1) > 0).all()
    # Both models aggregate a set; positional/temporal claims are out of scope.
    permutation = torch.tensor([2, 0, 4, 1, 3])
    for model in (separate, shared):
        logits, maps = model(x, mask)
        changed_logits, changed_maps = model(x[:, permutation], mask[:, permutation])
        torch.testing.assert_close(logits, changed_logits, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(maps[:, :, permutation], changed_maps, atol=1e-6, rtol=1e-6)


class AttentionVideos:
    def __init__(self):
        self.ids = ['clip_a', 'clip_b', 'clip_c']
        self.counts = np.array([2, 5, 3])
        rng = torch.Generator().manual_seed(15)
        self.frames = [torch.randn(int(n), 16, generator=rng) for n in self.counts]
        self.targets = np.ones((3, 21), dtype=np.float32) / 21
        self.frame_records = [dict(sample_id=vid, total_frames=int(n),
            frames=[dict(index=j, start_frame=str(j * 10), start_time=str(j * .5)) for j in range(n)])
            for vid, n in zip(self.ids, self.counts)]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        return self.frames[i], torch.from_numpy(self.targets[i]), i


def test_prediction_attention_roundtrip_and_exact_frame_alignment(tmp_path):
    dataset = AttentionVideos()
    cfg = load_config('configs/experiments/e_reaction_query.yaml')
    cfg['training']['batch_size'] = 2
    model = ReactionPredictor(16, 32, aggregation='query').eval()
    plain = predict(model, dataset, cfg, 'cpu')
    exported = predict(model, dataset, cfg, 'cpu', attention_output=tmp_path)
    np.testing.assert_array_equal(plain, exported)
    saved = np.load(tmp_path / 'attention.npz', allow_pickle=False)
    np.testing.assert_array_equal(saved['offsets'], [0, 2, 7, 10])
    np.testing.assert_array_equal(saved['sample_id'], dataset.ids)
    for i, frame in enumerate(dataset.frames):
        with torch.no_grad():
            _, attention = model(frame[None], torch.ones(1, len(frame), dtype=torch.bool))
        begin, end = saved['offsets'][i:i + 2]
        np.testing.assert_allclose(saved['weights'][begin:end].T, attention[0].numpy(), atol=1e-6)
        np.testing.assert_array_equal(saved['top_frame_position'][i], saved['weights'][begin:end].argmax(0))
    inventory = json.loads((tmp_path / 'attention_frames.json').read_text())
    assert inventory['videos'] == dataset.frame_records
    with pytest.raises(ValueError, match='original chronological'):
        predict(model, dataset, cfg, 'cpu', shuffle_frames=True, attention_output=tmp_path)


def test_recorder_rejects_misaligned_missing_duplicate_and_padded_attention(tmp_path):
    dataset = AttentionVideos()
    recorder = AttentionRecorder(dataset)
    with pytest.raises(ValueError, match='incomplete'):
        recorder.save(tmp_path)
    with pytest.raises(ValueError, match='without attention'):
        recorder.add(torch.tensor([0]), torch.ones(1, 2, dtype=torch.bool), None)
    with pytest.raises(ValueError, match='padded'):
        recorder.add(torch.tensor([0]), torch.tensor([[True, True, False]]), torch.ones(1, 21, 3) / 3)
    recorder.add(torch.tensor([0]), torch.ones(1, 2, dtype=torch.bool), torch.ones(1, 21, 2) / 2)
    with pytest.raises(ValueError, match='Duplicate'):
        recorder.add(torch.tensor([0]), torch.ones(1, 2, dtype=torch.bool), torch.ones(1, 21, 2) / 2)
    dataset.frame_records[0]['frames'].pop()
    with pytest.raises(ValueError, match='inventory differs'):
        AttentionRecorder(dataset)


def test_query_configs_change_only_the_attention_sharing_rule():
    specific = load_config('configs/experiments/e_reaction_query.yaml')
    shared = load_config('configs/experiments/e_shared_query_control.yaml')
    specific.pop('experiment'); shared.pop('experiment')
    assert specific['model'].pop('aggregation') == 'query'
    assert shared['model'].pop('aggregation') == 'shared_query'
    assert specific == shared
