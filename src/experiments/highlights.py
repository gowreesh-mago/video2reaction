"""Learned sparse highlight pooling and auditable frame-level outputs.

Sparsemax is the Euclidean projection onto the probability simplex:
Martins & Astudillo (2016), https://proceedings.mlr.press/v48/martins16.html.
No frame-level highlight annotations are used.
"""
from pathlib import Path
import json

import numpy as np
import torch

from .registry import atomic_json
from .runtime import digest_file


def masked_sparsemax(scores, mask):
    """Project each row onto the simplex over valid frames, with exact zeros."""
    if scores.ndim != 2 or scores.shape != mask.shape or mask.dtype != torch.bool:
        raise ValueError('Sparsemax requires matching B x T scores and boolean mask')
    if not mask.any(1).all() or not torch.isfinite(scores[mask]).all():
        raise ValueError('Each sequence needs finite scores and a valid frame')
    values = scores.float() if scores.dtype in {torch.float16, torch.bfloat16} else scores
    values = values - values.masked_fill(~mask, -torch.inf).max(1, keepdim=True).values
    # The valid maximum is now zero, so the simplex threshold is >= -1.
    # A padded value of -1 can never get positive mass; this avoids infinities
    # in cumulative sums and works even when padding sorts ahead of low scores.
    values = values.masked_fill(~mask, -1.)
    ordered = values.sort(dim=1, descending=True).values
    sums = ordered.cumsum(1)
    ranks = torch.arange(1, scores.shape[1] + 1, device=scores.device, dtype=values.dtype)[None]
    support = (1 + ranks * ordered > sums).sum(1, keepdim=True)
    threshold = (sums.gather(1, support - 1) - 1) / support
    return (values - threshold).clamp_min(0).masked_fill(~mask, 0).to(scores.dtype)


def export_highlights(directory, mode):
    """Export the exact shared weights used for prediction, with frame identities."""
    if mode not in {'sparsemax', 'softmax'}:
        raise ValueError('Unknown highlight normalization')
    directory = Path(directory)
    inventory = json.loads((directory / 'attention_frames.json').read_text())
    if inventory['attention_sha256'] != digest_file(directory / 'attention.npz'):
        raise ValueError('Attention checksum mismatch')
    records = []
    with np.load(directory / 'attention.npz', allow_pickle=False) as saved:
        for i, video in enumerate(inventory['videos']):
            begin, end = saved['offsets'][i:i + 2]
            maps = saved['weights'][begin:end]
            if not np.array_equal(maps, np.broadcast_to(maps[:, :1], maps.shape)):
                raise ValueError('Joint highlight pooling requires one shared frame map')
            if str(saved['sample_id'][i]) != video['sample_id'] or len(maps) != len(video['frames']):
                raise ValueError('Highlight frame identities do not match attention')
            weights = maps[:, 0].astype(np.float64)
            selected = np.flatnonzero(weights > 0)
            if len(selected) == 0 or not np.isclose(weights.sum(), 1, atol=1e-6):
                raise ValueError('Invalid highlight distribution')
            records.append({'sample_id': video['sample_id'], 'frame_count': len(weights),
                'weights': weights.tolist(), 'selected_positions': selected.tolist(),
                'selected_frames': [video['frames'][int(p)] for p in selected],
                'top_frame_position': int(weights.argmax()), 'support_count': len(selected),
                'support_fraction': len(selected) / len(weights),
                'effective_frame_count': float(1 / np.square(weights).sum()),
                'entropy': float(-(weights * np.log(np.maximum(weights, 1e-30))).sum())})
    atomic_json(directory / 'highlights.json', {'schema': 1, 'normalization': mode,
        'supervision': 'clip-level reaction distribution only; no highlight labels',
        'selection_rule': 'strictly positive pooling weight; adaptive count, no fixed K',
        'interpretation': 'Learned predictive frame selection, not validated human highlight detection or a causal explanation.',
        'attention_sha256': inventory['attention_sha256'],
        'frame_inventory_sha256': digest_file(directory / 'attention_frames.json'), 'videos': records})
    atomic_json(directory / 'highlight_statistics.json', {'videos': len(records), 'normalization': mode,
        'mean_support_count': float(np.mean([r['support_count'] for r in records])),
        'mean_support_fraction': float(np.mean([r['support_fraction'] for r in records])),
        'mean_effective_frame_count': float(np.mean([r['effective_frame_count'] for r in records])),
        'single_frame_fraction': float(np.mean([r['support_count'] == 1 for r in records])),
        'all_frame_fraction': float(np.mean([r['support_count'] == r['frame_count'] for r in records]))})
