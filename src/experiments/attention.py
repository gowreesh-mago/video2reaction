"""Save reaction attention with an explicit, verified link to source keyframes."""
from pathlib import Path

import numpy as np

from .registry import atomic_json
from .runtime import digest_file
from .taxonomy import REACTION_CLASSES


class AttentionRecorder:
    def __init__(self, dataset):
        self.ids = list(dataset.ids)
        self.records = dataset.frame_records
        self.counts = np.asarray(dataset.counts, dtype=np.int64)
        if (len(set(self.ids)) != len(self.ids) or len(self.records) != len(self.ids)
                or self.counts.shape != (len(self.ids),) or (self.counts < 1).any()):
            raise ValueError('Invalid attention sample inventory')
        for vid, count, record in zip(self.ids, self.counts, self.records):
            if record['sample_id'] != vid or len(record.get('frames', [])) != count:
                raise ValueError('Attention frame inventory differs from cached features')
        self.offsets = np.concatenate(([0], self.counts.cumsum()))
        self.weights = np.empty((self.offsets[-1], len(REACTION_CLASSES)), dtype=np.float32)
        self.seen = np.zeros(len(self.ids), dtype=bool)

    def add(self, indices, mask, attention):
        if attention is None:
            raise ValueError('Attention export requested from a model without attention')
        values = attention.detach().cpu().numpy()
        valid = mask.cpu().numpy()
        indices = indices.cpu().numpy()
        if values.shape != (len(indices), len(REACTION_CLASSES), valid.shape[1]):
            raise ValueError('Unexpected attention dimensions')
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError('Invalid attention values')
        for row, index in enumerate(indices):
            if index < 0 or index >= len(self.ids) or self.seen[index]:
                raise ValueError('Duplicate or unknown attention sample')
            count = self.counts[index]
            if not np.array_equal(valid[row], np.arange(valid.shape[1]) < count):
                raise ValueError('Attention mask differs from chronological cached frames')
            if (values[row, :, count:] != 0).any():
                raise ValueError('Attention assigned to padded frames')
            if not np.allclose(values[row, :, :count].sum(1), 1, atol=1e-6):
                raise ValueError('Attention is not normalized over frames')
            self.weights[self.offsets[index]:self.offsets[index + 1]] = values[row, :, :count].T
            self.seen[index] = True

    def save(self, directory):
        if not self.seen.all():
            raise ValueError('Cannot save incomplete attention')
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        entropies, top_positions, divergences = [], [], []
        for start, end in zip(self.offsets, self.offsets[1:]):
            weights = self.weights[start:end].astype(np.float64)
            log_weights = np.log(np.maximum(weights, 1e-30))
            entropies.append(-(weights * log_weights).sum(0))
            top_positions.append(weights.argmax(0))
            mixture = weights.mean(1, keepdims=True)
            # Generalized Jensen-Shannon divergence among 21 attention maps.
            divergences.append((weights * (log_weights - np.log(np.maximum(mixture, 1e-30)))).sum(0).mean())
        np.savez_compressed(directory / 'attention.npz', weights=self.weights, offsets=self.offsets,
                            sample_id=np.asarray(self.ids), class_order=np.asarray(REACTION_CLASSES),
                            entropy=np.asarray(entropies), top_frame_position=np.asarray(top_positions),
                            class_map_js=np.asarray(divergences))
        atomic_json(directory / 'attention_frames.json', {
            'schema': 1, 'layout': 'weights[offsets[i]:offsets[i+1], c] is the frame distribution for sample i and class c',
            'order': 'original chronological keyframe order; no frame shuffle',
            'class_order': REACTION_CLASSES, 'videos': self.records,
            'attention_sha256': digest_file(directory / 'attention.npz'),
            'interpretation': 'Model attention is a diagnostic, not a causal explanation of audience reactions.'})
