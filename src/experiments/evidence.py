"""Label-independent peak selection, coarse emotion evidence, and frame diagnostics."""
import copy
import hashlib
from pathlib import Path

import numpy as np
import torch

from .emotion import load_emotion_logits, model_descriptor, probabilities, read_coarse_vad
from .registry import atomic_json


def peak_scores(q, vad_matrix, neutral):
    q, vad_matrix, neutral = map(lambda x: np.asarray(x, dtype=np.float64), (q, vad_matrix, neutral))
    if (q.ndim != 2 or q.shape[1] != 8 or vad_matrix.shape != (8, 3) or neutral.shape != (3,)
            or not all(np.isfinite(x).all() for x in (q, vad_matrix, neutral))
            or (q < 0).any() or not np.allclose(q.sum(1), 1, atol=1e-6)
            or (vad_matrix < 0).any() or (vad_matrix > 1).any() or (neutral < 0).any() or (neutral > 1).any()):
        raise ValueError('Peak scoring requires probabilities, sourced [0,1] VAD, and an explicit neutral point')
    vad = q @ vad_matrix
    distance = np.linalg.norm(vad - neutral, axis=1)
    entropy = -(q * np.log(np.maximum(q, 1e-30))).sum(1)
    confidence = np.clip(1 - entropy / np.log(q.shape[1]), 0, 1)
    return vad, {'arousal': vad[:, 1], 'distance': distance, 'confidence': confidence * distance}


def select_frames(count, method, k, scores, sample_id, seed):
    if count < 1 or (k is not None and (not isinstance(k, int) or isinstance(k, bool) or k < 1)):
        raise ValueError('Frame count and K must be positive integers')
    if method not in {'all', 'uniform', 'random', 'arousal', 'distance', 'confidence'}:
        raise ValueError('Unknown frame selection method')
    if method == 'all':
        if k is not None:
            raise ValueError('All-frame selection requires K=null')
        return np.arange(count)
    if k is None:
        raise ValueError('A subset selector needs K')
    k = min(k, count)
    if method == 'uniform':
        # Midpoint of each equally spaced bin; K=1 selects the middle frame.
        chosen = np.floor((np.arange(k) + .5) * count / k).astype(np.int64)
    elif method == 'random':
        identity = hashlib.sha256(f'{seed}:{sample_id}'.encode()).digest()
        rng = np.random.default_rng(int.from_bytes(identity[:8], 'big'))
        chosen = rng.choice(count, size=k, replace=False)
    else:
        values = np.asarray(scores[method])
        if values.shape != (count,) or not np.isfinite(values).all():
            raise ValueError('Invalid peak scores')
        # A tie chooses the earlier original keyframe, independent of NumPy's quicksort tie order.
        chosen = np.argsort(-values, kind='stable')[:k]
    return np.sort(chosen)


class EvidenceVideos:
    def __init__(self, cfg, visual):
        self.visual, self.cfg = visual, cfg
        self.ids, self.targets = visual.ids, visual.targets
        self.stratum_counts = visual.counts.copy()
        self.logits = load_emotion_logits(cfg, visual)
        self.q = probabilities(self.logits)
        self.vad, self.scores = peak_scores(self.q, read_coarse_vad(cfg), cfg['evidence']['neutral'])
        self.mode = cfg['evidence']['input_mode']
        self.evidence_dim = cfg['model'].get('evidence_dim', 0)
        self.fusion = cfg['model'].get('fusion', None)
        if self.mode not in {'none', 'logits', 'vad', 'both'}:
            raise ValueError('Unknown evidence input mode')
        if self.evidence_dim not in {0, 11} or (self.mode != 'none' and self.evidence_dim != 11):
            raise ValueError('Coarse evidence packets contain eight logits and three VAD coordinates')
        options = cfg['evidence']
        self.selected, self.frame_records, counts = [], [], []
        for i, vid in enumerate(self.ids):
            start, end = visual.offsets[i:i + 2]
            chosen = select_frames(end - start, options['selector'], options['k'],
                {key: values[start:end] for key, values in self.scores.items()}, vid, cfg['seed'])
            self.selected.append(chosen)
            record = copy.deepcopy(visual.frame_records[i])
            if self.fusion is None:
                record['frames'] = [record['frames'][j] for j in chosen]
            counts.append(len(record['frames']))
            self.frame_records.append(record)
        self.counts = np.asarray(counts)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        start, end = self.visual.offsets[i:i + 2]
        frames = np.array(self.visual.features[start:end], copy=True)
        if self.evidence_dim:
            auxiliary = np.zeros((end - start, 11), dtype=np.float32)
            if self.mode in {'logits', 'both'}:
                auxiliary[:, :8] = self.logits[start:end]
            if self.mode in {'vad', 'both'}:
                auxiliary[:, 8:] = self.vad[start:end]
            frames = np.concatenate((frames, auxiliary), axis=1)
        if self.fusion:
            indicator = np.zeros((end - start, 1), dtype=np.float32)
            indicator[self.selected[i]] = 1
            frames = np.concatenate((frames, indicator), axis=1)
        else:
            frames = frames[self.selected[i]]
        return torch.from_numpy(frames), torch.from_numpy(self.targets[i]), i

    def save_evidence(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        peak_weights = np.zeros(len(self.logits), dtype=np.float32)
        global_weights = np.empty_like(peak_weights)
        records = []
        for i, (vid, chosen) in enumerate(zip(self.ids, self.selected)):
            start, end = self.visual.offsets[i:i + 2]
            peak_weights[start + chosen] = 1 / len(chosen)
            global_weights[start:end] = 1 / (end - start)
            records.append(dict(sample_id=vid, selected_positions=chosen.tolist(),
                selected_frames=[self.visual.frame_records[i]['frames'][j] for j in chosen],
                total_frames=int(end - start)))
        np.savez_compressed(directory / 'frame_evidence.npz', sample_id=np.asarray(self.ids),
            offsets=np.asarray(self.visual.offsets), emotion_logits=self.logits, emotion_probability=self.q,
            expected_vad=self.vad, arousal=self.scores['arousal'], distance=self.scores['distance'],
            confidence_strength=self.scores['confidence'], peak_weights=peak_weights,
            global_weights=global_weights, coarse_class_order=np.asarray(model_descriptor(self.cfg)['class_order']))
        atomic_json(directory / 'selected_frames.json', dict(schema=1, evidence=self.cfg['evidence'],
            fusion=self.fusion, selection_uses_targets=False, videos=records,
            source_frame_inventory=self.visual.frame_records,
            interpretation='Frame emotion estimates and VAD are proxies, not observed audience reactions.'))
