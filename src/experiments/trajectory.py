"""Visual VAD trajectories and duration-aware mixtures of moment probabilities.

No text encoder, description, audio or comment enters this model. Fixed NRC
coordinates define the distance decoder; all learned inputs are visual features.
"""
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .attention import AttentionRecorder
from .registry import atomic_json
from .runtime import digest_file
from .taxonomy import REACTION_CLASSES


POOLING = {'duration', 'peak', 'class_peak', 'soft', 'sparse', 'power', 'sparse_proximity'}


def load_prototypes(options):
    path = Path(os.environ['V2R_VAD_V2_FILE'])
    if digest_file(path) != options['asset_sha256']:
        raise ValueError('NRC v2.1 checksum mismatch')
    asset = json.loads(path.read_text())
    if (asset['class_order'] != REACTION_CLASSES or asset['scale'] != [-1, 1]
            or asset['source']['version'] != '2.1' or set(asset['labels']) != set(REACTION_CLASSES)):
        raise ValueError('NRC v2.1 scale, version or class order mismatch')
    for label, row in asset['labels'].items():
        if row['source_word'] != label or row['approximation'] is not False:
            raise ValueError('Exact NRC entries required')
    matrix = np.asarray([[asset['labels'][c][axis] for axis in ('valence', 'arousal', 'dominance')]
                         for c in REACTION_CLASSES], dtype=np.float32)
    if not np.isfinite(matrix).all() or (np.abs(matrix) > 1).any():
        raise ValueError('NRC v2 coordinates must be finite and in [-1,1]')
    permutation = np.arange(len(matrix))
    if options['mapping'] == 'permuted':
        permutation = np.random.default_rng(options['permutation_seed']).permutation(len(matrix))
    elif options['mapping'] != 'sourced':
        raise ValueError('Unknown prototype mapping')
    return torch.from_numpy(matrix[permutation]), {
        'source': asset['source'], 'asset_sha256': options['asset_sha256'],
        'scale': [-1, 1], 'mapping': options['mapping'], 'output_class_order': REACTION_CLASSES,
        'vad_word_for_each_class': [REACTION_CLASSES[i] for i in permutation],
        'uses_text_features': False, 'uses_language_teacher': False}


def seconds(value):
    parts = str(value).split(':')
    if len(parts) == 3:
        result = float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
    elif len(parts) == 1:
        result = float(parts[0])
    else:
        raise ValueError('Expected seconds or HH:MM:SS timestamp')
    if not math.isfinite(result) or result < 0:
        raise ValueError('Invalid timestamp')
    return result


def interval_packet(records):
    """Keep observed scene intervals; never extend a sampled keyframe across gaps."""
    values, previous_end = [], 0.
    for frame in records:
        start, end = seconds(frame['start_time']), seconds(frame['end_time'])
        if end <= start or start < previous_end - 1e-5:
            raise ValueError('Scene intervals must be positive and nonoverlapping')
        values.append([end - start, start, end])
        previous_end = end
    if not values:
        raise ValueError('An interval sequence must be nonempty')
    return torch.tensor(values, dtype=torch.float32)


class TrajectoryVideos:
    def __init__(self, visual):
        self.visual = visual
        for key in ('ids', 'counts', 'targets', 'frame_records'):
            setattr(self, key, getattr(visual, key))
        self.intervals = [interval_packet(r['frames']) for r in self.frame_records]
        if any(len(t) != n for t, n in zip(self.intervals, self.counts)):
            raise ValueError('Timing and visual frame inventory differ')

    def __len__(self):
        return len(self.visual)

    def __getitem__(self, i):
        x, target, index = self.visual[i]
        return torch.cat((x, self.intervals[i]), dim=-1), target, index


def permute_intervals(packet, permutation):
    """Move a scene with its duration, rebuilding a contiguous diagnostic timeline."""
    result = packet[permutation].clone()
    duration = result[:, -3]
    end = duration.cumsum(0)
    result[:, -2], result[:, -1] = end - duration, end
    return result


def temporal_weights(scores, duration, mask, mode):
    """Soft/sparse densities with respect to time, not the number of keyframes.

    Sparse density solves sum_i mu_i * relu(score_i - threshold) = 1,
    where mu_i = duration_i / total_duration. Splitting a constant interval
    into identical subintervals leaves the integrated contribution unchanged.
    """
    if (scores.ndim != 2 or scores.shape != mask.shape or duration.shape != mask.shape
            or mask.dtype != torch.bool or not mask.any(1).all()
            or not torch.isfinite(scores[mask]).all()
            or not torch.isfinite(duration[mask]).all() or (duration[mask] <= 0).any()):
        raise ValueError('Expected finite scores and positive durations on valid intervals')
    duration = duration.masked_fill(~mask, 0)
    mu = duration / duration.sum(1, keepdim=True)
    if mode == 'duration':
        return mu, mask.to(scores.dtype)
    shifted = scores - scores.masked_fill(~mask, -torch.inf).max(1, keepdim=True).values
    if mode == 'soft':
        log_mu = mu.clamp_min(torch.finfo(mu.dtype).tiny).log()
        weights = (shifted + log_mu).masked_fill(~mask, -torch.inf).softmax(1)
        return weights, torch.where(mask, weights / mu.clamp_min(torch.finfo(mu.dtype).tiny), 0.)
    if mode != 'sparse':
        raise ValueError('Unknown relevance normalization')
    ordered, positions = shifted.masked_fill(~mask, -torch.inf).sort(dim=1, descending=True)
    masses = mu.gather(1, positions)
    ordered = ordered.masked_fill(masses == 0, 0)
    cumulative_mass = masses.cumsum(1)
    candidates = ((masses * ordered).cumsum(1) - 1) / cumulative_mass.clamp_min(torch.finfo(mu.dtype).tiny)
    support = ((masses > 0) & (ordered > candidates)).sum(1, keepdim=True)
    threshold = candidates.gather(1, support - 1)
    density = (shifted - threshold).clamp_min(0).masked_fill(~mask, 0)
    weights = mu * density
    # Only correct roundoff; the analytic solution already integrates to one.
    normalizer = weights.sum(1, keepdim=True)
    return weights / normalizer, density / normalizer


def aggregate_moments(log_q, log_affinity, duration, mask, scores, pooling, power=None):
    """Return log P, conditional moment attribution, and actual mass contributions."""
    mode = pooling if pooling in {'soft', 'sparse'} else ('sparse' if pooling == 'sparse_proximity' else 'duration')
    weights, density = temporal_weights(scores, duration, mask, mode)
    log_w = weights.clamp_min(torch.finfo(weights.dtype).tiny).log().masked_fill(weights == 0, -torch.inf)
    if pooling in {'duration', 'soft', 'sparse'}:
        log_terms = log_w[:, :, None] + log_q
        log_p = torch.logsumexp(log_terms, dim=1)
    elif pooling == 'sparse_proximity':
        log_terms = log_w[:, :, None] + log_affinity
        raw = torch.logsumexp(log_terms, dim=1)
        log_p = raw.log_softmax(-1)
    elif pooling == 'peak':
        # One global closest approach to any prototype, with stable earliest ties.
        selected = log_affinity.max(-1).values.masked_fill(~mask, -torch.inf).argmax(1)
        weights = torch.zeros_like(weights).scatter(1, selected[:, None], 1.)
        density = weights / (duration.masked_fill(~mask, 0) / duration.masked_fill(~mask, 0).sum(1, keepdim=True)).clamp_min(1e-30)
        log_terms = log_q.masked_fill(~weights.bool()[:, :, None], -torch.inf)
        log_p = torch.logsumexp(log_terms, dim=1)
    elif pooling == 'class_peak':
        raw, selected = log_affinity.masked_fill(~mask[:, :, None], -torch.inf).max(1)
        log_p = raw.log_softmax(-1)
        support = torch.zeros_like(log_q, dtype=torch.bool).scatter(1, selected[:, None, :], True)
        log_terms = log_affinity.masked_fill(~support, -torch.inf)
    elif pooling == 'power':
        if power is None:
            raise ValueError('Power pooling requires a nonnegative exponent')
        log_terms = log_w[:, :, None] + (power + 1) * log_q
        raw = torch.logsumexp(log_terms, 1) - torch.logsumexp(log_w[:, :, None] + power * log_q, 1)
        log_p = raw.log_softmax(-1)
    else:
        raise ValueError('Unknown trajectory aggregation')
    attention = log_terms.softmax(1)
    contributions = attention * log_p.exp()[:, None, :]
    return log_p, attention.transpose(1, 2), {
        'moment_probabilities': log_q.exp(), 'log_affinity': log_affinity,
        'weights': weights, 'relevance_density': density, 'relevance_scores': scores,
        'contributions': contributions, 'duration_seconds': duration.masked_fill(~mask, 0)}


class TrajectoryPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, aggregation='trajectory', layers=2,
                 positional_encoding=True, *, prototypes, pooling='sparse', decoder='vad',
                 relevance_temperature=.2, temperature_init=.25):
        super().__init__()
        if aggregation != 'trajectory' or pooling not in POOLING or decoder not in {'vad', 'free'}:
            raise ValueError('Invalid trajectory model')
        if decoder == 'free' and pooling not in {'duration', 'soft', 'sparse', 'power'}:
            raise ValueError('Absolute VAD proximity requires a VAD decoder')
        if hidden_dim % 4 or layers < 1 or relevance_temperature <= 0 or not .05 < temperature_init < 2:
            raise ValueError('Invalid trajectory dimensions or temperature')
        if prototypes.shape != (21, 3) or not torch.isfinite(prototypes).all() or (prototypes.abs() > 1).any():
            raise ValueError('Expected 21 finite fixed VAD coordinates in [-1,1]')
        self.input_dim, self.aggregation, self.pooling, self.decoder = input_dim, aggregation, pooling, decoder
        self.hidden_dim, self.positional_encoding = hidden_dim, positional_encoding
        self.relevance_temperature = relevance_temperature
        self.register_buffer('prototypes', prototypes.clone())
        self.projection = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_dim))
        block = nn.TransformerEncoderLayer(hidden_dim, 4, hidden_dim * 4, dropout=0., batch_first=True, norm_first=True)
        self.temporal = nn.TransformerEncoder(block, layers, enable_nested_tensor=False)
        self.moment_hidden = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU())
        # Initialize common parameters before the variable-width output head.
        self.relevance = nn.Sequential(nn.Linear(hidden_dim, 32), nn.Tanh(), nn.Linear(32, 1, bias=False))
        self.output = nn.Linear(hidden_dim, 3 if decoder == 'vad' else 21)
        if decoder == 'vad':
            fraction = (temperature_init - .05) / 1.95
            self.raw_temperature = nn.Parameter(torch.tensor(math.log(fraction / (1 - fraction))))
        if pooling == 'power':
            self.raw_power = nn.Parameter(torch.tensor(math.log(1 / 7)))

    def forward(self, frames, mask, return_details=False):
        if (frames.ndim != 3 or frames.shape[-1] != self.input_dim + 3 or mask.shape != frames.shape[:2]
                or mask.dtype != torch.bool or not mask.any(1).all()):
            raise ValueError('Trajectory input needs visual features and duration/start/end timestamps')
        # Sanitizing padding prevents arbitrary padded values from entering attention.
        frames = frames.masked_fill(~mask[:, :, None], 0)
        duration, start, end = frames[..., -3:].unbind(-1)
        if not torch.isfinite(frames).all() or (duration[mask] <= 0).any():
            raise ValueError('Invalid visual/timing packet')
        h = self.projection(frames[..., :self.input_dim])
        if self.positional_encoding:
            midpoint = (start + end) * .5
            frequency = torch.exp(torch.arange(0, self.hidden_dim, 2, device=h.device, dtype=h.dtype)
                                  * (-math.log(10000) / self.hidden_dim))
            phase = midpoint[:, :, None] * frequency
            encoding = torch.stack((phase.sin(), phase.cos()), dim=-1).flatten(-2)
            h = h + encoding
        context = self.temporal(h, src_key_padding_mask=~mask)
        scores = self.relevance(context).squeeze(-1) / self.relevance_temperature
        output = self.output(self.moment_hidden(context))
        if self.decoder == 'vad':
            z = output.tanh()
            temperature = .05 + 1.95 * self.raw_temperature.sigmoid()
            log_affinity = -(z[:, :, None, :] - self.prototypes).square().sum(-1) / temperature
            log_q = log_affinity.log_softmax(-1)
        else:
            z = None
            log_q = output.log_softmax(-1)
            log_affinity = log_q  # No absolute VAD interpretation for free logits.
        power = 8 * self.raw_power.sigmoid() if self.pooling == 'power' else None
        log_p, attention, details = aggregate_moments(log_q, log_affinity, duration, mask, scores, self.pooling, power)
        if return_details:
            if z is not None:
                details['vad'] = z
            return log_p, attention, details
        return log_p, attention


class TrajectoryRecorder(AttentionRecorder):
    def __init__(self, dataset, model):
        super().__init__(dataset)
        self.details = {}
        self.model = model
        self.predictions = np.empty((len(self.ids), 21), dtype=np.float32)

    def add_details(self, indices, mask, attention, details, prediction):
        super().add(indices, mask, attention)
        for key, tensor in details.items():
            values = tensor.detach().cpu().numpy()
            if not np.isfinite(values[mask.cpu().numpy()]).all():
                raise ValueError(f'Nonfinite trajectory export: {key}')
            if key not in self.details:
                self.details[key] = np.empty((self.offsets[-1],) + values.shape[2:], dtype=np.float32)
            for row, index in enumerate(indices.tolist()):
                lo, hi = self.offsets[index:index + 2]
                self.details[key][lo:hi] = values[row, :hi-lo]
        self.predictions[indices.numpy()] = prediction.detach().cpu().numpy()

    def save(self, directory):
        super().save(directory)
        directory = Path(directory)
        support, mass_support, effective = [], [], []
        for i, (lo, hi) in enumerate(zip(self.offsets, self.offsets[1:])):
            mass = self.details['contributions'][lo:hi]
            np.testing.assert_allclose(mass.sum(0), self.predictions[i], atol=2e-6, rtol=2e-5)
            weights = self.details['weights'][lo:hi]
            support.append(int((weights > 0).sum()))
            moment_mass = mass.sum(1)
            mass_support.append(int((moment_mass > 0).sum()))
            effective.append(float(1 / np.square(moment_mass.astype(np.float64)).sum()))
        np.savez_compressed(directory / 'trajectory.npz', **self.details, offsets=self.offsets,
                            sample_id=np.asarray(self.ids), class_order=np.asarray(REACTION_CLASSES),
                            prediction=self.predictions)
        atomic_json(directory / 'trajectory_summary.json', {
            'schema': 1, 'pooling': self.model.pooling, 'decoder': self.model.decoder,
            'trajectory_sha256': digest_file(directory / 'trajectory.npz'),
            'frames_sha256': digest_file(directory / 'attention_frames.json'),
            'video_count': len(self.ids), 'frame_count': int(self.offsets[-1]),
            'mean_weight_support': float(np.mean(support)),
            'mean_contributing_moments': float(np.mean(mass_support)),
            'mean_effective_contributing_moments': float(np.mean(effective)),
            'mean_contributing_fraction': float(np.mean(np.asarray(mass_support) / self.counts)),
            'temperature': float(.05 + 1.95 * self.model.raw_temperature.detach().sigmoid()) if self.model.decoder == 'vad' else None,
            'power': float(8 * self.model.raw_power.detach().sigmoid()) if self.model.pooling == 'power' else None,
            'contributions_reconstruct_prediction': True,
            'interpretation': 'Piecewise-constant scene estimates from keyframes; not observed within-scene VAD or verified comment attribution.',
            'weights_interpretation': ('Common mixture weights' if self.model.pooling in {'duration', 'soft', 'sparse', 'peak'}
                                       else 'Base temporal weights; class-specific contributions also depend on the pooling formula')})
