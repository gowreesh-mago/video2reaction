"""Sourced reaction VAD supervision with an explicit semantic-permutation control."""
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .runtime import digest_file
from .taxonomy import REACTION_CLASSES


def reaction_vad(options, device):
    path = Path(os.environ['V2R_REACTION_VAD_FILE'])
    if digest_file(path) != options['asset_sha256']:
        raise ValueError('Reaction VAD asset checksum mismatch')
    asset = json.loads(path.read_text())
    if set(asset['labels']) != set(REACTION_CLASSES):
        raise ValueError('Reaction VAD taxonomy differs from the benchmark')
    for label in REACTION_CLASSES:
        entry = asset['labels'][label]
        if entry['source_word'] != label or entry['approximation'] is not False:
            raise ValueError('Exact sourced reaction VAD entries are required')
    values = np.asarray([[asset['labels'][c][key] for key in ('valence', 'arousal', 'dominance')]
                         for c in REACTION_CLASSES], dtype=np.float32)
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError('Reaction VAD must be finite and in [0,1]')
    mapping = options['mapping']
    if mapping == 'sourced':
        permutation = np.arange(len(REACTION_CLASSES))
    elif mapping == 'permuted':
        permutation = np.random.default_rng(options['permutation_seed']).permutation(len(REACTION_CLASSES))
    else:
        raise ValueError('VAD mapping must be sourced or explicitly permuted')
    return torch.tensor(values[permutation], device=device), {
        'mapping': mapping, 'asset_sha256': options['asset_sha256'],
        'source': asset['source'], 'output_class_order': REACTION_CLASSES,
        'vad_word_for_each_class': [REACTION_CLASSES[i] for i in permutation],
        'permutation_seed': options['permutation_seed'] if mapping == 'permuted' else None}


def expected_vad_loss(predicted_vad, target, matrix):
    """Mean over clips of the squared Euclidean error (sum over the three axes)."""
    if predicted_vad is None or predicted_vad.shape != (len(target), 3):
        raise ValueError('Auxiliary VAD supervision needs a three-dimensional prediction')
    expected = target @ matrix
    return (predicted_vad - expected).square().sum(-1).mean()


def geometry_loss(classifier_weights, matrix, gamma):
    """Mean squared error on unordered, distinct class pairs; no diagonal terms."""
    if gamma <= 0 or matrix.shape != (len(classifier_weights), 3):
        raise ValueError('Geometry requires positive gamma and one VAD point per class')
    weights = F.normalize(classifier_weights, dim=-1)
    similarity = weights @ weights.T
    target_similarity = torch.exp(-gamma * (matrix[:, None] - matrix[None, :]).square().sum(-1))
    i, j = torch.triu_indices(len(matrix), len(matrix), offset=1, device=matrix.device)
    return (similarity[i, j] - target_similarity[i, j]).square().mean()


def vad_objective(model, predicted_vad, target, matrix, options):
    if options['weight'] <= 0:
        raise ValueError('VAD experiments require a positive, predeclared loss weight')
    if options['kind'] == 'auxiliary':
        return expected_vad_loss(predicted_vad, target, matrix)
    if options['kind'] == 'geometry':
        return geometry_loss(model.head[-1].weight, matrix, options['gamma'])
    raise ValueError('Unknown VAD objective')
