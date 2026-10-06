"""Training-only soft-label rarity sampling with an unbiased objective control."""
import numpy as np
import torch
from torch.nn import functional as F


def sampling_probabilities(targets, options):
    targets = np.asarray(targets, dtype=np.float64)
    if (targets.ndim != 2 or not np.isfinite(targets).all() or (targets < 0).any()
            or not np.allclose(targets.sum(1), 1, atol=1e-5)):
        raise ValueError('Sampling needs normalized soft targets from training')
    exponent, cap = options['exponent'], options['cap']
    if not 0 < exponent <= 1 or cap < 1:
        raise ValueError('Invalid rare sampling strength')
    prevalence = targets.mean(0)
    inverse = np.maximum(prevalence, 1e-12) ** (-exponent)
    weights = targets @ inverse
    weights = np.minimum(weights / weights.mean(), cap)
    return weights / weights.sum()


def corrected_kl(logits, target, probabilities, indices):
    per_clip = F.kl_div(logits.log_softmax(-1), target, reduction='none').sum(-1)
    p = torch.as_tensor(probabilities, dtype=logits.dtype, device=logits.device)[indices.to(logits.device)]
    return (per_clip / (len(probabilities) * p)).mean()
