"""Numerical integrity checks for the offline scene explorer."""
from copy import deepcopy

import numpy as np
import pytest

from scripts.export_trajectory_examples import select_examples, validate_example


def example():
    # A deliberately nonuniform mixture; equal averaging would be incorrect.
    rng = np.random.default_rng(7)
    prototypes = rng.uniform(-1, 1, (21, 3))
    z = rng.uniform(-1, 1, (3, 3))
    tau = .09
    logits = -((z[:, None]-prototypes)**2).sum(-1)/tau
    q = np.exp(logits-logits.max(1, keepdims=True))
    q /= q.sum(1, keepdims=True)
    weights = np.array([.05, .8, .15])
    contributions = weights[:, None]*q
    row = dict(vad=z.tolist(), moment_probabilities=q.tolist(), weights=weights.tolist(),
               contributions=contributions.tolist(), prediction=contributions.sum(0).tolist(),
               target=(np.ones(21)/21).tolist(), dino_prediction=(np.ones(21)/21).tolist(),
               intervals_seconds=[[0, 2], [2, 3], [6, 11]])
    return row, prototypes, tau


def test_decoder_and_unequal_mixture_reconstruct_with_time_gaps():
    row, prototypes, tau = example()
    distances = validate_example(row, prototypes, tau)
    assert distances.shape == (3, 21)
    cumulative = np.cumsum(row['contributions'], axis=0)
    cumulative /= cumulative.sum(1, keepdims=True)
    np.testing.assert_allclose(cumulative[-1], row['prediction'])
    assert not np.allclose(np.mean(row['moment_probabilities'], axis=0), row['prediction'])


@pytest.mark.parametrize('change', ['class_permutation', 'wrong_temperature', 'bad_mixture', 'overlap'])
def test_detects_misleading_export(change):
    row, prototypes, tau = example()
    row = deepcopy(row)
    if change == 'class_permutation':
        prototypes = prototypes[::-1]
    elif change == 'wrong_temperature':
        tau *= 2
    elif change == 'bad_mixture':
        row['prediction'] = np.mean(row['moment_probabilities'], axis=0).tolist()
    else:
        row['intervals_seconds'][1] = [1, 3]
    with pytest.raises((ValueError, AssertionError)):
        validate_example(row, prototypes, tau)


def test_selection_is_unique_stable_and_ignores_single_scene_clips():
    ids = np.array([f'v{i:02}' for i in range(30)])
    errors = np.linspace(.1, 3, 30)
    rare = np.linspace(.3, 0, 30)  # Highest rarity overlaps low-error selections.
    counts = np.full(30, 4)
    counts[0] = 1
    chosen = select_examples(ids, errors, rare, counts)
    assert len(chosen) == len({i for i, _ in chosen}) == 10
    assert 0 not in [i for i, _ in chosen]
    assert [i for i, group in chosen if group == 'Low VAD error'] == [1, 2, 3]
    assert [i for i, group in chosen if group == 'High rare-reaction mass'] == [4, 5]
    perm = np.random.default_rng(2).permutation(30)
    shuffled = select_examples(ids[perm], errors[perm], rare[perm], counts[perm])
    assert [(ids[i], group) for i, group in chosen] == [(ids[perm[i]], group) for i, group in shuffled]


def test_kl_extremes_rank_entire_test_set_including_single_scene_videos():
    ids = np.array([f'v{i:02}' for i in range(35)])
    errors = np.arange(35, dtype=float)
    counts = np.full(35, 4)
    counts[0] = 1
    selected = select_examples(ids, errors, np.zeros(35), counts, 'kl_extremes')
    assert [i for i, _ in selected[:10]] == list(range(10))
    assert [i for i, _ in selected[10:]] == list(range(34, 24, -1))
    assert len({i for i, _ in selected}) == 20
    perm = np.random.default_rng(3).permutation(35)
    shuffled = select_examples(ids[perm], errors[perm], np.zeros(35), counts[perm], 'kl_extremes')
    assert [(ids[i], g) for i, g in selected] == [(ids[perm[i]], g) for i, g in shuffled]
