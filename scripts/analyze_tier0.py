"""Validate saved Tier 0 predictions and estimate paired test KL uncertainty.

Uses generated run artifacts only; does not read frames or the feature cache.
Run in the locked cluster uv environment to preserve NumPy's top-k tie order.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.experiments.metrics import evaluate


RUNS = {
    'b0_prior': 'b0_prior_27382108',
    'b1_meanpool': 'b1_meanpool_27382294',
    'b2_temporal': 'b2_temporal_27382295',
    'b2_set_control': 'b2_set_control_27382296',
    'a5_distribution': 'a5_distribution_27382297',
}


def kl_per_clip(target, prediction):
    p, q = target.astype(np.float64), prediction.astype(np.float64)
    return (p * np.log(p / (q + 1e-10) + 1e-10)).sum(1)


def analyze(root, resamples, seed, runs=None, additional_pairs=()):
    runs = RUNS if runs is None else runs
    report = {'runs': {}, 'checks': [], 'paired_kl': {}}
    arrays, losses = {}, {}
    for name, run_id in runs.items():
        directory = root / run_id
        summary = json.loads((directory / 'metrics.json').read_text())
        expected_numpy = json.loads((directory / 'environment.json').read_text())['packages']['numpy']
        if np.__version__ != expected_numpy:
            raise ValueError(f'{run_id}: NumPy {expected_numpy} is required for matching top-k ties; '
                             f'found {np.__version__}. Use the locked cluster uv environment.')
        for split, expected_n in [('val', 1035), ('test', 2070)]:
            with np.load(directory / split / 'predictions.npz', allow_pickle=False) as archive:
                data = {key: archive[key] for key in archive.files}
            if len(data['sample_id']) != expected_n or len(np.unique(data['sample_id'])) != expected_n:
                raise ValueError(f'{run_id}/{split}: sample count or uniqueness mismatch')
            actual = evaluate(data['target_distribution'], data['predicted_distribution'])[0]
            for key, value in actual.items():
                np.testing.assert_allclose(value, summary[split][key], rtol=0, atol=1e-10)
            if name != 'b0_prior':
                for key in ('sample_id', 'movie_id', 'class_order', 'target_distribution', 'frame_count'):
                    np.testing.assert_array_equal(data[key], arrays[('b0_prior', split)][key])
            arrays[(name, split)] = data
        if name != 'b0_prior':
            history = json.loads((directory / 'history.json').read_text())
            training = summary['training']
            assert len(history) == training['epochs_run']
            selected = history[training['best_epoch'] - 1]['val']['kl']
            np.testing.assert_allclose(selected, training['val_selection_kl'], rtol=0, atol=1e-10)
            np.testing.assert_allclose(selected, summary['val']['kl'], rtol=0, atol=1e-10)
            assert training['checkpoint_reload_exact']
            assert (directory / 'best.pt').is_file() and (directory / 'last.pt').is_file()
        for split in ('val', 'test'):
            assert (directory / split / 'per_class.json').is_file()
        test = arrays[(name, 'test')]
        losses[name] = kl_per_clip(test['target_distribution'], test['predicted_distribution'])
        diagnostics = json.loads((directory / 'test/metrics.json').read_text())
        report['runs'][name] = {
            'run_id': run_id,
            'source_commit': json.loads((directory / 'code_version.json').read_text())['git_commit'],
            'predictions_sha256': hashlib.sha256((directory / 'test/predictions.npz').read_bytes()).hexdigest(),
            'training': summary['training'], 'val': summary['val'], 'test': summary['test'],
            'diagnostics': {key: value for key, value in diagnostics.items() if key != 'metrics'},
        }
    test = arrays[('b1_meanpool', 'test')]
    movies, inverse, counts = np.unique(test['movie_id'], return_inverse=True, return_counts=True)
    # Each draw resamples movies with replacement and retains every clip from
    # each drawn movie. The ratio retains the benchmark's clip-weighted mean.
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(len(movies), np.full(len(movies), 1 / len(movies)), size=resamples)
    denominators = weights @ counts

    def interval(label, difference):
        totals = np.bincount(inverse, weights=difference, minlength=len(movies))
        draws = (weights @ totals) / denominators
        report['paired_kl'][label] = {
            'difference': float(difference.mean()),
            'percentile_95_ci': np.quantile(draws, [.025, .975]).tolist(),
        }

    for candidate, reference in [('b1_meanpool', 'b0_prior'), ('b2_temporal', 'b1_meanpool'),
                                 ('b2_temporal', 'b2_set_control'), ('b2_set_control', 'b1_meanpool'),
                                 ('a5_distribution', 'b1_meanpool'), *additional_pairs]:
        interval(f'{candidate} minus {reference}', losses[candidate] - losses[reference])
    temporal = arrays[('b2_temporal', 'test')]
    interval('b2_shuffled minus b2_ordered',
             kl_per_clip(temporal['target_distribution'], temporal['shuffled_prediction']) - losses['b2_temporal'])
    report['uncertainty_method'] = {
        'method': 'paired movie-cluster bootstrap; clip-weighted mean KL; percentile intervals',
        'resamples': resamples, 'seed': seed, 'test_clips': len(inverse), 'test_movies': len(movies),
        'negative_difference': 'lower KL for the first model',
        'limits': 'Test-sample uncertainty only, conditional on trained models. No training-seed variance or multiple-comparison correction. Shuffle uses one saved permutation per clip.',
        'numpy_version': np.__version__,
    }
    report['checks'] = [
        'All validation/test metrics recompute from saved predictions to absolute tolerance 1e-10.',
        f'All {len(runs)} runs have identical sample IDs, movie IDs, class order, targets, and frame counts.',
        'Validation/test contain 1035/2070 unique clips respectively.',
        'Learned runs have best/last checkpoints, exact reload flags, and selected validation KL matching training history.',
        'Every run has saved validation/test per-class results.',
    ]
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outputs', type=Path, default=Path('outputs/experiments'))
    parser.add_argument('--output', type=Path, default=Path('research_notes/tier0_results.json'))
    parser.add_argument('--resamples', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=20260930)
    args = parser.parse_args()
    if args.resamples < 2:
        parser.error('--resamples must be at least 2')
    result = analyze(args.outputs, args.resamples, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'checks': result['checks'], 'uncertainty_method': result['uncertainty_method'],
                      'paired_kl': result['paired_kl']}, indent=2))
