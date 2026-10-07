"""Recompute trajectory metrics and compare seed-averaged losses on the cluster."""
import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.analyze_tier0 import RUNS, analyze, kl_per_clip


def main(root):
    runs = dict(RUNS)
    for pattern in ('visual_trajectories-*.json', 'trajectory_overnight-*.json'):
        files = [p for p in (root / 'results/submissions').glob(pattern) if '.intent.' not in p.name]
        if len(files) != 1:
            raise ValueError(f'Expected one submitted batch for {pattern}, found {files}')
        for name, job in json.loads(files[0].read_text()).items():
            if name != 'dino_cache':
                runs[name] = f'{name}_{job}'
    if len(runs) != 269:
        raise ValueError('Expected 264 trajectory benchmarks and five earlier references')
    result = analyze(root / 'outputs/experiments', 10000, 20261007, runs=runs)
    groups, losses = defaultdict(list), defaultdict(list)
    for name, run in runs.items():
        if name in RUNS:
            continue
        directory = root / 'outputs/experiments' / run
        row = result['runs'][name]
        variant, seed = name.rsplit('_s', 1)
        with np.load(directory / 'test/predictions.npz', allow_pickle=False) as data:
            losses[variant].append(kl_per_clip(data['target_distribution'], data['predicted_distribution']))
            movie_ids = data['movie_id']
            rare = json.loads((directory / 'test/rare_diagnostics.json').read_text())
            np.testing.assert_allclose(np.abs(data['target_distribution'].astype(float) - data['predicted_distribution']).mean(0),
                                       rare['class_probability_mae'], atol=1e-12, rtol=0)
        row['rare_diagnostics'] = rare
        path = directory / 'test/trajectory_summary.json'
        if path.exists():
            row['trajectory_summary'] = json.loads(path.read_text())
            assert row['trajectory_summary']['contributions_reconstruct_prediction']
        groups[variant].append((int(seed), row))
    result['groups'] = {}
    for variant, rows in groups.items():
        assert sorted(s for s, _ in rows) == [42, 43, 44]
        record = {'seeds': [s for s, _ in rows], 'runs': [r['run_id'] for _, r in rows]}
        for split in ('val', 'test'):
            record[split + '_mean'] = {k: float(np.mean([r[split][k] for _, r in rows])) for k in rows[0][1][split]}
            record[split + '_sd'] = {k: float(np.std([r[split][k] for _, r in rows], ddof=1)) for k in rows[0][1][split]}
        record['test_kl_per_seed'] = {str(s): r['test']['kl'] for s, r in rows}
        record['rare_mae_mean'] = float(np.mean([r['rare_diagnostics']['groups']['rare']['mean_class_probability_mae'] for _, r in rows]))
        record['rare_top1_macro_f1_mean'] = float(np.mean([r['rare_diagnostics']['groups']['rare']['topk']['1']['macro_f1_all_group_classes'] for _, r in rows]))
        result['groups'][variant] = record
    result['validation_ranking'] = sorted(groups, key=lambda n: result['groups'][n]['val_mean']['kl'])
    movies, inverse, counts = np.unique(movie_ids, return_inverse=True, return_counts=True)
    rng = np.random.default_rng(20261007)
    weights = rng.multinomial(len(movies), np.full(len(movies), 1 / len(movies)), size=10000)
    denominators = weights @ counts
    pairs = [('on_dino_meanpool_wide', 'dino_meanpool'), ('on_vad_duration_free', 'dino_meanpool'),
             ('traj_free_soft', 'on_vad_duration_free'), ('traj_free_sparse', 'traj_free_soft'),
             ('on_free_sparse_shallow', 'traj_free_sparse'), ('on_vad_soft_tau010', 'dino_meanpool'),
             ('on_vad_soft_tau010', 'traj_vad_soft'), ('traj_vad_duration_rare', 'traj_vad_duration'),
             ('on_free_sparse_rare', 'traj_free_sparse'), ('traj_vad_sparse_permuted', 'traj_vad_sparse')]
    result['seed_averaged_paired_kl'] = {}
    for left, right in pairs:
        difference = np.mean(losses[left], axis=0) - np.mean(losses[right], axis=0)
        totals = np.bincount(inverse, weights=difference, minlength=len(movies))
        samples = (weights @ totals) / denominators
        result['seed_averaged_paired_kl'][f'{left} minus {right}'] = {
            'difference': float(difference.mean()), 'percentile_95_ci': np.quantile(samples, [.025, .975]).tolist()}
    result['checked_at_utc'] = datetime.now(timezone.utc).isoformat()
    result['benchmark_run_count'] = 264
    result['variant_count'] = 88
    result['checks'].append('Saved per-class probability MAE recomputes for all 264 new runs.')
    result['uncertainty_limits'] = 'Movie bootstrap of the mean loss across three fixed trained seeds; not an ensemble prediction, not an interval over training randomness, and not corrected for exploratory multiple comparisons.'
    result['trajectory_audit_limit'] = 'Contribution reconstruction flags checked; full trajectory tensors were not independently recomputed in this metrics audit.'
    destination = root / 'results/trajectory_results.json'
    destination.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'output': str(destination), 'checks': result['checks'],
                     'top_validation_groups': result['validation_ranking'][:5],
                     'seed_averaged_paired_kl': result['seed_averaged_paired_kl']}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    main(parser.parse_args().root)
