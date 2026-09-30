"""Collect per-run artifacts and reconcile terminal SLURM failures; run on cluster."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.experiments.registry import atomic_json, update_registry, utc_now


def collect(registry_path, output, reconcile=False):
    registry = json.loads(Path(registry_path).read_text())['experiments']
    runs = {}
    for name, record in registry.items():
        item = dict(record)
        if record.get('job_id'):
            state = subprocess.check_output(['sacct', '-j', str(record['job_id']), '-X', '-n', '-P',
                                             '--format=State,ExitCode,Elapsed'], text=True).strip()
            item['scheduler'] = state
            primary = state.split('|')[0].split()[0] if state else ''
            if reconcile and record['status'] in {'submitted', 'running'} and primary in {
                'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL', 'BOOT_FAIL', 'DEADLINE', 'PREEMPTED'}:
                update_registry(registry_path, name, status='failed', end_time=utc_now(), error=f'SLURM: {state}')
                item['status'] = 'failed'
        path = Path(record.get('output_dir', '/nonexistent')) / 'metrics.json'
        if path.is_file():
            item['artifact_metrics'] = json.loads(path.read_text())
        runs[name] = item
    result = {'collected_at': utc_now(), 'runs': runs}
    output = Path(output)
    atomic_json(output / 'run_summary.json', result)
    lines = ['# Experiment results', '', f"Collected: {result['collected_at']}", '',
             '| Run | Status | Job | KL | Cosine | MRR | F1@1 | F1@3 |',
             '|---|---|---|---:|---:|---:|---:|---:|']
    for name, run in runs.items():
        metrics = run.get('artifact_metrics', {}).get('test', {})
        scores = [f"{metrics[key]:.6f}" if key in metrics else '—' for key in ('kl', 'cosine', 'mrr', 'f1_top1', 'f1_top3')]
        lines.append(f"| {name} | {run['status']} | {run.get('job_id', '—')} | " + ' | '.join(scores) + ' |')
    lines += ['', 'Smoke scores are omitted from benchmark comparisons. Pending/failed runs have no inferred scores.', '']
    (output / 'run_summary.md').write_text('\n'.join(lines))
    print('\n'.join(lines))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--registry', default=os.environ.get('V2R_REGISTRY'))
    parser.add_argument('--output', default=os.environ.get('V2R_ROOT', '.') + '/results/collected')
    parser.add_argument('--reconcile', action='store_true')
    args = parser.parse_args()
    collect(args.registry, args.output, args.reconcile)
