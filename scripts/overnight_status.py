"""Write a cluster-side status snapshot without importing ML or reading video data."""
import json
import os
from pathlib import Path
import subprocess
from datetime import datetime, timezone


def main():
    root = Path(os.environ['V2R_ROOT'])
    records = root / 'results/submissions'
    jobs = {}
    for pattern in ('visual_trajectories-*.json', 'trajectory_overnight-*.json'):
        for path in records.glob(pattern):
            if '.intent.' not in path.name:
                jobs.update(json.loads(path.read_text()))
    ids = [str(j) for j in jobs.values() if str(j).isdigit()]
    states = {}
    if ids:
        raw = subprocess.check_output(['sacct', '-j', ','.join(ids), '-X', '-n', '-P',
            '--format=JobIDRaw,State,ExitCode,Elapsed,Start,End'], text=True)
        for line in raw.splitlines():
            row = line.strip().split('|')
            if len(row) >= 6:
                states[row[0]] = dict(zip(('state', 'exit_code', 'elapsed', 'start', 'end'), row[1:6]))
    results = []
    for name, job in jobs.items():
        row = {'name': name, 'job_id': str(job), **states.get(str(job), {'state': 'UNKNOWN'})}
        path = root / 'outputs/experiments' / f'{name}_{job}' / 'metrics.json'
        if path.exists():
            value = json.loads(path.read_text())
            row.update(benchmark_result=value.get('benchmark_result', False), metrics_path=str(path))
            if value.get('benchmark_result'):
                row.update(val_kl=value['val']['kl'], test_kl=value['test']['kl'], training=value.get('training'))
        row['completed_with_metrics'] = row.get('state') == 'COMPLETED' and row.get('exit_code') == '0:0' and path.exists()
        results.append(row)
    counts = {}
    for row in results:
        counts[row['state']] = counts.get(row['state'], 0) + 1
    smoke = []
    for path in (root / 'outputs/smoke').glob('smoke_trajectory_overnight_3videos_*/metrics.json'):
        value = json.loads(path.read_text())
        smoke.append({'path': str(path), 'status': value['status'], 'variant_count': len(value['variants'])})
    report = {'checked_at_utc': datetime.now(timezone.utc).isoformat(), 'counts': counts,
              'results': results, 'smoke_shards': smoke,
              'metrics_independently_recomputed': False,
              'note': 'Saved-run scores and scheduler states only; validation is the selection criterion. A missing full batch can indicate a failed or queued smoke/launcher.'}
    destination = root / 'results/overnight_status_20261007_0600'
    destination.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
    lines = ['# Overnight experiment status', '', report['checked_at_utc'], '',
             f'Scheduler counts: {counts}', '', report['note'], '',
             '| Experiment | Job | State | Validation KL | Test KL |', '|---|---:|---|---:|---:|']
    for row in results:
        lines.append(f"| {row['name']} | {row['job_id']} | {row['state']} | {row.get('val_kl', '')} | {row.get('test_kl', '')} |")
    destination.with_suffix('.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'report': str(destination), 'counts': counts, 'smoke_shards': smoke}, indent=2))


if __name__ == '__main__':
    main()
