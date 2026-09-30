"""Submit the reviewed first batch once per frozen release, without login-node ML imports."""
import fcntl
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.experiments.registry import atomic_json, update_registry, utc_now


def command(*args):
    return subprocess.check_output(args, text=True).strip()


def main(skip_prior=False):
    root = Path(os.environ['V2R_ROOT'])
    manifest = json.loads(Path('code_version.json').read_text())
    names = ['feature_cache', 'b0_prior', 'b1_meanpool', 'b2_temporal', 'b2_set_control', 'a5_distribution']
    if skip_prior:
        names.remove('b0_prior')
    records = root / 'results' / 'submissions'
    records.mkdir(parents=True, exist_ok=True)
    (root / 'logs' / 'slurm').mkdir(parents=True, exist_ok=True)
    record = records / f"tier0-{manifest['source_sha256']}.json"
    intent_path = record.with_suffix('.intent.json')
    with (records / 'tier0.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        submissions = json.loads(record.read_text()) if record.exists() else {}
        intents = json.loads(intent_path.read_text()) if intent_path.exists() else {}
        # Preflight every resource request before submitting any part of the batch.
        for name in names:
            if name not in submissions:
                subprocess.run(['sbatch', '--test-only', f'slurm/{name}.sbatch'], check=True)
        print(command('squeue', '-u', os.environ['USER']), flush=True)
        for name in names:
            if name in submissions:
                print(f"Already recorded {name}: {submissions[name]}", flush=True)
                continue
            # A stable scheduler comment closes the retry gap after sbatch but before recording the ID.
            token = f"v2r:{manifest['source_sha256']}:{name}"
            live = command('squeue', '-h', '-u', os.environ['USER'], '-o', '%i|%k')
            recovered = [line.split('|')[0] for line in live.splitlines() if line.endswith('|' + token)]
            if not recovered and name in intents:
                accounted = command('sacct', '-u', os.environ['USER'], '-S', intents[name][:10],
                                    '-X', '-n', '-P', '--format=JobIDRaw,Comment%128')
                recovered = [line.split('|')[0].strip() for line in accounted.splitlines()
                             if len(line.split('|')) > 1 and line.split('|')[1].strip() == token]
                if not recovered:
                    raise RuntimeError(f'{name}: earlier submission outcome is unresolved; inspect SLURM before retrying')
            if len(recovered) > 1:
                raise RuntimeError(f'Duplicate live jobs for {name}')
            if recovered:
                job = recovered[0]
            else:
                args = ['sbatch', '--parsable', f'--comment={token}',
                        f'--output={root}/logs/slurm/{name}-%j.out']
                if name not in {'feature_cache', 'b0_prior'}:
                    args += [f"--dependency=afterok:{submissions['feature_cache']}", '--kill-on-invalid-dep=yes']
                intents[name] = utc_now()
                atomic_json(intent_path, intents)
                try:
                    job = command(*args, f'slurm/{name}.sbatch').split(';')[0]
                except subprocess.CalledProcessError:
                    del intents[name]
                    atomic_json(intent_path, intents)
                    raise
            if not job.isdigit():
                raise ValueError(f'Invalid SLURM job ID: {job}')
            submissions[name] = job
            atomic_json(record, submissions)
            update_registry(os.environ['V2R_REGISTRY'], f'{name}_{job}', status='submitted', job_id=job,
                            experiment_name=name, submitted_at=utc_now(), source_sha256=manifest['source_sha256'],
                            git_commit=manifest['git_commit'], release_dir=str(Path.cwd()),
                            output_dir=str(root / 'outputs' / 'experiments' / f'{name}_{job}'),
                            dependency=None if name in {'feature_cache', 'b0_prior'} else submissions['feature_cache'])
            print(f'Submitted {name}: {job}', flush=True)
        print(json.dumps(submissions, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-prior', action='store_true', help='Keep a completed B0 when retrying feature-dependent jobs')
    args = parser.parse_args()
    main(args.skip_prior)
