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


def submit_batch(names, batch, dependencies=None, max_parallel=None):
    """Submit a fixed reviewed batch; each dependency names an earlier batch member."""
    dependencies = dependencies or {}
    if not names or len(set(names)) != len(names):
        raise ValueError('Batch must contain distinct experiments')
    if max_parallel is not None and (not isinstance(max_parallel, int) or max_parallel < 1):
        raise ValueError('Parallel job bound must be positive')
    for name, dependency in dependencies.items():
        if name not in names or dependency not in names[:names.index(name)]:
            raise ValueError('Dependencies must name an earlier experiment in this batch')
    root = Path(os.environ['V2R_ROOT'])
    manifest = json.loads(Path('code_version.json').read_text())
    if manifest['dirty']:
        raise ValueError('Only committed, frozen releases may be submitted')
    records = root / 'results' / 'submissions'
    records.mkdir(parents=True, exist_ok=True)
    (root / 'logs' / 'slurm').mkdir(parents=True, exist_ok=True)
    record = records / f"{batch}-{manifest['source_sha256']}.json"
    intent_path = record.with_suffix('.intent.json')
    with (records / 'submit.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        submissions = json.loads(record.read_text()) if record.exists() else {}
        intents = json.loads(intent_path.read_text()) if intent_path.exists() else {}
        # Preflight every resource request before submitting any part of the batch.
        for name in names:
            if name not in submissions:
                subprocess.run(['sbatch', '--test-only', f'slurm/{name}.sbatch'], check=True)
        print(command('squeue', '-u', os.environ['USER']), flush=True)
        for index, name in enumerate(names):
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
                conditions = []
                if name in dependencies:
                    conditions.append(f"afterok:{submissions[dependencies[name]]}")
                if max_parallel and index >= max_parallel:
                    previous = names[index - max_parallel]
                    if dependencies.get(name) != previous:
                        conditions.append(f'afterany:{submissions[previous]}')
                if conditions:
                    args += ['--dependency=' + ','.join(conditions), '--kill-on-invalid-dep=yes']
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
                            dependency=submissions[dependencies[name]] if name in dependencies else None,
                            throttle_after=submissions[names[index-max_parallel]] if max_parallel and index >= max_parallel else None)
            print(f'Submitted {name}: {job}', flush=True)
        print(json.dumps(submissions, indent=2))


def main(skip_prior=False):
    names = ['feature_cache', 'b0_prior', 'b1_meanpool', 'b2_temporal', 'b2_set_control', 'a5_distribution']
    if skip_prior:
        names.remove('b0_prior')
    dependencies = {name: 'feature_cache' for name in names if name not in {'feature_cache', 'b0_prior'}}
    submit_batch(names, 'tier0', dependencies)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-prior', action='store_true', help='Keep a completed B0 when retrying feature-dependent jobs')
    args = parser.parse_args()
    main(args.skip_prior)
