"""Submit the fixed overnight matrix after every exact-source smoke shard passes."""
import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.submit_emotion import require_completed_cache
from scripts.submit_tier0 import submit_batch


def require_smokes(jobs, manifest):
    source = json.loads(Path('code_version.json').read_text())
    if source['dirty'] or len(set(jobs)) != manifest['smoke_shards'] or len(jobs) != manifest['smoke_shards']:
        raise ValueError('Require clean source and one distinct successful job per smoke shard')
    observed = set()
    shards = set()
    for job in jobs:
        if not job.isdigit():
            raise ValueError('Expected numeric smoke job ID')
        require_completed_cache(job)
        out = Path(os.environ['V2R_ROOT']) / 'outputs/smoke' / f'smoke_trajectory_overnight_3videos_{job}'
        identity = json.loads((out / 'code_version.json').read_text())
        result = json.loads((out / 'metrics.json').read_text())
        index = result['shard_index']
        if (identity['source_sha256'] != source['source_sha256'] or result['status'] != 'completed'
                or result['benchmark_result'] or not result['visual_only'] or result['video_count'] != 3
                or result['frame_count'] != 24 or result['batch'] != 'overnight'
                or result['shard_count'] != manifest['smoke_shards'] or result['epochs'] != manifest['smoke_epochs']
                or not 0 <= index < manifest['smoke_shards'] or index in shards):
            raise RuntimeError('Smoke source, shard or data contract mismatch')
        expected = set(manifest['variants'][index::manifest['smoke_shards']])
        if set(result['variants']) != expected or observed & expected:
            raise RuntimeError('Smoke variant coverage mismatch')
        for name, record in result['variants'].items():
            if not all(record[k] for k in ('loss_decreased', 'checkpoint_reload_exact', 'optimizer_resume_passed')):
                raise RuntimeError('Smoke numerical or lifecycle check failed')
            if 'dino_meanpool' not in name and not record['contribution_export_verified']:
                raise RuntimeError('Missing trajectory contribution verification')
        observed.update(expected)
        shards.add(index)
    if observed != set(manifest['variants']):
        raise RuntimeError('Incomplete smoke coverage')


def main(args):
    manifest = json.loads(Path('configs/trajectory_overnight.json').read_text())
    require_smokes(args.completed_smoke_jobs, manifest)
    require_completed_cache(args.completed_cache_job)
    names = [f'{name}_s{seed}' for seed in manifest['seeds'] for name in manifest['variants']]
    if len(names) != manifest['predictor_count']:
        raise RuntimeError('Incorrect matrix size')
    submit_batch(names, 'trajectory_overnight', max_parallel=manifest['max_parallel'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--completed-smoke-jobs', nargs='+', required=True)
    parser.add_argument('--completed-cache-job', required=True)
    main(parser.parse_args())
