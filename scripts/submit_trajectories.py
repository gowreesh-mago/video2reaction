"""Submit the reviewed visual-only batch only after its exact source passes smoke."""
import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.submit_emotion import require_completed_cache
from scripts.submit_tier0 import submit_batch

# Keep this list dependency-free: submission runs on the login node.
VARIANTS = [
    'traj_vad_duration', 'traj_vad_peak', 'traj_vad_class_peak', 'traj_vad_soft', 'traj_vad_sparse',
    'traj_vad_power', 'traj_vad_sparse_proximity', 'traj_vad_sparse_permuted',
    'traj_vad_duration_rare', 'traj_vad_sparse_rare', 'traj_vad_duration_importance', 'traj_vad_sparse_importance',
    'traj_free_soft', 'traj_free_sparse', 'traj_vad_sparse_no_time', 'dino_meanpool']
SEEDS = (42, 43, 44)


def require_trajectory_smoke(job):
    if not job.isdigit():
        raise ValueError('Expected numeric smoke job ID')
    require_completed_cache(job)
    out = Path(os.environ['V2R_ROOT']) / 'outputs/smoke' / f'smoke_trajectory_3videos_{job}'
    source = json.loads((out/'code_version.json').read_text())
    current = json.loads(Path('code_version.json').read_text())
    result = json.loads((out/'metrics.json').read_text())
    if (source['source_sha256'] != current['source_sha256'] or current['dirty']
            or result['status'] != 'completed' or result['benchmark_result'] or not result['visual_only']
            or result['video_count'] != 3 or result['frame_count'] != 24 or set(result['variants']) != set(VARIANTS)):
        raise RuntimeError('Trajectory smoke must pass for this exact source and experiment set')
    for name, record in result['variants'].items():
        if not all(record[key] for key in ('loss_decreased','checkpoint_reload_exact','optimizer_resume_passed')):
            raise RuntimeError('A trajectory smoke check failed')
        if name != 'dino_meanpool' and not record['contribution_export_verified']:
            raise RuntimeError('Missing trajectory contribution audit')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--completed-smoke-job', required=True)
    args = parser.parse_args()
    require_trajectory_smoke(args.completed_smoke_job)
    runs = [f'{name}_s{seed}' for seed in SEEDS for name in VARIANTS]
    submit_batch(['dino_cache'] + runs, 'visual_trajectories', {n:'dino_cache' for n in runs}, max_parallel=2)
