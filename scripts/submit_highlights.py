"""Submit the reviewed joint and pretrained highlight baselines, bounded to two jobs."""
import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.submit_emotion import require_completed_cache
from scripts.submit_tier0 import submit_batch


def require_highlight_smoke(job_id):
    require_completed_cache(job_id)
    root = Path(os.environ['V2R_ROOT'])
    out = root / 'outputs/smoke' / f'smoke_highlights_3videos_{job_id}'
    smoke = json.loads((out/'metrics.json').read_text())
    source = json.loads((out/'code_version.json').read_text())
    current = json.loads(Path('code_version.json').read_text())
    expected = {'meanpool', 'joint_highlight_sparse', 'joint_highlight_soft_control', 'pretrained_dsnet_k4'}
    if (source['source_sha256'] != current['source_sha256'] or smoke['status'] != 'completed'
            or smoke['benchmark_result'] or smoke['video_count'] != 3 or smoke['frame_count'] != 24
            or set(smoke['variants']) != expected):
        raise RuntimeError('Highlight smoke does not validate this source and experiment set')
    for variant in smoke['variants'].values():
        if not all(variant[key] for key in ('loss_decreased', 'checkpoint_reload_exact', 'optimizer_resume_passed')):
            raise RuntimeError('Highlight smoke model checks did not all pass')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--completed-smoke-job', required=True)
    args = parser.parse_args()
    if not args.completed_smoke_job.isdigit():
        parser.error('A completed highlight smoke job ID is required')
    require_highlight_smoke(args.completed_smoke_job)
    submit_batch(['joint_highlight_sparse', 'joint_highlight_soft_control', 'highlight_cache', 'pretrained_dsnet_k4'],
        'highlights', {'pretrained_dsnet_k4': 'highlight_cache'}, max_parallel=2)
