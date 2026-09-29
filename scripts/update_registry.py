import argparse
import json
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.experiments.registry import update_registry

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--registry', default=os.environ.get('V2R_REGISTRY', 'results/experiment_registry.json'))
    p.add_argument('--experiment', required=True)
    p.add_argument('--status', required=True)
    p.add_argument('--job-id', default=os.environ.get('SLURM_JOB_ID'))
    p.add_argument('--error')
    args = p.parse_args()
    fields = {'status': args.status, 'job_id': args.job_id}
    if args.error:
        fields['error'] = args.error
    print(json.dumps(update_registry(args.registry, args.experiment, **fields)))
