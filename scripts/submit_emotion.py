"""Run the reviewed C/D/F batch in at most four concurrent lanes after its emotion cache."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.submit_tier0 import command, submit_batch


PRIMARY = ['c_emotion_logits', 'c_emotion_vad', 'c_emotion_both',
    'f_global_peak', 'f_global_control', 'f_peak_control',
    'd_arousal_k4', 'd_distance_k4', 'd_confidence_k4', 'd_uniform_k4', 'd_random_k4']
CURVE = [f'd_{method}_k{k}' for k in [1, 2, 8]
         for method in ['arousal', 'distance', 'confidence', 'uniform', 'random']]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['primary', 'curve'], required=True)
    parser.add_argument('--completed-cache-job', help='Required for the K-curve phase; must have completed successfully')
    args = parser.parse_args()
    if args.phase == 'primary':
        submit_batch(['emotion_cache'] + PRIMARY, 'emotion_primary',
                     {name: 'emotion_cache' for name in PRIMARY}, max_parallel=4)
    else:
        if not args.completed_cache_job or not args.completed_cache_job.isdigit():
            parser.error('Curve phase requires --completed-cache-job ID')
        state = command('sacct', '-j', args.completed_cache_job, '-X', '-n', '-P', '--format=State,ExitCode')
        if state.strip() != 'COMPLETED|0:0|':
            raise RuntimeError(f'Emotion cache is not verified completed: {state}')
        submit_batch(CURVE, 'emotion_curve', max_parallel=4)
