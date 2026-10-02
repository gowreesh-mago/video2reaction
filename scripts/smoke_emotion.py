"""Three-video check of the exact production cache, selectors, and fusion training path."""
import argparse
import copy
import json
import os
from pathlib import Path
import socket
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import yaml

from scripts.check_gpu import gpu_info
from src.experiments.data import load_metadata
from src.experiments.emotion import load_emotion_logits, prepare_emotion_cache, probabilities
from src.experiments.evidence import EvidenceVideos
from src.experiments.features import CachedVideos, prepare_features
from src.experiments.metrics import evaluate
from src.experiments.models import ReactionPredictor
from src.experiments.registry import atomic_json, update_registry, utc_now
from src.experiments.runtime import load_config, seed_everything, verify_code
from src.experiments.taxonomy import REACTION_CLASSES
from src.experiments.training import fit, predict


VARIANTS = ['c_emotion_logits', 'c_emotion_vad', 'c_emotion_both',
    'd_arousal_k1', 'd_arousal_k2', 'd_arousal_k4', 'd_arousal_k8',
    'd_distance_k4', 'd_confidence_k4', 'd_uniform_k4', 'd_random_k4',
    'f_global_peak', 'f_global_control', 'f_peak_control',
    'b_vad_aux', 'b_vad_aux_permuted', 'b_vad_geometry', 'b_vad_geometry_permuted']


def main(args):
    if not torch.cuda.is_available() or not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Actual-data smoke requires a SLURM GPU allocation and cluster uv')
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / 'configs/experiments/smoke_emotion_3videos.yaml')
    code = verify_code(root)
    out = Path(os.environ['V2R_OUTPUT_DIR']) / 'smoke' / args.run_name
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Refusing to overwrite smoke output {out}')
    out.mkdir(parents=True, exist_ok=True)
    (out / 'config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
    atomic_json(out / 'code_version.json', code)
    atomic_json(out / 'environment.json', gpu_info())
    update_registry(args.registry, args.run_name, status='running', run_type='smoke', benchmark_result=False,
        start_time=utc_now(), job_id=os.environ['SLURM_JOB_ID'], hostname=socket.gethostname(),
        git_commit=code['git_commit'], source_sha256=code['source_sha256'], output_dir=str(out),
        hypothesis=cfg['experiment']['hypothesis'])
    torch.set_num_threads(min(4, int(os.environ.get('SLURM_CPUS_PER_TASK', '4'))))
    seed_everything(cfg['seed'])
    rows, sha = load_metadata(os.environ['V2R_METADATA_DIR'], 'train')
    if sha != cfg['data']['split_sha256']['train']:
        raise ValueError('Official training split changed')
    ids = cfg['data']['smoke_video_ids']
    if not 2 <= len(ids) <= 3 or len(set(ids)) != len(ids) or any(vid not in rows for vid in ids):
        raise ValueError('Smoke must use 2–3 distinct official training videos')
    splits = {'train': {vid: rows[vid] for vid in ids}}
    # A separate cache namespace prevents any mutation of the completed full cache.
    os.environ['V2R_FEATURE_DIR'] = str(out / 'visual_cache')
    os.environ['V2R_EMOTION_FEATURE_DIR'] = str(out / 'emotion_cache')
    visual_result = prepare_features(cfg, splits, out)
    torch.cuda.empty_cache()
    emotion_result = prepare_emotion_cache(cfg, splits, out)
    visual = CachedVideos(cfg, 'train', splits['train'])
    logits = load_emotion_logits(cfg, visual)
    q = probabilities(logits)
    atomic_json(out / 'sample_manifest.json', {'split': 'train', 'split_sha256': sha,
        'videos': visual.frame_records, 'benchmark_result': False})
    if visual.counts.tolist() != [8] * len(ids):
        raise ValueError('Expected exactly eight selected frames per smoke video')
    results = {}
    for name in VARIANTS:
        variant = load_config(root / f'configs/experiments/{name}.yaml')
        options = copy.deepcopy(cfg)
        for key in ('experiment', 'model', 'evidence', 'vad_objective'):
            if key in variant:
                options[key] = copy.deepcopy(variant[key])
        dataset = EvidenceVideos(options, visual)
        path = out / name
        dataset.save_evidence(path)
        seed_everything(options['seed'])
        initial = ReactionPredictor(**options['model']).to('cuda')
        initial_kl = evaluate(dataset.targets, predict(initial, dataset, options, 'cuda'))[0]['kl']
        del initial
        model, training = fit(options, dataset, dataset, path, 'cuda', code['source_sha256'])
        prediction = predict(model, dataset, options, 'cuda')
        shuffled = predict(model, dataset, options, 'cuda', shuffle_frames=True)
        metrics, diagnostics = evaluate(dataset.targets, prediction)
        if not metrics['kl'] < initial_kl:
            raise AssertionError(f'{name}: three-video fitting did not reduce KL')
        np.testing.assert_allclose(prediction.sum(1), 1, atol=1e-6)
        np.testing.assert_allclose(prediction, shuffled, atol=2e-6, rtol=2e-6)
        np.savez_compressed(path / 'predictions.npz', sample_id=np.asarray(dataset.ids),
            target_distribution=dataset.targets, predicted_distribution=prediction,
            target_topk=np.argsort(dataset.targets, axis=1)[:, -3:][:, ::-1],
            predicted_topk=np.argsort(prediction, axis=1)[:, -3:][:, ::-1], class_order=np.asarray(REACTION_CLASSES))
        record = dict(benchmark_result=False, evaluation_split='same three training videos',
            initial_kl=initial_kl, metrics=metrics, training=training, loss_decreased=True,
            max_shuffle_change=float(np.abs(prediction-shuffled).max()))
        atomic_json(path / 'metrics.json', record)
        atomic_json(path / 'diagnostics.json', diagnostics)
        results[name] = record
        print('SMOKE VARIANT', name, json.dumps(record), flush=True)
    summary = dict(status='completed', benchmark_result=False, video_count=len(ids), frame_count=len(logits),
        visual_cache=visual_result, emotion_cache=emotion_result, variants=results,
        emotion_probability_range=np.ptp(q, axis=0).tolist(), emotion_probability_mean=q.mean(0).tolist())
    atomic_json(out / 'metrics.json', summary)
    update_registry(args.registry, args.run_name, status='completed', end_time=utc_now(), metrics=summary)
    print(f'EMOTION SMOKE PASSED: {out}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-name', required=True)
    parser.add_argument('--registry', default=os.environ.get('V2R_REGISTRY'))
    args = parser.parse_args()
    try:
        main(args)
    except Exception:
        update_registry(args.registry, args.run_name, status='failed', end_time=utc_now(), error=traceback.format_exc())
        raise
