"""Three-video production integration check; these are not benchmark results."""
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
from scripts.submit_descriptions import PREDICTORS
from src.experiments.data import load_metadata
from src.experiments.descriptions import DescriptionVideos, prepare_description_cache
from src.experiments.diagnostics import boundaries, save_evaluation
from src.experiments.features import CachedVideos, prepare_features
from src.experiments.metrics import evaluate
from src.experiments.models import ReactionPredictor
from src.experiments.registry import atomic_json, update_registry, utc_now
from src.experiments.runtime import load_config, seed_everything, verify_code
from src.experiments.training import fit, predict


def main(args):
    if not torch.cuda.is_available() or not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Actual-data smoke requires a SLURM GPU allocation and cluster uv')
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / 'configs/experiments/smoke_descriptions_3videos.yaml')
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
    ids = cfg['data']['smoke_video_ids']
    if sha != cfg['data']['split_sha256']['train']:
        raise ValueError('Official training split changed')
    if not 2 <= len(ids) <= 3 or len(set(ids)) != len(ids) or any(vid not in rows for vid in ids):
        raise ValueError('Smoke requires 2–3 distinct official training videos')
    rows = {vid: rows[vid] for vid in ids}
    os.environ['V2R_FEATURE_DIR'] = str(out / 'visual_cache')
    os.environ['V2R_DESCRIPTION_FEATURE_DIR'] = str(out / 'description_cache')
    visual_result = prepare_features(cfg, {'train': rows}, out)
    torch.cuda.empty_cache()
    text_result = prepare_description_cache(cfg, {'train': rows}, out)
    visual = CachedVideos(cfg, 'train', rows)
    if visual.counts.tolist() != [8] * len(ids):
        raise ValueError('Expected eight frames per smoke video')
    atomic_json(out / 'sample_manifest.json', {'split': 'train', 'split_sha256': sha,
        'videos': visual.frame_records, 'benchmark_result': False})
    bins = boundaries(visual.targets, visual.counts)
    movies = {str(row['imdbid']) for row in rows.values()}
    results = {}
    for name in ['b1_meanpool'] + PREDICTORS:
        variant = load_config(root / f'configs/experiments/{name}.yaml')
        options = copy.deepcopy(cfg)
        for key in ('experiment', 'model', 'description'):
            if key in variant:
                options[key] = copy.deepcopy(variant[key])
        dataset = DescriptionVideos(options, visual, rows) if 'description' in options else visual
        path = out / name
        path.mkdir()
        (path / 'config.yaml').write_text(yaml.safe_dump(options, sort_keys=False))
        seed_everything(options['seed'])
        initial = ReactionPredictor(**options['model']).to('cuda')
        initial_kl = evaluate(dataset.targets, predict(initial, dataset, options, 'cuda'))[0]['kl']
        del initial
        model, training = fit(options, dataset, dataset, path, 'cuda', code['source_sha256'])
        prediction = predict(model, dataset, options, 'cuda')
        shuffled = predict(model, dataset, options, 'cuda', shuffle_frames=True)
        result = save_evaluation(path, 'train_smoke', dataset.ids, rows, dataset.targets, prediction,
                                 visual.counts, movies, bins, shuffled=shuffled)
        if not result['metrics']['kl'] < initial_kl:
            raise AssertionError(f'{name}: fitting did not reduce KL')
        np.testing.assert_allclose(prediction.sum(1), 1, atol=1e-6)
        np.testing.assert_allclose(prediction, shuffled, atol=2e-6, rtol=2e-6)
        record = {'benchmark_result': False, 'evaluation_split': 'same three training videos',
                  'initial_kl': initial_kl, 'metrics': result['metrics'], 'training': training,
                  'loss_decreased': True, 'max_shuffle_change': float(np.abs(prediction-shuffled).max())}
        atomic_json(path / 'metrics.json', record)
        results[name] = record
        print('DESCRIPTION SMOKE', name, json.dumps(record), flush=True)
    summary = {'status': 'completed', 'benchmark_result': False, 'video_count': len(ids),
               'visual_cache': visual_result, 'description_cache': text_result, 'variants': results}
    atomic_json(out / 'metrics.json', summary)
    update_registry(args.registry, args.run_name, status='completed', end_time=utc_now(), metrics=summary)
    print(f'DESCRIPTION SMOKE PASSED: {out}', flush=True)


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
