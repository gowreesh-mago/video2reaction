"""Run one reproducible full experiment; actual data execution is cluster-only."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import socket
import traceback

import numpy as np
import torch
import yaml

from src.experiments.data import frame_index, target_distribution
from src.experiments.diagnostics import boundaries, save_evaluation
from src.experiments.descriptions import DescriptionVideos, description_directory, prepare_description_cache
from src.experiments.features import CachedVideos, prepare_features
from src.experiments.emotion import emotion_directory, prepare_emotion_cache
from src.experiments.evidence import EvidenceVideos
from src.experiments.registry import atomic_json, update_registry, utc_now
from src.experiments.runtime import cache_directory, digest_file, load_config, official_splits, seed_everything, verify_code
from src.experiments.training import fit, predict
from scripts.check_gpu import gpu_info


def run(args):
    if not torch.cuda.is_available() or not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Real experiments require a SLURM GPU allocation and the cluster uv environment')
    cfg = load_config(args.config)
    code = verify_code(Path(__file__).resolve().parent)
    out = Path(os.environ['V2R_OUTPUT_DIR']) / 'experiments' / args.run_name
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Use a new attempt ID; refusing to overwrite {out}')
    out.mkdir(parents=True, exist_ok=True)
    log_path = os.environ.get('V2R_RUN_LOG')
    if log_path:
        (out / 'stdout.log').symlink_to(log_path)
    (out / 'config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
    atomic_json(out / 'code_version.json', code)
    environment = gpu_info()
    environment['packages'] = {name: importlib.metadata.version(name) for name in
                               ('torch', 'transformers', 'numpy', 'scikit-learn', 'pillow', 'pyyaml')}
    atomic_json(out / 'environment.json', environment)
    update_registry(args.registry, args.run_name, status='running', job_id=os.environ['SLURM_JOB_ID'],
                    start_time=utc_now(), hostname=socket.gethostname(), git_commit=code['git_commit'],
                    source_sha256=code['source_sha256'], config=str(out / 'config.yaml'),
                    experiment_name=cfg['experiment']['name'], hypothesis=cfg['experiment']['hypothesis'],
                    output_dir=str(out), run_type='feature_cache' if cfg['experiment']['name'] in {'feature_cache', 'emotion_cache', 'description_cache'} else 'benchmark')
    torch.set_num_threads(min(4, int(os.environ.get('SLURM_CPUS_PER_TASK', '4'))))
    seed_everything(cfg['seed'])
    splits, split_hashes = official_splits(cfg)
    atomic_json(out / 'split_hashes.json', split_hashes)
    name = cfg['experiment']['name']
    if name == 'feature_cache':
        summary = prepare_features(cfg, splits, out)
    elif name == 'emotion_cache':
        summary = prepare_emotion_cache(cfg, splits, out)
    elif name == 'description_cache':
        summary = prepare_description_cache(cfg, splits, out)
    else:
        train_movies = {str(row['imdbid']) for row in splits['train'].values()}
        if name == 'b0_prior':
            ids = {s: sorted(rows) for s, rows in splits.items()}
            targets = {s: np.stack([target_distribution(rows[vid]) for vid in ids[s]]) for s, rows in splits.items()}
            counts = {s: np.array([frame_index(os.environ['V2R_FRAME_DIR'], vid, cfg['data']['max_frames'])[1] for vid in ids[s]]) for s in splits}
            # The only fitted quantity uses training labels exclusively.
            prior = targets['train'].astype(np.float64).mean(0)
            prior /= prior.sum()
            atomic_json(out / 'prior.json', {'fit_split': 'train', 'distribution': prior.tolist()})
            info = {'trainable_parameters': 0, 'fit_split': 'train'}
            bins = boundaries(targets['train'], counts['train'])
            result = {s: save_evaluation(out, s, ids[s], splits[s], targets[s],
                      np.broadcast_to(prior, targets[s].shape).copy(), counts[s], train_movies, bins) for s in ('val', 'test')}
        else:
            # Load train/validation first. Test predictions are made only after checkpoint selection.
            train = CachedVideos(cfg, 'train', splits['train'])
            val = CachedVideos(cfg, 'val', splits['val'])
            if 'description' in cfg:
                train = DescriptionVideos(cfg, train, splits['train'])
                val = DescriptionVideos(cfg, val, splits['val'])
                text_manifest = description_directory(cfg) / 'manifest.json'
                atomic_json(out / 'description_cache.json', {'path': str(text_manifest.parent),
                    'manifest_sha256': digest_file(text_manifest), 'manifest': json.loads(text_manifest.read_text()),
                    'input_mode': cfg['description']['mode']})
            if 'evidence' in cfg:
                train, val = EvidenceVideos(cfg, train), EvidenceVideos(cfg, val)
                emotion_manifest = emotion_directory(cfg) / 'manifest.json'
                atomic_json(out / 'emotion_cache.json', {'path': str(emotion_manifest.parent),
                    'manifest_sha256': digest_file(emotion_manifest),
                    'vad_sha256': cfg['emotion_vad_sha256'], 'manifest': json.loads(emotion_manifest.read_text())})
            manifest_path = cache_directory(cfg) / 'manifest.json'
            atomic_json(out / 'feature_cache.json', {'path': str(manifest_path.parent),
                        'manifest_sha256': digest_file(manifest_path), 'manifest': json.loads(manifest_path.read_text())})
            model, info = fit(cfg, train, val, out, 'cuda', code['source_sha256'], args.resume_from)
            bins = boundaries(train.targets, getattr(train, 'stratum_counts', train.counts))
            result = {}
            for s in ('val', 'test'):
                dataset = val if s == 'val' else CachedVideos(cfg, s, splits[s])
                if 'description' in cfg and s == 'test':
                    dataset = DescriptionVideos(cfg, dataset, splits[s])
                if 'evidence' in cfg:
                    if s == 'test':
                        dataset = EvidenceVideos(cfg, dataset)
                    dataset.save_evidence(out / s)
                result[s] = save_evaluation(out, s, dataset.ids, splits[s], dataset.targets,
                            predict(model, dataset, cfg, 'cuda', attention_output=out / s
                                    if cfg['evaluation'].get('save_attention', False) else None),
                            getattr(dataset, 'stratum_counts', dataset.counts), train_movies, bins,
                            shuffled=predict(model, dataset, cfg, 'cuda', shuffle_frames=True))
        summary = {'benchmark_result': True, 'experiment': cfg['experiment'], 'training': info,
                   'val': result['val']['metrics'], 'test': result['test']['metrics'],
                   'seed': cfg['seed'], 'selection': cfg['training']['selection_metric'] if name != 'b0_prior' else 'none'}
        if 'description' in cfg:
            summary['input_mode'] = cfg['description']['mode']
            summary['uses_description'] = cfg['description']['mode'] != 'visual_visual'
    atomic_json(out / 'metrics.json', summary)
    update_registry(args.registry, args.run_name, status='completed', end_time=utc_now(),
                    benchmark_result=summary['benchmark_result'], metrics=summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--run-name', required=True)
    parser.add_argument('--registry', default=os.environ.get('V2R_REGISTRY'))
    parser.add_argument('--resume-from', type=Path)
    args = parser.parse_args()
    try:
        run(args)
    except Exception:
        try:
            update_registry(args.registry, args.run_name, status='failed', end_time=utc_now(), error=traceback.format_exc())
        except Exception:
            traceback.print_exc()
        raise
