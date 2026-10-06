"""Exercise the production visual-only trajectory path on three training videos."""
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
from scripts.configure_trajectories import VARIANTS
from src.experiments.data import load_metadata
from src.experiments.features import CachedVideos, prepare_features
from src.experiments.losses import distribution_loss
from src.experiments.metrics import evaluate
from src.experiments.registry import atomic_json, update_registry, utc_now
from src.experiments.runtime import load_config, restore_rng, seed_everything, verify_code
from src.experiments.training import build_model, fit, loader, predict
from src.experiments.trajectory import TrajectoryVideos


def check_optimizer_resume(cfg, dataset, checkpoint):
    state = torch.load(checkpoint, map_location='cuda', weights_only=False)
    actual = []
    for _ in range(2):
        model = build_model(cfg, 'cuda')
        model.load_state_dict(state['model'])
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['training']['learning_rate'],
                                      weight_decay=cfg['training']['weight_decay'])
        optimizer.load_state_dict(copy.deepcopy(state['optimizer']))
        restore_rng(state['rng'])
        x, mask, target, _ = next(iter(loader(dataset, cfg)))
        loss = distribution_loss(model(x.cuda(), mask.cuda())[0], target.cuda(), **cfg['loss'])
        if not torch.isfinite(loss):
            raise AssertionError('Nonfinite resumed loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['training']['clip_grad_norm'], error_if_nonfinite=True)
        optimizer.step()
        if not any(not torch.equal(value, state['model'][key]) for key,value in model.state_dict().items()):
            raise AssertionError('Resumed optimizer did not update any parameters')
        actual.append({k:v.detach().cpu().clone() for k,v in model.state_dict().items()})
    for key in actual[0]:
        torch.testing.assert_close(actual[0][key], actual[1][key], rtol=0, atol=0)
    return True


def main(args):
    if not torch.cuda.is_available() or not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Actual-data smoke requires an allocated GPU and cluster uv')
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / 'configs/experiments/smoke_trajectory_3videos.yaml')
    code = verify_code(root)
    out = Path(os.environ['V2R_OUTPUT_DIR']) / 'smoke' / args.run_name
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Refusing to overwrite {out}')
    out.mkdir(parents=True, exist_ok=True)
    atomic_json(out/'code_version.json', code)
    atomic_json(out/'environment.json', gpu_info())
    (out/'config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
    update_registry(args.registry, args.run_name, status='running', run_type='smoke', benchmark_result=False,
        start_time=utc_now(), job_id=os.environ['SLURM_JOB_ID'], hostname=socket.gethostname(),
        git_commit=code['git_commit'], source_sha256=code['source_sha256'], output_dir=str(out))
    torch.set_num_threads(4)
    seed_everything(cfg['seed'])
    rows, sha = load_metadata(os.environ['V2R_METADATA_DIR'], 'train')
    if sha != cfg['data']['split_sha256']['train']:
        raise ValueError('Official training metadata changed')
    ids = cfg['data']['smoke_video_ids']
    if len(set(ids)) != 3 or any(vid not in rows for vid in ids):
        raise ValueError('Smoke requires exactly three distinct official training videos')
    rows = {vid: rows[vid] for vid in ids}
    os.environ['V2R_DINO_FEATURE_DIR'] = str(out/'visual_cache')
    cache = prepare_features(cfg, {'train':rows}, out)
    torch.cuda.empty_cache()
    visual = CachedVideos(cfg, 'train', rows)
    if visual.counts.tolist() != [8,8,8]:
        raise ValueError('Expected eight observed scenes per smoke video')
    atomic_json(out/'sample_manifest.json', {'split':'train','split_sha256':sha,
        'videos':visual.frame_records, 'benchmark_result':False})
    trajectory = TrajectoryVideos(visual)
    results = {}
    for name in VARIANTS:
        options = load_config(root / f'configs/experiments/{name}.yaml')
        options['data'], options['encoder'] = copy.deepcopy(cfg['data']), copy.deepcopy(cfg['encoder'])
        options['training'].update(cfg['training'])
        dataset = trajectory if 'trajectory' in options else visual
        path = out/name
        path.mkdir()
        (path/'config.yaml').write_text(yaml.safe_dump(options, sort_keys=False))
        seed_everything(options['seed'])
        initial = build_model(options, 'cuda')
        initial_kl = evaluate(dataset.targets, predict(initial, dataset, options, 'cuda'))[0]['kl']
        del initial
        model, info = fit(options, dataset, dataset, path, 'cuda', code['source_sha256'])
        prediction = predict(model, dataset, options, 'cuda', attention_output=path/'evaluation' if 'trajectory' in options else None)
        metrics, _ = evaluate(dataset.targets, prediction)
        if not metrics['kl'] < initial_kl:
            raise AssertionError(f'{name}: smoke fitting did not lower KL')
        np.testing.assert_allclose(prediction.sum(1), 1, atol=1e-6)
        resumed = check_optimizer_resume(options, dataset, path/'last.pt')
        results[name] = {'initial_kl':initial_kl, 'fitted_kl':metrics['kl'],
            'loss_decreased':True, 'checkpoint_reload_exact':info['checkpoint_reload_exact'],
            'optimizer_resume_passed':resumed, 'contribution_export_verified':'trajectory' in options,
            'trainable_parameters':info['trainable_parameters']}
        atomic_json(path/'metrics.json', dict(results[name], benchmark_result=False))
        print('TRAJECTORY SMOKE', name, json.dumps(results[name]), flush=True)
        del model
    summary = {'status':'completed', 'benchmark_result':False, 'video_count':3, 'frame_count':24,
        'visual_only':True, 'encoder':cfg['encoder'], 'visual_cache':cache, 'variants':results}
    atomic_json(out/'metrics.json', summary)
    update_registry(args.registry, args.run_name, status='completed', end_time=utc_now(), metrics=summary)
    print(f'TRAJECTORY SMOKE PASSED: {out}', flush=True)


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
