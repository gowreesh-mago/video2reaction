"""Cluster-only, held-out VAD -> DINO reconstruction and frozen-head evaluation."""
import argparse
import copy
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.analyze_previous_runs import sha, dump, write_csv, diagnostic, sample_metrics, clustered_interval
from src.experiments.data import target_distribution
from src.experiments.taxonomy import REACTION_CLASSES
from src.experiments.vad_dino import Basis, RidgeMap, scene_sample, reconstruction_metrics, clip_mean, retrieval

RUNS = {42: ('27685259', '27682150'), 43: ('27685364', '27682169'), 44: ('27685447', '27682189')}


def log(event, **values):
    print(json.dumps({'event': event, **values}), flush=True)


def load_split(root, cache, split, cfg, smoke):
    metadata = root / 'data/metadata' / (split + '.json')
    assert sha(metadata) == cfg['data']['split_sha256'][split]
    rows = json.loads(metadata.read_text())
    folder = cache / split
    manifest = json.loads((folder / 'complete.json').read_text())
    for file, field in [('features.npy', 'features_sha256'), ('index.json', 'index_sha256')]:
        if sha(folder / file) != manifest[field]:
            raise ValueError(f'Corrupt {split} feature cache: {file}')
    index = json.loads((folder / 'index.json').read_text())
    ids = [v['sample_id'] for v in index['videos']]
    assert ids == sorted(rows)
    assert index['split_sha256'] == cfg['data']['split_sha256'][split]
    count = 1 if smoke else len(ids)
    ids = ids[:count]
    offsets = np.asarray(index['offsets'][:count + 1])
    return dict(ids=ids, offsets=offsets, frames=index['videos'][:count],
                features=np.load(folder / 'features.npy', mmap_mode='r')[:offsets[-1]],
                target=np.stack([target_distribution(rows[i]) for i in ids]),
                movies=np.asarray([str(rows[i]['imdbid']) for i in ids]))


def predicted_vad(data, model, torch, out):
    from src.experiments.trajectory import interval_packet
    values = np.empty((data['offsets'][-1], 3), np.float32)
    outputs = np.empty((len(data['ids']), 21), np.float32)
    started = time.monotonic()
    with torch.inference_mode():
        for start in range(0, len(data['ids']), 16):
            end = min(start + 16, len(data['ids']))
            frames = []
            for i in range(start, end):
                a, b = data['offsets'][i:i+2]
                x = torch.from_numpy(np.array(data['features'][a:b]))
                frames.append(torch.cat([x, interval_packet(data['frames'][i]['frames'])], dim=-1))
            lengths = torch.tensor([len(x) for x in frames])
            x = torch.nn.utils.rnn.pad_sequence(frames, batch_first=True)
            mask = torch.arange(x.shape[1])[None, :] < lengths[:, None]
            logits, _, details = model(x, mask, return_details=True)
            for j, i in enumerate(range(start, end)):
                a, b = data['offsets'][i:i+2]
                values[a:b] = details['vad'][j, :b-a].numpy()
            outputs[start:end] = logits.softmax(-1).numpy()
            if start % 512 == 0:
                log('infer_vad', completed=end, total=len(data['ids']), seconds=time.monotonic()-started)
    if not np.isfinite(values).all() or (np.abs(values) > 1).any():
        raise ValueError('Invalid predicted VAD')
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / 'vad.npy', values)
    np.savez_compressed(out / 'predictions.npz', sample_id=data['ids'], prediction=outputs, offsets=data['offsets'])
    return values, outputs


def saved_vad(data, source, split):
    path = source / split / 'trajectory.npz'
    summary = json.loads((source / split / 'trajectory_summary.json').read_text())
    assert sha(path) == summary['trajectory_sha256']
    with np.load(path, allow_pickle=False) as f:
        n = len(data['ids'])
        np.testing.assert_array_equal(f['sample_id'][:n], data['ids'])
        np.testing.assert_array_equal(f['offsets'][:n+1], data['offsets'])
        np.testing.assert_array_equal(f['class_order'], REACTION_CLASSES)
        return f['vad'][:data['offsets'][-1]], f['prediction'][:n]


def head_prediction(model, values, offsets, torch):
    """Preserve the baseline's per-scene LayerNorm -> projection -> mean -> head."""
    hidden = np.empty((len(values), model.hidden_dim), np.float32)
    with torch.inference_mode():
        for start in range(0, len(values), 2048):
            hidden[start:start+2048] = model.projection(torch.from_numpy(np.array(values[start:start+2048], dtype=np.float32))).numpy()
        pooled = clip_mean(hidden, offsets).astype(np.float32)
        q = model.head(torch.from_numpy(pooled)).softmax(-1).numpy()
    return q


def choose_mapping(kind, train_z, train_y, train_w, val_z, val_y, val_w, seed, folder):
    candidates, best = [], None
    for width in ([.5, 1., 2.] if kind == 'rbf' else [1.]):
        basis = Basis(kind, seed=seed, bandwidth=width).fit(train_z, train_w)
        phi, vphi = basis.transform(train_z), basis.transform(val_z)
        for alpha in (.0001, .01, 1.):
            model = RidgeMap().fit(phi, train_y, train_w, alpha)
            q = model.predict(vphi)
            error = float(np.average(np.square(q - val_y).mean(1), weights=val_w))
            candidate = dict(kind=kind, bandwidth=width, alpha=alpha, val_mse=error)
            candidates.append(candidate)
            if best is None or error < best[0]['val_mse']:
                best = candidate, basis, model
    candidate, basis, model = best
    dump(folder / 'selection.json', {'selection_split': 'val', 'criterion': 'equal-clip sampled-scene reconstruction MSE',
                                    'candidates': candidates, 'selected': candidate})
    np.savez_compressed(folder / 'parameters.npz', basis_mean=basis.mean, basis_scale=basis.scale,
                        basis_centers=basis.centers, ridge_xmean=model.xmean, ridge_ymean=model.ymean,
                        ridge_coef=model.coef)
    return basis, model, candidate


def run(args):
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Real data execution requires a Snellius Slurm allocation')
    import torch
    import yaml
    from src.experiments.models import ReactionPredictor
    from src.experiments.trajectory import TrajectoryPredictor, load_prototypes
    from src.experiments.runtime import verify_code
    code = verify_code(Path(__file__).resolve().parents[1])
    torch.set_num_threads(min(8, int(os.environ.get('SLURM_CPUS_PER_TASK', 4))))
    torch.manual_seed(args.seed)
    out = args.out
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Refusing to overwrite {out}')
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    root = args.root
    source = root / 'outputs/experiments' / f'on_vad_soft_tau010_s{args.seed}_{RUNS[args.seed][0]}'
    baseline = root / 'outputs/experiments' / f'dino_meanpool_s{args.seed}_{RUNS[args.seed][1]}'
    cfg = yaml.safe_load((source / 'config.yaml').read_text())
    cache = Path(json.loads((source / 'feature_cache.json').read_text())['path'])
    assert cache == Path(json.loads((baseline / 'feature_cache.json').read_text())['path'])
    environment = {p: importlib.metadata.version(p) for p in ['torch', 'numpy', 'scikit-learn', 'scipy']}
    provenance = dict(seed=args.seed, smoke=args.smoke, slurm_job_id=os.environ['SLURM_JOB_ID'], code=code,
                      packages=environment, source_run=str(source), baseline_run=str(baseline),
                      source_checkpoint_sha256=sha(source / 'best.pt'), baseline_checkpoint_sha256=sha(baseline / 'best.pt'),
                      cache_manifest_sha256=sha(cache / 'manifest.json'), metadata_sha256=cfg['data']['split_sha256'])
    dump(out / 'provenance.json', provenance)
    options = copy.deepcopy(cfg['trajectory'])
    prototypes, _ = load_prototypes(options.pop('prototypes'))
    trajectory = TrajectoryPredictor(**cfg['model'], prototypes=prototypes, **options).eval()
    state = torch.load(source / 'best.pt', map_location='cpu', weights_only=True)
    assert state['config'] == cfg
    trajectory.load_state_dict(state['model'])
    base_cfg = yaml.safe_load((baseline / 'config.yaml').read_text())
    dino = ReactionPredictor(**base_cfg['model']).eval()
    dino.load_state_dict(torch.load(baseline / 'best.pt', map_location='cpu', weights_only=True)['model'])
    train = load_split(root, cache, 'train', cfg, args.smoke)
    val = load_split(root, cache, 'val', cfg, args.smoke)
    assert not set(train['ids']) & set(val['ids'])
    train_z, _ = predicted_vad(train, trajectory, torch, out / 'cluster_only/train')
    val_z, val_expected = saved_vad(val, source, 'val')
    # Re-inference on validation proves the loaded model/runtime reproduces saved outputs.
    val_new_z, val_new_q = predicted_vad(val, trajectory, torch, out / 'cluster_only/val')
    np.testing.assert_allclose(val_new_z, val_z, atol=2e-5, rtol=2e-5)
    np.testing.assert_allclose(val_new_q, val_expected, atol=2e-5, rtol=2e-5)
    ti, tw = scene_sample(train['offsets']); vi, vw = scene_sample(val['offsets'])
    ty = np.array(train['features'][ti], dtype=float)
    vy = np.array(val['features'][vi], dtype=float)
    z, vz = train_z[ti].astype(float), val_z[vi].astype(float)
    mean = np.average(ty, axis=0, weights=tw)
    pca = PCA(n_components=min(32, len(ty), ty.shape[1]), svd_solver='randomized', random_state=args.seed).fit(ty)
    selected, models = {}, {}
    for name, kind, shuffle in [('vad_linear', 'linear', False), ('vad_quadratic', 'quadratic', False),
                                ('vad_rbf', 'rbf', False), ('shuffled_vad_rbf', 'rbf', True)]:
        # All clips have >= 8 scenes in the real dataset: shuffle whole clip blocks,
        # preserving within-video trajectories while destroying visual pairing.
        fit_z = z.copy()
        if shuffle:
            sizes = [len(np.flatnonzero((ti >= a) & (ti < b))) for a, b in zip(train['offsets'][:-1], train['offsets'][1:])]
            if len(set(sizes)) != 1:
                raise ValueError('Shuffled control requires equal sampled scenes per training clip')
            blocks = fit_z.reshape(len(train['ids']), sizes[0], 3)
            perm = np.random.default_rng(args.seed + 1000).permutation(len(blocks))
            fit_z = blocks[perm].reshape(-1, 3)
        basis, model, choice = choose_mapping(kind, fit_z, ty, tw, vz, vy, vw, args.seed, out / 'models' / name)
        models[name] = basis, model
        selected[name] = choice
        log('mapping_selected', name=name, **choice)
    # Freeze every choice before opening held-out test inputs/labels.
    dump(out / 'selection_frozen.json', {'selected': selected, 'test_used_in_selection': False,
        'train_clips': len(train['ids']), 'val_clips': len(val['ids']), 'train_sampled_scenes': len(ti),
        'sampled_scenes_per_clip_max': 8, 'fit_weights': 'equal total weight per clip'})
    np.savez_compressed(out / 'models/pca.npz', components=pca.components_, mean=pca.mean_, train_mean=mean)
    del ty, vy
    test = load_split(root, cache, 'test', cfg, args.smoke)
    assert not set(test['ids']) & (set(train['ids']) | set(val['ids']))
    test_z, vad_q = saved_vad(test, source, 'test')
    target = test['target']; original = test['features']; offsets = test['offsets']
    original_means = clip_mean(original, offsets)
    base_q = head_prediction(dino, original, offsets, torch)
    with np.load(baseline / 'test/predictions.npz', allow_pickle=False) as f:
        np.testing.assert_array_equal(f['sample_id'][:len(target)], test['ids'])
        np.testing.assert_array_equal(f['target_distribution'][:len(target)], target)
        np.testing.assert_allclose(f['predicted_distribution'][:len(target)], base_q, atol=2e-5, rtol=2e-5)
    base_errors = sample_metrics(target, base_q)
    results, plot = {}, {'sample_id': np.asarray(test['ids']), 'class_order': np.asarray(REACTION_CLASSES),
        'target': target, 'train_class_mass': train['target'].mean(0), 'prediction_original_dino': base_q,
        'prediction_original_vad': vad_q, 'original_dino_pca3': pca.transform(original_means)[:, :3]}
    for name in ['original_dino', 'train_mean', 'dino_pca3', 'dino_pca32', *models]:
        if name == 'original_dino':
            reconstructed = original
        elif name == 'train_mean':
            reconstructed = np.broadcast_to(mean, original.shape)
        elif name.startswith('dino_pca'):
            k = min(int(name.replace('dino_pca', '')), len(pca.components_))
            c = pca.components_[:k]
            reconstructed = (np.asarray(original) - pca.mean_) @ c.T @ c + pca.mean_
        else:
            basis, decoder = models[name]
            reconstructed = np.empty(original.shape, np.float32)
            for start in range(0, len(original), 2048):
                reconstructed[start:start+2048] = decoder.predict(basis.transform(test_z[start:start+2048]))
        q = base_q if name == 'original_dino' else head_prediction(dino, reconstructed, offsets, torch)
        recon, mse = reconstruction_metrics(original, reconstructed, mean, offsets)
        means = clip_mean(reconstructed, offsets)
        retrieval_metrics = retrieval(original_means, means, mean)
        details = diagnostic(target, q, train['target'])
        per_video = sample_metrics(target, q)
        interval = clustered_interval(per_video['kl']-base_errors['kl'], test['movies'], np.random.default_rng(args.seed))
        details.update(reconstruction=recon, retrieval=retrieval_metrics, kl_difference_vs_dino=interval)
        results[name] = details
        dump(out / 'diagnostics' / (name + '.json'), details)
        write_csv(out / 'diagnostics' / (name + '_per_class.csv'), details['class_rows'])
        rows = [dict(sample_id=str(test['ids'][i]), movie_id=str(test['movies'][i]),
                     reconstruction_mse=float(mse[i]), **{k: float(v[i]) for k, v in per_video.items()}) for i in range(len(q))]
        write_csv(out / 'diagnostics' / (name + '_per_video.csv'), sorted(rows, key=lambda r: -r['kl']))
        plot['prediction_' + name] = q
        plot[name + '_pca3'] = pca.transform(means)[:, :3]
        np.savez_compressed(out / 'diagnostics' / (name + '_predictions.npz'), sample_id=test['ids'], movie_id=test['movies'],
                            class_order=REACTION_CLASSES, target_distribution=target, predicted_distribution=q)
        log('test_result', name=name, reconstruction=recon, reaction=details['metrics'])
        del reconstructed
    results['original_vad'] = diagnostic(target, vad_q, train['target'])
    np.savez_compressed(out / 'plot_arrays.npz', **plot)
    summary = dict(schema=1, status='completed', seed=args.seed, smoke=args.smoke, test_clips=len(target),
        test_scenes=len(test_z), elapsed_seconds=time.monotonic()-started, selected=selected, results=results,
        validation_reproduction_max_abs=float(np.max(np.abs(val_new_q-val_expected))),
        interpretation=['Predicted VAD comes from the frozen pure-vision trajectory model.',
            'Reconstruction tests scene information retained in VAD, not whether extra information is created.',
            'Reaction distributions use the original frozen DINO head; no head is refitted on reconstructed features.',
            'PCA controls use DINO features directly; they are not VAD methods.',
            'Feature fitting uses at most eight evenly sampled scenes per training clip; test uses every scene.',
            'Training coordinates are in-sample predictions of the original model; validation/test are held out.',
            'The original benchmark split may share movies; intervals resample movies to respect clustering.'])
    dump(out / 'results.json', summary)
    log('completed', seconds=summary['elapsed_seconds'], output=str(out))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--seed', type=int, choices=RUNS, required=True)
    p.add_argument('--smoke', action='store_true', help='Exactly three clips: one from each official split. Not benchmark results.')
    run(p.parse_args())
