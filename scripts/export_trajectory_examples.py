"""Export selected existing test trajectories, without loading a model or video data."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


RUN = 'on_vad_soft_tau010_s42_27685259'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def seconds(value):
    parts = str(value).split(':')
    if len(parts) == 3:
        return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
    if len(parts) == 1:
        return float(parts[0])
    raise ValueError('Invalid timestamp')


def kl(p, q):
    return (p * np.log(p / (q + 1e-10) + 1e-10)).sum(-1)


def select_examples(ids, errors, rare_mass, counts, mode='diagnostic'):
    """Stable strata for visual inspection, deliberately not a random test sample."""
    if mode == 'kl_extremes':
        if len(ids) < 20 or len(set(ids)) != len(ids) or not np.isfinite(errors).all():
            raise ValueError('Need at least twenty unique test videos with finite errors')
        if (np.asarray(counts) < 1).any():
            raise ValueError('Every ranked video must have a saved scene')
        ascending = sorted(range(len(ids)), key=lambda i: (errors[i], str(ids[i])))
        best = ascending[:10]
        worst = sorted([i for i in ascending if i not in set(best)], key=lambda i: (-errors[i], str(ids[i])))[:10]
        return [(i, 'Best 10 · lowest KL') for i in best] + [(i, 'Worst 10 · highest KL') for i in worst]
    if mode != 'diagnostic':
        raise ValueError('Unknown example selection mode')
    valid = [i for i in range(len(ids)) if counts[i] >= 2]
    if len(valid) < 10:
        raise ValueError('Need at least ten videos with two or more scenes')
    ascending = sorted(valid, key=lambda i: (errors[i], str(ids[i])))
    median = np.median(errors[valid])
    selected = []
    for label, order, n in [
        ('Low VAD error', ascending, 3),
        ('Near median VAD error', sorted(valid, key=lambda i: (abs(errors[i]-median), str(ids[i]))), 2),
        ('High VAD error', sorted(valid, key=lambda i: (-errors[i], str(ids[i]))), 3),
        ('High rare-reaction mass', sorted(valid, key=lambda i: (-rare_mass[i], str(ids[i]))), 2),
    ]:
        used = {i for i, _ in selected}
        selected.extend((i, label) for i in [i for i in order if i not in used][:n])
    assert len(selected) == len({i for i, _ in selected}) == 10
    return selected


def validate_example(example, prototypes, temperature):
    """Verify the actual decoder, mixture, and chronology, not only array shapes."""
    z = np.asarray(example['vad'], dtype=float)
    q = np.asarray(example['moment_probabilities'], dtype=float)
    w = np.asarray(example['weights'], dtype=float)
    contributions = np.asarray(example['contributions'], dtype=float)
    intervals = np.asarray(example['intervals_seconds'], dtype=float)
    pred = np.asarray(example['prediction'], dtype=float)
    target = np.asarray(example['target'], dtype=float)
    dino = np.asarray(example['dino_prediction'], dtype=float)
    n = len(z)
    if z.shape != (n, 3) or q.shape != (n, 21) or w.shape != (n,) or contributions.shape != q.shape:
        raise ValueError('Invalid trajectory dimensions')
    if intervals.shape != (n, 2) or n < 1 or temperature <= 0:
        raise ValueError('Invalid intervals or temperature')
    for a in [z, q, w, contributions, intervals, pred, target, dino]:
        if not np.isfinite(a).all():
            raise ValueError('Nonfinite data')
    if (np.abs(z) > 1.00001).any() or (intervals[:, 1] <= intervals[:, 0]).any():
        raise ValueError('Invalid VAD range or duration')
    if (intervals[:, 0] < 0).any() or (intervals[1:, 0] < intervals[:-1, 1]-1e-5).any():
        raise ValueError('Negative or overlapping scene times')
    for a in [q, w, contributions, pred, target, dino]:
        if (a < 0).any():
            raise ValueError('Negative probability')
    distance_squared = ((z[:, None, :] - prototypes[None, :, :])**2).sum(-1)
    logit = -distance_squared / temperature
    decoded = np.exp(logit - logit.max(1, keepdims=True))
    decoded /= decoded.sum(1, keepdims=True)
    np.testing.assert_allclose(q, decoded, atol=3e-6, rtol=3e-5)
    np.testing.assert_allclose(q.sum(1), 1, atol=2e-6)
    np.testing.assert_allclose(w.sum(), 1, atol=2e-6)
    for a in [pred, target, dino]:
        if a.shape != (21,):
            raise ValueError('Expected 21 reactions')
        np.testing.assert_allclose(a.sum(), 1, atol=2e-6)
    np.testing.assert_allclose(contributions, w[:, None]*q, atol=2e-6, rtol=3e-5)
    np.testing.assert_allclose(contributions.sum(0), pred, atol=2e-6, rtol=3e-5)
    return distance_squared


def export(root, analysis, out, selection='diagnostic'):
    root, analysis, out = map(Path, (root, analysis, out))
    run = root/'outputs/experiments'/RUN
    paths = {
        'trajectory': run/'test/trajectory.npz',
        'trajectory_summary': run/'test/trajectory_summary.json',
        'frames': run/'test/attention_frames.json',
        'provenance': run/'trajectory_provenance.json',
        'predictions': run/'test/predictions.npz',
        'analysis_arrays': analysis/'plot_arrays.npz',
        'metadata': analysis/'sample_metadata.json',
        'prototype_asset': root/'data/lexicons/nrc-vad-v2.1/reaction_vad.json',
    }
    hashes = {k: sha(p) for k, p in paths.items()}
    summary = json.loads(paths['trajectory_summary'].read_text())
    provenance = json.loads(paths['provenance'].read_text())
    if hashes['trajectory'] != summary['trajectory_sha256'] or hashes['frames'] != summary['frames_sha256']:
        raise ValueError('Trajectory export differs from saved hashes')
    if hashes['prototype_asset'] != provenance['asset_sha256']:
        raise ValueError('Prototype asset differs from checkpoint provenance')
    if summary['pooling'] != 'soft' or summary['decoder'] != 'vad':
        raise ValueError('This explorer expects the saved soft VAD mixture')
    a = np.load(paths['analysis_arrays'], allow_pickle=False)
    t = np.load(paths['trajectory'], allow_pickle=False)
    p = np.load(paths['predictions'], allow_pickle=False)
    classes = a['class_order'].tolist()
    for source in [p, t]:
        np.testing.assert_array_equal(source['class_order'], classes)
        if len(set(source['sample_id'])) != len(source['sample_id']):
            raise ValueError('Duplicate sample IDs')
        if set(source['sample_id']) != set(a['sample_id']):
            raise ValueError('Different test sample inventories')
    if classes != provenance['output_class_order']:
        raise ValueError('Prototype class order differs')
    asset = json.loads(paths['prototype_asset'].read_text())
    if asset['scale'] != [-1, 1] or asset['class_order'] != classes:
        raise ValueError('Prototype asset scale or order differs')
    prototypes = np.array([[asset['labels'][word][axis] for axis in ('valence', 'arousal', 'dominance')]
                           for word in provenance['vad_word_for_each_class']], dtype=np.float32)
    frame_rows = json.loads(paths['frames'].read_text())['videos']
    frames = {v['sample_id']: v['frames'] for v in frame_rows}
    metadata = json.loads(paths['metadata'].read_text())
    ti = {str(s): i for i, s in enumerate(t['sample_id'])}
    pi = {str(s): i for i, s in enumerate(p['sample_id'])}
    counts = np.array([len(frames[str(s)]) for s in a['sample_id']])
    pred = a['prediction_on_vad_soft_tau010_s42']
    target = a['target']
    errors = kl(target, pred)
    dino_errors = kl(target, a['prediction_dino_meanpool_s42'])
    rare_mass = target[:, a['rare_indices']].sum(1)
    examples = []
    selected = select_examples(a['sample_id'], errors, rare_mass, counts, selection)
    ascending = sorted(range(len(errors)), key=lambda i: (errors[i], str(a['sample_id'][i])))
    rank = {i: k+1 for k, i in enumerate(ascending)}
    for i, group in selected:
        sid = str(a['sample_id'][i])
        j, k = ti[sid], pi[sid]
        start, end = t['offsets'][j:j+2]
        records = frames[sid]
        if len(records) != end-start:
            raise ValueError('Scene inventory differs from trajectory')
        np.testing.assert_allclose(target[i], p['target_distribution'][k], atol=1e-8)
        np.testing.assert_allclose(pred[i], p['predicted_distribution'][k], atol=1e-8)
        np.testing.assert_allclose(t['prediction'][j], pred[i], atol=2e-6)
        intervals = np.array([[seconds(r['start_time']), seconds(r['end_time'])] for r in records])
        np.testing.assert_allclose(intervals[:, 1]-intervals[:, 0], t['duration_seconds'][start:end], atol=2e-5)
        example = {
            'sample_id': sid, 'selection_group': group, **metadata[sid],
            'kl': float(errors[i]), 'dino_kl': float(dino_errors[i]),
            'kl_rank': rank[i],
            'rare_target_mass': float(rare_mass[i]),
            'error_percentile': float(100*np.mean(errors <= errors[i])),
            'intervals_seconds': intervals.tolist(),
            'scene_numbers': [str(r['scene_number']) for r in records],
            'target': target[i].tolist(), 'prediction': pred[i].tolist(),
            'dino_prediction': a['prediction_dino_meanpool_s42'][i].tolist(),
            **{key: t[key][start:end].tolist() for key in
               ['vad', 'moment_probabilities', 'weights', 'contributions']},
        }
        example['distance_squared'] = validate_example(example, prototypes, summary['temperature']).tolist()
        examples.append(example)
    payload = {
        'schema': 1, 'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'run': RUN, 'class_order': classes, 'prototypes': prototypes.tolist(),
        'prototype_source': provenance['source'], 'temperature': summary['temperature'],
        'training_mean': a['train_class_mass'].tolist(),
        'test_mean': target.mean(0).tolist(), 'test_video_count': len(target),
        'example_mean': np.mean([e['target'] for e in examples], axis=0).tolist(),
        'rare_labels': [classes[i] for i in a['rare_indices']],
        'selection_mode': selection,
        'example_count': len(examples),
        'selection': ('Ten lowest and ten highest VAD KL errors across all 2,070 test videos. '
                     'Ranked by this VAD model, not the DINO baseline. Ties use sample ID. '
                     'These extremes illustrate successes and failures, not average performance.'
                     if selection == 'kl_extremes' else
                     'Three lowest and three highest VAD KL errors, two nearest the median, '
                     'and two with highest rare target mass not already selected. Ties use sample ID. '
                     'Only clips with at least two scenes qualify. These are illustrative, not representative.'),
        'interpretation': [
            'VAD points are saved visual-model scene estimates with full-video context, not measured emotion states.',
            'Lines indicate scene order; they do not assert smooth emotional motion within scenes.',
            'Audience targets come from video-level reaction annotations, not scene-level labels.',
            'The running mixture normalizes saved contributions through the current scene; it is not online inference.',
            'All coordinates use [-1, 1]. Probabilities use softmax(-squared Euclidean distance / temperature).',
        ],
        'sources': {k: {'path': str(p), 'sha256': hashes[k]} for k, p in paths.items()},
        'checks': ['Exact saved hashes', 'Sample and class alignment', 'Scene chronology',
                   'Distance decoder matches saved probabilities', 'Weighted scenes reconstruct video predictions'],
        'examples': examples,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, separators=(',', ':'), allow_nan=False)+'\n')
    print(json.dumps({'output': str(out), 'bytes': out.stat().st_size,
                      'videos': len(examples), 'scenes': sum(len(e['vad']) for e in examples), 'sha256': sha(out)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--analysis', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--selection', choices=['diagnostic', 'kl_extremes'], default='diagnostic')
    args = parser.parse_args()
    export(args.root, args.analysis, args.out, args.selection)
