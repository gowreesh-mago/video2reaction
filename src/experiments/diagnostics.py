"""Official scores plus explicitly separate distribution and movie diagnostics."""
from pathlib import Path
import numpy as np

from .metrics import evaluate
from .registry import atomic_json
from .taxonomy import REACTION_CLASSES


def entropy(target):
    return -(target * np.log(np.maximum(target, 1e-12))).sum(1)


def boundaries(target, counts):
    return {name: np.quantile(values, [.25, .5, .75]).tolist()
            for name, values in [('entropy', entropy(target)), ('frame_count', counts),
                                 ('dominant_probability', target.max(1))]}


def save_rare_diagnostics(directory, train_targets, targets, predictions):
    """Define rarity on training soft mass; keep diagnostics outside official scores."""
    train_mass = np.asarray(train_targets, dtype=np.float64).mean(0)
    p, q = np.asarray(targets, dtype=np.float64), np.asarray(predictions, dtype=np.float64)
    rare = np.argsort(train_mass, kind='stable')[:6]
    groups = {'rare': rare, 'common': np.setdiff1d(np.arange(len(train_mass)), rare)}
    _, classes = evaluate(p, q)
    result = {'class_order': REACTION_CLASSES, 'rarity_rule': 'Six lowest training soft-probability masses; fixed across seeds/variants',
        'train_class_mass': train_mass.tolist(), 'target_class_mass': p.mean(0).tolist(),
        'predicted_class_mass': q.mean(0).tolist(), 'class_probability_mae': np.abs(p-q).mean(0).tolist(),
        'brier_sum': float(np.square(p-q).sum(1).mean()), 'groups': {}}
    for name, positions in groups.items():
        result['groups'][name] = {'labels': [REACTION_CLASSES[i] for i in positions],
            'mean_class_probability_mae': float(np.abs(p-q)[:, positions].mean()),
            'target_mass': float(p[:, positions].sum(1).mean()),
            'predicted_mass': float(q[:, positions].sum(1).mean()), 'topk': {}}
        for k in (1, 2, 3):
            diagnostic = classes[f'top{k}']
            supported = positions[np.asarray(diagnostic['target_support'])[positions] > 0]
            result['groups'][name]['topk'][str(k)] = {
                'macro_f1_all_group_classes': float(np.asarray(diagnostic['f1'])[positions].mean()),
                'supported_classes': [REACTION_CLASSES[i] for i in supported],
                'macro_recall_supported_classes': float(np.asarray(diagnostic['recall'])[supported].mean()) if len(supported) else None}
    atomic_json(Path(directory) / 'rare_diagnostics.json', result)


def save_evaluation(out, split, ids, rows, target, predicted, counts, train_movies, bins, shuffled=None):
    out = Path(out) / split
    out.mkdir(parents=True, exist_ok=True)
    metrics, classes = evaluate(target, predicted)
    strata = {}
    variables = {'entropy': entropy(target), 'frame_count': np.asarray(counts),
                 'dominant_probability': target.max(1)}
    for name, values in variables.items():
        group = np.searchsorted(bins[name], values, side='right')
        strata[name] = {}
        for quartile in range(4):
            mask = group == quartile
            strata[name][str(quartile + 1)] = {'n': int(mask.sum()),
                'metrics': evaluate(target[mask], predicted[mask])[0] if mask.any() else None}
    movie_ids = np.asarray([str(rows[vid]['imdbid']) for vid in ids])
    seen = np.asarray([movie in train_movies for movie in movie_ids])
    strata['movie_overlap'] = {}
    for label, mask in [('seen_in_train', seen), ('unseen_in_train', ~seen)]:
        strata['movie_overlap'][label] = {'n': int(mask.sum()),
            'metrics': evaluate(target[mask], predicted[mask])[0] if mask.any() else None}
    result = {'n': len(ids), 'metrics': metrics, 'strata': strata, 'bins_from_train': bins}
    if shuffled is not None:
        result['shuffled_order_metrics'] = evaluate(target, shuffled)[0]
        result['max_probability_change_after_shuffle'] = float(np.abs(predicted - shuffled).max())
    atomic_json(out / 'metrics.json', result)
    atomic_json(out / 'per_class.json', {'class_order': REACTION_CLASSES, 'scores': classes})
    fields = {'sample_id': np.asarray(ids), 'movie_id': movie_ids, 'target_distribution': target,
              'predicted_distribution': predicted, 'class_order': np.asarray(REACTION_CLASSES),
              'target_topk': np.argsort(target, axis=1)[:, -3:][:, ::-1],
              'predicted_topk': np.argsort(predicted, axis=1)[:, -3:][:, ::-1], 'frame_count': np.asarray(counts)}
    if shuffled is not None:
        fields['shuffled_prediction'] = shuffled
    np.savez_compressed(out / 'predictions.npz', **fields)
    return result
