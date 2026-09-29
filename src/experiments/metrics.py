"""Required benchmark metrics with reference numerical/tie conventions preserved."""
import numpy as np
from sklearn.metrics import precision_recall_fscore_support


def topk_mask(distribution, k):
    result = np.zeros_like(distribution, dtype=int)
    # Match upstream np.argsort(...)[-k:], including zero-probability ties.
    np.put_along_axis(result, np.argsort(distribution, axis=1)[:, -k:], 1, axis=1)
    return result


def evaluate(target, predicted):
    p, q = np.asarray(target, dtype=np.float64), np.asarray(predicted, dtype=np.float64)
    if p.shape != q.shape or p.ndim != 2 or len(p) == 0:
        raise ValueError('Expected matching nonempty N x C arrays')
    for value in (p, q):
        if not np.isfinite(value).all() or (value < 0).any() or not np.allclose(value.sum(1), 1., atol=1e-5):
            raise ValueError('Invalid probability distribution')
    dominant = p.argmax(1)
    ranking = np.argsort(q, axis=1)[:, ::-1]
    rank = (ranking == dominant[:, None]).argmax(1) + 1
    metrics = {
        'chebyshev': float(np.abs(p-q).max(1).mean()),
        'kl': float((p * np.log(p / (q + 1e-10) + 1e-10)).sum(1).mean()),
        'clark': float(np.sqrt(((p-q)**2 / ((p+q)**2 + 1e-8)).sum(1)).mean()),
        # Reference CAD normalizes each row before computing cumulative sums.
        'cad': float(np.abs(np.cumsum(p/p.sum(1, keepdims=True),1) - np.cumsum(q/q.sum(1, keepdims=True),1)).sum(1).mean()),
        'cosine': float(((p*q).sum(1)/(np.linalg.norm(p,axis=1)*np.linalg.norm(q,axis=1))).mean()),
        'intersection': float(np.minimum(p,q).sum(1).mean()),
        'mrr': float((1/rank).mean()),
        'tpe': float(np.abs(p[np.arange(len(p)), dominant]-q[np.arange(len(p)), dominant]).mean()),
    }
    diagnostics = {}
    for k in (1, 2, 3):
        true, pred = topk_mask(p,k), topk_mask(q,k)
        precision, recall, f1, support = precision_recall_fscore_support(true,pred,average=None,zero_division=0)
        metrics[f'f1_top{k}'] = float(np.sum(f1 * support) / support.sum())
        diagnostics[f'top{k}'] = dict(precision=precision.tolist(), recall=recall.tolist(), f1=f1.tolist(),
                                     target_support=support.tolist(), prediction_support=pred.sum(0).tolist())
    return metrics, diagnostics
