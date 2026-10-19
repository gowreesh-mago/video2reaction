"""Regression checks for diagnostic metrics and sample identity."""
import numpy as np
import pytest

from scripts.analyze_previous_runs import aligned, diagnostic, sample_metrics
from src.experiments.metrics import evaluate
from src.experiments.taxonomy import REACTION_CLASSES


def test_decomposable_metrics_preserve_official_sparse_and_tie_conventions():
    rng = np.random.default_rng(42)
    p = rng.dirichlet(np.ones(21), size=20)
    p[:3] = 0
    p[:3,:2] = .5
    q = rng.dirichlet(np.ones(21), size=20)
    q[0] = 1/21
    official, _ = evaluate(p,q)
    per_clip = sample_metrics(p,q)
    for name in official:
        if name in per_clip:
            np.testing.assert_allclose(per_clip[name].mean(), official[name], atol=1e-12)
    assert (per_clip['js_divergence'] >= -1e-12).all()


def test_rare_classes_come_from_training_not_test():
    train = np.arange(1,22,dtype=float)[None,:]
    train /= train.sum(1,keepdims=True)
    p = train[:,::-1]
    q = train.copy()
    d = diagnostic(p,q,train)
    assert d['groups']['rare']['labels'] == REACTION_CLASSES[:6]
    np.testing.assert_allclose(sum(r['bias'] for r in d['class_rows']),0,atol=1e-12)
    assert sum(map(sum,d['confusion'])) == len(p)


def test_alignment_reorders_ids_and_checks_targets(tmp_path):
    p = np.eye(21)[:3]
    ref = dict(sample_id=np.array(['a','b','c']),movie_id=np.array(['m1','m1','m2']),
        class_order=np.array(REACTION_CLASSES),target_distribution=p,
        predicted_distribution=p*.9+.1/21,frame_count=np.array([2,4,3]))
    path=tmp_path/'predictions.npz'
    shuffled = {k:v[[2,0,1]] if k!='class_order' else v for k,v in ref.items()}
    np.savez(path,**shuffled)
    actual = aligned(path,ref)
    np.testing.assert_array_equal(actual['sample_id'],ref['sample_id'])
    shuffled['target_distribution']=shuffled['target_distribution'][::-1]
    np.savez(path,**shuffled)
    with pytest.raises(AssertionError):
        aligned(path,ref)


def test_alignment_rejects_duplicate_ids(tmp_path):
    p=np.eye(21)[:2]
    path=tmp_path/'bad.npz'
    np.savez(path,sample_id=['a','a'],class_order=REACTION_CLASSES,
        target_distribution=p,predicted_distribution=p)
    with pytest.raises(ValueError,match='Duplicate'):
        aligned(path)
