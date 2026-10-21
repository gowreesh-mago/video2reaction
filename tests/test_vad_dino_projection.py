import numpy as np
from src.experiments.vad_dino import Basis, RidgeMap, scene_sample, reconstruction_metrics, retrieval


def test_sample_equal_clip_weight_and_no_crossing():
    indices, weights = scene_sample([0, 3, 20], 8)
    assert len(indices) == 11
    assert len(np.unique(indices)) == 11
    np.testing.assert_allclose(weights[indices < 3].sum(), .5)
    np.testing.assert_allclose(weights[indices >= 3].sum(), .5)


def test_linear_recovers_heldout_mapping_without_refitting():
    rng = np.random.default_rng(17)
    x = rng.normal(size=(300, 3)); y = x @ rng.normal(size=(3, 40)) + 5
    w = np.ones(200) / 200
    basis = Basis('linear').fit(x[:200], w)
    model = RidgeMap().fit(basis.transform(x[:200]), y[:200], w, 1e-9)
    before = basis.mean.copy()
    np.testing.assert_allclose(model.predict(basis.transform(x[200:])), y[200:], atol=1e-6)
    basis.transform(x[200:] + 100)
    np.testing.assert_array_equal(before, basis.mean)


def test_quadratic_recovers_nonlinear_signal():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(200, 3)); y = (x[:, 0] * x[:, 1])[:, None]
    w = np.ones(150) / 150
    basis = Basis('quadratic').fit(x[:150], w)
    model = RidgeMap().fit(basis.transform(x[:150]), y[:150], w, 1e-9)
    np.testing.assert_allclose(model.predict(basis.transform(x[150:])), y[150:], atol=1e-6)


def test_reconstruction_zero_and_perfect_controls():
    y = np.array([[1., 2], [3, 4], [5, 6]])
    mean = np.array([2., 2.])
    null, _ = reconstruction_metrics(y, np.broadcast_to(mean, y.shape), mean, [0, 2, 3])
    exact, _ = reconstruction_metrics(y, y, mean, [0, 2, 3])
    assert null['r2_vs_train_mean'] == 0
    assert exact['r2_vs_train_mean'] == 1


def test_retrieval_perfect_and_constant_not_order_dependent():
    y = np.eye(30); mean = np.zeros(30)
    assert retrieval(y, y, mean)['recall_at_1'] == 1
    constant = retrieval(y, np.zeros_like(y), mean)
    assert constant['recall_at_10'] == 0
    assert constant['median_rank'] == 15.5


def test_rbf_landmarks_use_training_only_and_are_reproducible():
    x = np.random.default_rng(3).normal(size=(30, 3)); w = np.ones(30) / 30
    a, b = Basis('rbf', landmarks=12).fit(x, w), Basis('rbf', landmarks=12).fit(x, w)
    np.testing.assert_array_equal(a.transform(x), b.transform(x))
    assert a.transform(x).shape == (30, 15)
