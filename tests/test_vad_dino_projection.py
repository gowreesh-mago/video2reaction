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


def test_frozen_head_keeps_per_scene_normalization_before_pooling():
    import torch
    from src.experiments.models import ReactionPredictor
    from src.experiments.features import collate_videos
    from scripts.run_vad_dino_projection import head_prediction
    torch.manual_seed(5)
    model = ReactionPredictor(input_dim=12, hidden_dim=8).eval()
    a, b = torch.randn(3, 12), torch.randn(5, 12) * 4 + 2
    batch = [(a, torch.ones(21)/21, 0), (b, torch.ones(21)/21, 1)]
    x, mask, _, _ = collate_videos(batch)
    with torch.inference_mode(): expected = model(x, mask)[0].softmax(-1).numpy()
    actual = head_prediction(model, torch.cat([a, b]).numpy(), [0, 3, 8], torch)
    np.testing.assert_allclose(actual, expected, atol=1e-7)


def test_validation_selection_is_the_best_saved_candidate(tmp_path):
    import json
    from scripts.run_vad_dino_projection import choose_mapping
    rng = np.random.default_rng(2)
    train = rng.normal(size=(40, 3)); val = rng.normal(size=(15, 3))
    weights = rng.normal(size=(3, 10))
    _, _, chosen = choose_mapping('linear', train, train@weights, np.ones(40),
                                  val, val@weights, np.ones(15), 42, tmp_path)
    saved = json.loads((tmp_path/'selection.json').read_text())
    assert chosen == min(saved['candidates'], key=lambda r:r['val_mse'])
    assert saved['selection_split'] == 'val'


def test_projected_anchor_distances_and_duration_mixture():
    from scripts.evaluate_projected_emotions import scene_distance, pool_probabilities
    anchors=np.array([[1.,0.],[0.,1.]])
    distance=scene_distance(anchors,anchors,np.zeros(2),'centered_cosine')
    np.testing.assert_allclose(distance,[[0,1],[1,0]])
    uniform=pool_probabilities(distance,[0,2],np.array([6.,3.]),.001,'uniform')
    duration=pool_probabilities(distance,[0,2],np.array([6.,3.]),.001,'duration')
    np.testing.assert_allclose(uniform,[[.5,.5]])
    np.testing.assert_allclose(duration,[[2/3,1/3]])


def test_saved_projection_parameters_round_trip(tmp_path):
    from scripts.run_vad_dino_projection import choose_mapping
    from scripts.evaluate_projected_emotions import restore_map
    rng=np.random.default_rng(9);x=rng.normal(size=(30,3));y=rng.normal(size=(30,10))
    b,m,_=choose_mapping('rbf',x[:20],y[:20],np.ones(20),x[20:],y[20:],np.ones(10),42,tmp_path)
    restored_b,restored_m=restore_map(tmp_path)
    np.testing.assert_allclose(restored_m.predict(restored_b.transform(x)),m.predict(b.transform(x)),atol=0,rtol=0)
