"""Mock checks only: no dataset downloads, model downloads or GPU required."""
import ast
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from src.experiments.losses import distribution_loss
from src.experiments.metrics import evaluate
from src.experiments.models import ReactionPredictor
from src.experiments.registry import update_registry
from src.experiments.data import frame_index, target_distribution


def test_hand_worked_metrics():
    target=np.array([[.6,.3,.1],[.1,.7,.2]])
    predicted=np.array([[.2,.7,.1],[.6,.3,.1]])
    metrics,_=evaluate(target,predicted)
    for key,value in dict(mrr=.5,tpe=.4,chebyshev=.45,cad=.5,f1_top1=0.,f1_top2=2/3,f1_top3=1.).items():
        assert metrics[key] == pytest.approx(value)


def test_metrics_match_verbatim_reference():
    # Isolate original metric source to bypass its broken legacy dataset import.
    tree=ast.parse(Path('src/metrics.py').read_text())
    names={'kl','clark','chebyshev','intersection','cosine','compute_mean_reciprocal_rank','CumulativeAbsoluteDistanceLoss'}
    source=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
    space={'np':np,'torch':torch,'nn':torch.nn,'EPS':1e-8}
    exec(compile(ast.Module(body=source,type_ignores=[]),'reference_metrics','exec'),space)
    rng=np.random.default_rng(42)
    p=rng.dirichlet(np.ones(21),size=9)
    q=rng.dirichlet(np.ones(21),size=9)
    p[0]=0; p[0,0]=1  # Includes zero target values and ties.
    q[1]=1/21
    ours,_=evaluate(p,q)
    for key in ('kl','clark','chebyshev','intersection','cosine'):
        assert ours[key] == pytest.approx(space[key](p,q),abs=1e-10)
    assert ours['mrr'] == space['compute_mean_reciprocal_rank'](q,p)
    assert ours['cad'] == pytest.approx(space['CumulativeAbsoluteDistanceLoss']().score(q,p),abs=1e-10)
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='compute_all_classification_metrics')
    loop=next(n for n in fn.body if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='top_k')
    space.update(pred_dist=q,target_dist=p,results={},f1_score=f1_score,precision_score=precision_score,
                 recall_score=recall_score,accuracy_score=accuracy_score,compute_eec_multi_label_top_k=lambda *a,**k:0)
    exec(compile(ast.Module(body=[loop],type_ignores=[]),'reference_topk','exec'),space)
    for k in (1,2,3):
        assert ours[f'f1_top{k}'] == space['results'][f'f1_weighted_top_{k}']


@pytest.mark.parametrize('kind',['ce','kl','js','cosine','composite'])
def test_losses_are_finite_and_differentiable(kind):
    logits=torch.randn(3,21,requires_grad=True)
    target=torch.zeros(3,21); target[:,0]=.7; target[:,1]=.3
    loss=distribution_loss(logits,target,kind)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()


def test_ce_kl_same_gradient():
    target=torch.rand(3,21); target/=target.sum(1,keepdim=True)
    logits=torch.randn(3,21,requires_grad=True)
    ce=torch.autograd.grad(distribution_loss(logits,target,'ce'),logits)[0]
    kl=torch.autograd.grad(distribution_loss(logits,target,'kl'),logits)[0]
    torch.testing.assert_close(ce,kl)


@pytest.mark.parametrize('aggregation',['mean','temporal','query'])
def test_padding_does_not_change_prediction(aggregation):
    torch.manual_seed(42)
    model=ReactionPredictor(16,hidden_dim=32,aggregation=aggregation).eval()
    x=torch.randn(2,4,16)
    mask=torch.tensor([[True,True,False,False],[True,True,True,True]])
    with torch.no_grad():
        before,attention=model(x,mask)
        x[0,2:]=10000
        after,_=model(x,mask)
    torch.testing.assert_close(before,after,atol=1e-6,rtol=1e-6)
    if attention is not None:
        assert not attention[0,:,2:].any()
        torch.testing.assert_close(attention.sum(-1),torch.ones_like(attention.sum(-1)))


def registry_writer(args):
    path,index=args
    update_registry(path,f'run_{index}',status='running',value=index)
    update_registry(path,f'run_{index}',status='completed')


def test_registry_concurrent_updates(tmp_path):
    path=str(tmp_path/'registry.json')
    with ProcessPoolExecutor(max_workers=4) as pool:
        list(pool.map(registry_writer,[(path,i) for i in range(16)]))
    data=json.loads(Path(path).read_text())['experiments']
    assert len(data)==16
    assert all(v['status']=='completed' and v['value']==int(k[4:]) for k,v in data.items())
    update_registry(path,'run_0',status='submitted',job_id='123')
    assert json.loads(Path(path).read_text())['experiments']['run_0']['status']=='completed'


def test_missing_or_unordered_frames_fail(tmp_path):
    folder=tmp_path/'video'; folder.mkdir()
    (folder/'index.csv').write_text('scene_number,start_frame\n0,0\n1,10\n')
    with pytest.raises(FileNotFoundError):
        frame_index(tmp_path,'video')
    (folder/'index.csv').write_text('scene_number,start_frame\n0,10\n1,0\n')
    with pytest.raises(ValueError):
        frame_index(tmp_path,'video')


def test_bad_targets_fail():
    with pytest.raises(ValueError):
        target_distribution({'reaction_outcome':{'reaction_distribution':{'joy':.5}}})
