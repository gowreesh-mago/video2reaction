"""Joint selection must receive reaction gradients and export the actual frames."""
import json

import numpy as np
import pytest
import torch

from src.experiments.highlights import masked_sparsemax
from src.experiments.losses import distribution_loss
from src.experiments.models import ReactionPredictor
from src.experiments.runtime import load_config
from src.experiments.training import fit, predict
from src.experiments import pretrained_highlights as pretrained
from src.experiments.runtime import digest_file


def test_sparsemax_known_projection_padding_translation_and_gradients():
    scores = torch.tensor([[1., .8, -2., 9000.], [3., 100., 2., 1.]], dtype=torch.double, requires_grad=True)
    mask = torch.tensor([[True, True, True, False], [True, False, False, False]])
    weights = masked_sparsemax(scores, mask)
    torch.testing.assert_close(weights, torch.tensor([[.6, .4, 0., 0.], [1., 0., 0., 0.]], dtype=torch.double))
    torch.testing.assert_close(weights, masked_sparsemax(scores + 57, mask))
    assert torch.autograd.gradcheck(lambda x: masked_sparsemax(x, mask), (scores,))
    torch.testing.assert_close(masked_sparsemax(torch.zeros(1, 4), torch.ones(1, 4, dtype=torch.bool)), torch.full((1, 4), .25))
    with pytest.raises(ValueError, match='valid frame'):
        masked_sparsemax(scores, torch.zeros_like(mask))


@pytest.mark.parametrize('aggregation', ['highlight_sparse', 'highlight_soft'])
def test_selector_is_jointly_trained_and_is_the_only_pooling_path(aggregation):
    torch.manual_seed(43)
    model = ReactionPredictor(16, 32, aggregation=aggregation)
    x = torch.randn(3, 9, 16)
    mask = torch.tensor([[True] * 5 + [False] * 4, [True] * 9, [True] * 3 + [False] * 6])
    target = torch.randn(3, 21).softmax(-1)
    logits, attention = model(x, mask)
    expected = model.head((model.projection(x) * attention[:, 0, :, None]).sum(1))
    torch.testing.assert_close(logits, expected, rtol=0, atol=0)
    distribution_loss(logits, target, 'kl').backward()
    for component in (model.highlight_scorer, model.projection, model.head):
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in component.parameters())
        assert sum(p.grad.abs().sum().item() for p in component.parameters()) > 0
    assert not attention.masked_select(~mask[:, None, :]).any()
    torch.testing.assert_close(attention.sum(-1), torch.ones(3, 21))
    changed = x.clone(); changed[~mask] = 1e5
    torch.testing.assert_close(logits, model(changed, mask)[0], atol=1e-6, rtol=1e-6)
    perm = torch.tensor([5, 1, 8, 0, 4, 2, 3, 6, 7])
    other_logits, other_attention = model(x[:, perm], mask[:, perm])
    torch.testing.assert_close(logits, other_logits, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(attention[:, :, perm], other_attention, atol=1e-6, rtol=1e-6)


def test_sparse_and_soft_match_capacity_initialization_and_config():
    torch.manual_seed(42); sparse = ReactionPredictor(16, 32, aggregation='highlight_sparse')
    torch.manual_seed(42); soft = ReactionPredictor(16, 32, aggregation='highlight_soft')
    torch.manual_seed(42); mean = ReactionPredictor(16, 32)
    assert sparse.state_dict().keys() == soft.state_dict().keys()
    for key, value in sparse.state_dict().items():
        torch.testing.assert_close(value, soft.state_dict()[key], rtol=0, atol=0)
        if key in mean.state_dict():
            torch.testing.assert_close(value, mean.state_dict()[key], rtol=0, atol=0)
    configs = [load_config('configs/experiments/' + n + '.yaml') for n in
               ('joint_highlight_sparse', 'joint_highlight_soft_control')]
    for cfg in configs:
        cfg.pop('experiment'); cfg['model'].pop('aggregation')
    assert configs[0] == configs[1]


class HighlightVideos:
    def __init__(self):
        self.ids = ['a', 'b', 'c']
        self.counts = np.array([4, 7, 5])
        rng = torch.Generator().manual_seed(41)
        self.frames = [torch.randn(int(n), 16, generator=rng) for n in self.counts]
        self.targets = torch.randn(3, 21, generator=rng).softmax(-1).numpy()
        self.frame_records = [{'sample_id': vid, 'frames': [{'index': i, 'start_time': str(i * .5), 'start_frame': str(i * 10)} for i in range(n)]} for vid,n in zip(self.ids,self.counts)]

    def __len__(self):
        return 3

    def __getitem__(self, i):
        return self.frames[i], torch.from_numpy(self.targets[i]), i


@pytest.mark.parametrize('name', ['joint_highlight_sparse', 'joint_highlight_soft_control'])
def test_training_reload_and_exact_highlight_export(tmp_path, name):
    torch.set_num_threads(1)
    cfg = load_config(f'configs/experiments/{name}.yaml')
    cfg['model'].update(input_dim=16, hidden_dim=32)
    cfg['training'].update(epochs=3, batch_size=2)
    dataset = HighlightVideos()
    model, info = fit(cfg, dataset, dataset, tmp_path, 'cpu', 'synthetic')
    assert info['checkpoint_reload_exact']
    predictions = predict(model, dataset, cfg, 'cpu', attention_output=tmp_path/'test')
    np.testing.assert_array_equal(predictions, predict(model, dataset, cfg, 'cpu'))
    saved = json.loads((tmp_path/'test/highlights.json').read_text())
    with np.load(tmp_path/'test/attention.npz') as attention:
        for i,record in enumerate(saved['videos']):
            lo,hi=attention['offsets'][i:i+2]; weights=attention['weights'][lo:hi,0]
            np.testing.assert_array_equal(weights,record['weights'])
            chosen=np.flatnonzero(weights>0).tolist()
            assert record['selected_positions']==chosen
            assert record['selected_frames']==[dataset.frame_records[i]['frames'][p] for p in chosen]
    assert (tmp_path/'test/highlight_statistics.json').is_file()


def test_pretrained_ranking_has_stable_ties_and_checks_k():
    np.testing.assert_array_equal(pretrained.ranked_positions([.2,.8,.8,.1],1),[1])
    np.testing.assert_array_equal(pretrained.ranked_positions([.2,.8,.8,.1],3),[0,1,2])
    with pytest.raises(ValueError,match='finite'):
        pretrained.ranked_positions([float('nan')],1)
    with pytest.raises(ValueError,match='positive'):
        pretrained.ranked_positions([.5],True)


def test_pretrained_cache_resume_alignment_selection_and_corruption(tmp_path, monkeypatch):
    from PIL import Image
    monkeypatch.setenv('V2R_HIGHLIGHT_FEATURE_DIR',str(tmp_path/'cache'))
    monkeypatch.setenv('V2R_FRAME_DIR',str(tmp_path/'frames'))
    directory=tmp_path/'visual';directory.mkdir()
    records=[];hashes=[]
    for vid in ('a','b'):
        folder=tmp_path/'frames'/vid;folder.mkdir(parents=True)
        (folder/'index.csv').write_text('synthetic chronological index\n')
        frames=[]
        for i in range(3):
            path=folder/f'{i+1:03d}.jpg';Image.new('RGB',(8,8),(20*i,0,0)).save(path)
            hashes.append(digest_file(path));frames.append({'scene_number':str(i),'start_time':str(i*.5),'start_frame':str(i*10)})
        records.append({'sample_id':vid,'frames':frames,'index_sha256':digest_file(folder/'index.csv')})
    np.save(directory/'image_sha256.npy',np.asarray(hashes,dtype='S64'))
    class Visual:
        ids=['a','b'];offsets=[0,3,6];counts=np.array([3,3]);targets=np.ones((2,21),dtype=np.float32)/21
        split='train';manifest={'splits':{'train':{'identity':'mock'}}};frame_records=records
        def __len__(self):return 2
        def __getitem__(self,i):return torch.arange(3*16).reshape(3,16).float(),torch.from_numpy(self.targets[i]),i
    visual=Visual();visual.directory=directory
    monkeypatch.setattr(pretrained,'CachedVideos',lambda *args:visual)
    monkeypatch.setattr(pretrained,'load_highlight_teacher',lambda device:None)
    calls=[]
    def scores(images,parts,device):
        calls.append(len(images))
        if len(calls)==2:raise RuntimeError('synthetic interruption')
        return np.array([.1,.9,.2],dtype=np.float32),np.array([[0,1],[0,3],[1,3]],dtype=np.float32)
    monkeypatch.setattr(pretrained,'teacher_scores',scores)
    cfg=load_config('configs/experiments/pretrained_dsnet_k4.yaml');cfg['pretrained_highlight']['k']=1
    out=tmp_path/'out';out.mkdir()
    with pytest.raises(RuntimeError,match='interruption'):
        pretrained.prepare_highlight_cache(cfg,{'train':{}},out,device='cpu')
    pretrained.prepare_highlight_cache(cfg,{'train':{}},out,device='cpu')
    assert len(calls)==3  # Completed first clip was reused after interruption.
    subset=pretrained.PretrainedHighlightVideos(cfg,visual)
    np.testing.assert_array_equal(subset.stratum_counts,[3,3])
    np.testing.assert_array_equal(subset.counts,[1,1])
    torch.testing.assert_close(subset[0][0],visual[0][0][[1]])
    subset.save_evidence(tmp_path/'evaluation')
    saved=json.loads((tmp_path/'evaluation/pretrained_highlights.json').read_text())
    assert saved['selection_uses_targets'] is False
    assert saved['videos'][0]['selected_frames']==[records[0]['frames'][1]]
    pretrained.prepare_highlight_cache(cfg,{'train':{}},out,device='cpu');assert len(calls)==3
    path=pretrained.highlight_directory(cfg)/'train/a.npz'
    with path.open('ab') as stream:stream.write(b'corrupted')
    with pytest.raises(ValueError,match='Corrupted'):
        pretrained.PretrainedHighlightVideos(cfg,visual)


def test_pretrained_config_keeps_b1_predictor_and_training_fixed():
    baseline=load_config('configs/experiments/b1_meanpool.yaml')
    teacher=load_config('configs/experiments/pretrained_dsnet_k4.yaml')
    baseline.pop('experiment');teacher.pop('experiment')
    assert teacher.pop('pretrained_highlight')=={'k':4}
    assert baseline==teacher


def test_launch_gate_requires_the_same_source_and_all_smoke_variants(tmp_path, monkeypatch):
    from scripts import submit_highlights as submit
    from src.experiments.registry import atomic_json
    monkeypatch.chdir(tmp_path);monkeypatch.setenv('V2R_ROOT',str(tmp_path))
    monkeypatch.setattr(submit,'require_completed_cache',lambda job:None)
    out=tmp_path/'outputs/smoke/smoke_highlights_3videos_123';out.mkdir(parents=True)
    atomic_json(tmp_path/'code_version.json',{'source_sha256':'current'})
    atomic_json(out/'code_version.json',{'source_sha256':'current'})
    variants={name:{'loss_decreased':True,'checkpoint_reload_exact':True,'optimizer_resume_passed':True}
              for name in ('meanpool','joint_highlight_sparse','joint_highlight_soft_control','pretrained_dsnet_k4')}
    smoke={'status':'completed','benchmark_result':False,'video_count':3,'frame_count':24,'variants':variants}
    atomic_json(out/'metrics.json',smoke)
    submit.require_highlight_smoke('123')
    atomic_json(out/'code_version.json',{'source_sha256':'old'})
    with pytest.raises(RuntimeError,match='source'):
        submit.require_highlight_smoke('123')
    atomic_json(out/'code_version.json',{'source_sha256':'current'})
    variants.pop('pretrained_dsnet_k4');atomic_json(out/'metrics.json',smoke)
    with pytest.raises(RuntimeError,match='experiment set'):
        submit.require_highlight_smoke('123')
