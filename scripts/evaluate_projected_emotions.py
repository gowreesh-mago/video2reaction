"""Use learned VAD -> DINO maps to place the 21 fixed emotion anchors in DINO.

This complements reconstruction: original scene DINO features are decoded by
distance to projected emotion anchors, with temperature selected on validation.
"""
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
from scipy.special import softmax
from scipy.spatial.distance import cdist

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.analyze_previous_runs import sha, dump, saved_prototypes, diagnostic, write_csv, sample_metrics, clustered_interval
from src.experiments.data import target_distribution
from src.experiments.registry import atomic_json
from src.experiments.taxonomy import REACTION_CLASSES
from src.experiments.vad_dino import Basis, RidgeMap, clip_mean

TEMPERATURES = [.003, .01, .03, .1, .3, 1., 3.]


def restore_map(folder):
    choice=json.loads((folder/'selection.json').read_text())['selected']
    basis=Basis(choice['kind'],bandwidth=choice['bandwidth'])
    model=RidgeMap()
    with np.load(folder/'parameters.npz',allow_pickle=False) as f:
        basis.mean=f['basis_mean'];basis.scale=f['basis_scale'];basis.centers=f['basis_centers']
        model.xmean=f['ridge_xmean'];model.ymean=f['ridge_ymean'];model.coef=f['ridge_coef']
    return basis,model


def pool_probabilities(distance, offsets, duration, temperature, pooling):
    probability=softmax(-distance/temperature,axis=1)
    if pooling=='uniform':return clip_mean(probability,offsets)
    if pooling!='duration':raise ValueError(pooling)
    return np.stack([np.average(probability[a:b],axis=0,weights=duration[a:b]) for a,b in zip(offsets[:-1],offsets[1:])])


def scene_distance(features, anchors, mean, kind):
    if kind=='mse':return cdist(features,anchors,'sqeuclidean')/features.shape[1]
    if kind!='centered_cosine':raise ValueError(kind)
    x=np.asarray(features,float)-mean; a=anchors-mean
    x/=np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)
    a/=np.maximum(np.linalg.norm(a,axis=1,keepdims=True),1e-12)
    return np.maximum(0,1-x@a.T)


def load_split(root,cache,source,split,limit=None):
    folder=cache/split; complete=json.loads((folder/'complete.json').read_text())
    assert sha(folder/'index.json')==complete['index_sha256']
    assert sha(folder/'features.npy')==complete['features_sha256']
    index=json.loads((folder/'index.json').read_text());ids=[v['sample_id'] for v in index['videos']]
    metadata=root/'data/metadata'/f'{split}.json';assert sha(metadata)==index['split_sha256']
    rows=json.loads(metadata.read_text());assert ids==sorted(rows)
    traj=source/split/'trajectory.npz';summary=json.loads((source/split/'trajectory_summary.json').read_text())
    assert sha(traj)==summary['trajectory_sha256']
    with np.load(traj,allow_pickle=False) as f:
        np.testing.assert_array_equal(f['sample_id'],ids);np.testing.assert_array_equal(f['offsets'],index['offsets'])
        duration=f['duration_seconds']
    n=min(limit,len(ids)) if limit else len(ids);ids=ids[:n];offsets=np.asarray(index['offsets'][:n+1])
    return {'features':np.load(folder/'features.npy',mmap_mode='r')[:offsets[-1]], 'ids':ids,'offsets':offsets,
            'duration':duration[:offsets[-1]],'target':np.stack([target_distribution(rows[i]) for i in ids]),
            'movies':np.asarray([str(rows[i]['imdbid']) for i in ids])}


def run(args):
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Run real data on an allocated Snellius node')
    root=args.root;original=args.projection;out=args.out
    if out.exists() and any(out.iterdir()):raise FileExistsError(out)
    out.mkdir(parents=True)
    parent=json.loads((original/'results.json').read_text())
    if parent['smoke'] or parent['status']!='completed':raise ValueError('Need a completed full projection run')
    prov=json.loads((original/'provenance.json').read_text());source=Path(prov['source_run']);baseline=Path(prov['baseline_run'])
    cache=Path(json.loads((source/'feature_cache.json').read_text())['path'])
    source_prov=json.loads((source/'trajectory_provenance.json').read_text())
    prototypes=saved_prototypes(root/'data/lexicons/nrc-vad-v2.1/reaction_vad.json',source_prov)
    train_meta=root/'data/metadata/train.json';assert sha(train_meta)==prov['metadata_sha256']['train']
    train_rows=json.loads(train_meta.read_text());train_targets=np.stack([target_distribution(train_rows[k]) for k in sorted(train_rows)])
    train_vad=np.load(original/'cluster_only/train/vad.npy',mmap_mode='r')
    limits={'min':np.asarray(train_vad.min(0)).tolist(),'max':np.asarray(train_vad.max(0)).tolist()}
    outside=np.any((prototypes<train_vad.min(0))|(prototypes>train_vad.max(0)),axis=1)
    val=load_split(root,cache,source,'val',1 if args.smoke else None)
    mappings={name:restore_map(original/'models'/name) for name in ['vad_linear','vad_quadratic','vad_rbf','shuffled_vad_rbf']}
    anchors={name:model.predict(basis.transform(prototypes)) for name,(basis,model) in mappings.items()}
    choices={}
    for name,points in anchors.items():
        for metric in ['mse','centered_cosine']:
            distance=scene_distance(val['features'],points,mappings[name][1].ymean,metric)
            for pooling in ['uniform','duration']:
                key=f'{name}__{metric}__{pooling}'
                candidates=[]
                for temp in TEMPERATURES:
                    q=pool_probabilities(distance,val['offsets'],val['duration'],temp,pooling)
                    candidates.append({'temperature':temp,'val_kl':float(sample_metrics(val['target'],q)['kl'].mean())})
                choices[key]={'mapping':name,'distance':metric,'pooling':pooling,'selected':min(candidates,key=lambda r:r['val_kl']),'candidates':candidates}
    # The preferred non-shuffled model is fixed before opening test metadata/features.
    preferred=min([k for k in choices if not k.startswith('shuffled')],key=lambda k:choices[k]['selected']['val_kl'])
    dump(out/'selection_frozen.json',{'selection_split':'val','preferred':preferred,'choices':choices,'test_used_in_selection':False})
    test=load_split(root,cache,source,'test',2 if args.smoke else None)
    with np.load(baseline/'test/predictions.npz',allow_pickle=False) as f:
        n=len(test['ids']);np.testing.assert_array_equal(f['sample_id'][:n],test['ids']);base_q=f['predicted_distribution'][:n]
    base_kl=sample_metrics(test['target'],base_q)['kl']
    results={};arrays={'sample_id':np.asarray(test['ids']),'class_order':np.asarray(REACTION_CLASSES),'target':test['target'],
                     'prediction_original_dino':base_q,'prototypes_vad':prototypes}
    # Same train-fitted DINO axes as the parent reconstruction experiment.
    with np.load(original/'models/pca.npz',allow_pickle=False) as f:
        components=f['components'][:3];pca_mean=f['mean']
    arrays['dino_pca3']=(clip_mean(test['features'],test['offsets'])-pca_mean)@components.T
    for name,points in anchors.items():
        arrays[name+'_anchors_pca3']=(points-pca_mean)@components.T
        for metric in ['mse','centered_cosine']:
            distance=scene_distance(test['features'],points,mappings[name][1].ymean,metric)
            for pooling in ['uniform','duration']:
                key=f'{name}__{metric}__{pooling}';choice=choices[key]
                q=pool_probabilities(distance,test['offsets'],test['duration'],choice['selected']['temperature'],pooling)
                details=diagnostic(test['target'],q,train_targets)
                errors=sample_metrics(test['target'],q)
                details['kl_difference_vs_dino']=clustered_interval(errors['kl']-base_kl,test['movies'],np.random.default_rng(parent['seed']))
                results[key]=details;arrays['prediction_'+key]=q
                dump(out/'diagnostics'/(key+'.json'),details)
                write_csv(out/'diagnostics'/(key+'_per_class.csv'),details['class_rows'])
                write_csv(out/'diagnostics'/(key+'_per_video.csv'),sorted([dict(sample_id=str(s),movie_id=str(test['movies'][i]),**{k:float(v[i]) for k,v in errors.items()}) for i,s in enumerate(test['ids'])],key=lambda r:-r['kl']))
                print(json.dumps({'method':key,'val_kl':choice['selected']['val_kl'],'test_kl':details['metrics']['kl']}),flush=True)
    np.savez_compressed(out/'plot_arrays.npz',**arrays)
    atomic_json(out/'results.json',{'status':'completed','smoke':args.smoke,'seed':parent['seed'],'preferred_by_validation':preferred,'choices':choices,'results':results,
        'fixed_prototype_source':source_prov,'source_projection':str(original),'source_projection_results_sha256':sha(original/'results.json'),
        'script_sha256':sha(__file__),'slurm_job_id':os.environ['SLURM_JOB_ID'],'train_vad_limits':limits,
        'prototype_outside_training_bounding_box':{c:bool(v) for c,v in zip(REACTION_CLASSES,outside)},
        'test_clips':len(test['ids']),'test_scenes':len(test['features']),
        'interpretation':'Fixed emotion VAD coordinates are mapped into DINO. Original visual scene features are decoded by distance to these anchors. No learned reaction classifier is used. Temperature is selected on validation; pooling averages scene probabilities uniformly or by actual scene duration. Out-of-range prototypes require extrapolation.'})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--projection',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--smoke',action='store_true')
    run(p.parse_args())
