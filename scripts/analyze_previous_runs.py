"""Audit existing predictions and export compact, reproducible error-analysis results.

Run with the original cluster uv environment. Does not train, modify checkpoints,
download data, or run a new predictor. Large image features stay on the cluster.
"""
import argparse
import csv
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.experiments.data import target_distribution
from src.experiments.metrics import evaluate, topk_mask
from src.experiments.taxonomy import REACTION_CLASSES as CLASSES

LOWER = {'chebyshev', 'kl', 'clark', 'cad', 'tpe'}
DETAIL = {
    'dino_meanpool_s42': 'DINOv2 baseline',
    'on_dino_meanpool_wide_s42': 'DINOv2 wide',
    'on_vad_soft_tau010_s42': 'VAD best condition',
    'traj_vad_soft_s42': 'VAD original soft',
    'traj_vad_sparse_s42': 'VAD original sparse',
    'on_vad_class_peak_tau010_s42': 'VAD class peak low temperature',
    'on_vad_duration_free_s42': 'Free duration',
    'traj_free_soft_s42': 'Free soft',
    'traj_free_sparse_s42': 'Free sparse',
    'b1_meanpool': 'SigLIP2 baseline',
    'e_shared_query_control': 'SigLIP2 shared attention',
    'visual_description': 'Visual + description (text-assisted)',
    'b0_prior': 'Training prior',
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def dump(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def entropy(p):
    return -(p * np.log(np.maximum(p, 1e-30))).sum(-1)


def sample_metrics(p, q):
    """Decomposable official metrics; weighted F1 is computed at dataset level."""
    p, q = np.asarray(p, dtype=np.float64), np.asarray(q, dtype=np.float64)
    dominant = p.argmax(1)
    rank = (np.argsort(q, axis=1)[:, ::-1] == dominant[:, None]).argmax(1) + 1
    m = (p + q) / 2
    return {
        'chebyshev': np.abs(p-q).max(1),
        'kl': (p * np.log(p / (q + 1e-10) + 1e-10)).sum(1),
        'clark': np.sqrt(((p-q)**2 / ((p+q)**2 + 1e-8)).sum(1)),
        'cad': np.abs(np.cumsum(p/p.sum(1, keepdims=True), 1)-np.cumsum(q/q.sum(1, keepdims=True), 1)).sum(1),
        'cosine': (p*q).sum(1)/(np.linalg.norm(p, axis=1)*np.linalg.norm(q, axis=1)),
        'intersection': np.minimum(p,q).sum(1),
        'mrr': 1/rank,
        'tpe': np.abs(p[np.arange(len(p)), dominant]-q[np.arange(len(p)), dominant]),
        'js_divergence': (entropy(m)-(entropy(p)+entropy(q))/2),
        'brier_sum': np.square(p-q).sum(1),
    }


def aligned(path, reference=None):
    with np.load(path, allow_pickle=False) as f:
        data = {k: f[k] for k in f.files}
    if len(np.unique(data['sample_id'])) != len(data['sample_id']):
        raise ValueError(f'Duplicate IDs: {path}')
    np.testing.assert_array_equal(data['class_order'], CLASSES)
    if reference is not None:
        if set(data['sample_id']) != set(reference['sample_id']):
            raise ValueError(f'Mismatched samples: {path}')
        lookup = {v: i for i,v in enumerate(data['sample_id'])}
        idx = np.array([lookup[v] for v in reference['sample_id']])
        for k in data:
            if k != 'class_order' and len(data[k]) == len(idx):
                data[k] = data[k][idx]
        for k in ('sample_id', 'movie_id', 'target_distribution', 'frame_count'):
            np.testing.assert_array_equal(data[k], reference[k])
    evaluate(data['target_distribution'], data['predicted_distribution'])
    return data


def diagnostic(p, q, train_p):
    p, q = np.asarray(p, float), np.asarray(q, float)
    score, perclass = evaluate(p, q)
    rare = np.argsort(train_p.mean(0), kind='stable')[:6]
    rows = []
    for c, label in enumerate(CLASSES):
        row = dict(label=label, rare=bool(c in rare), train_mass=float(train_p[:,c].mean()),
                   target_mass=float(p[:,c].mean()), predicted_mass=float(q[:,c].mean()),
                   bias=float((q-p)[:,c].mean()), mae=float(np.abs(q-p)[:,c].mean()),
                   rmse=float(np.sqrt(np.square(q-p)[:,c].mean())),
                   target_sd=float(p[:,c].std()), predicted_sd=float(q[:,c].std()),
                   positive_target_n=int((p[:,c]>0).sum()),
                   kl_contribution=float((p[:,c]*np.log(p[:,c]/(q[:,c]+1e-10)+1e-10)).mean()))
        for k in (1,2,3):
            for field, values in perclass[f'top{k}'].items():
                row[f'top{k}_{field}'] = float(values[c])
        rows.append(row)
    calibration = []
    for c, label in enumerate(CLASSES):
        for b in range(10):
            mask = (np.minimum((q[:,c]*10).astype(int),9) == b)
            calibration.append({'label':label, 'bin':b, 'n':int(mask.sum()),
                'predicted':float(q[mask,c].mean()) if mask.any() else None,
                'target':float(p[mask,c].mean()) if mask.any() else None})
    confusion = topk_mask(p, 1).T @ topk_mask(q,1)
    mass = p[:,rare].sum(1)
    rare_nz = mass > 0
    groups = {}
    for label, indices in [('rare',rare),('other',np.setdiff1d(np.arange(len(CLASSES)),rare))]:
        groups[label] = {'labels':[CLASSES[c] for c in indices],
            'target_mass':float(p[:,indices].sum(1).mean()),
            'predicted_mass':float(q[:,indices].sum(1).mean()),
            'mean_class_mae':float(np.abs(p-q)[:,indices].mean()),
            'macro_f1':{str(k):float(np.array(perclass[f'top{k}']['f1'])[indices].mean()) for k in (1,2,3)}}
    return {'metrics':score, 'class_rows':rows, 'calibration':calibration,
        'confusion':confusion.tolist(), 'groups':groups,
        'distribution':{'target_entropy_mean':float(entropy(p).mean()),
            'prediction_entropy_mean':float(entropy(q).mean()),
            'target_mean_max':float(p.max(1).mean()), 'prediction_mean_max':float(q.max(1).mean()),
            'mean_js_divergence':float(sample_metrics(p,q)['js_divergence'].mean()),
            'rare_present_n':int(rare_nz.sum()),
            'rare_mass_mae_when_present':float(np.abs(p[:,rare].sum(1)-q[:,rare].sum(1))[rare_nz].mean()) if rare_nz.any() else None}}


def clustered_interval(difference, movie_ids, rng, resamples=3000):
    _, inverse, counts = np.unique(movie_ids, return_inverse=True, return_counts=True)
    totals = np.bincount(inverse, weights=difference)
    weights = rng.multinomial(len(counts), np.ones(len(counts))/len(counts), size=resamples)
    draws = weights @ totals / (weights @ counts)
    return {'difference':float(difference.mean()), 'ci95':np.quantile(draws,[.025,.975]).tolist()}


def main(root, out):
    out.mkdir(parents=True, exist_ok=True)
    train_rows = json.loads((root/'data/metadata/train.json').read_text())
    test_rows = json.loads((root/'data/metadata/test.json').read_text())
    train_ids = sorted(train_rows)
    train_p = np.stack([target_distribution(train_rows[v]) for v in train_ids]).astype(float)
    train_movies = {str(v['imdbid']) for v in train_rows.values()}
    rare = np.argsort(train_p.mean(0), kind='stable')[:6]
    runs, detail, excluded, ref = [], {}, [], None
    predictions, paths = {}, {}
    for directory in sorted((root/'outputs/experiments').iterdir()):
        path = directory/'test/predictions.npz'
        if not path.exists():
            excluded.append(directory.name)
            continue
        environment = json.loads((directory/'environment.json').read_text())
        if environment['packages']['numpy'] != np.__version__:
            raise ValueError('Use original NumPy version for official tie conventions')
        name = re.sub(r'_\d+$','',directory.name)
        print('AUDIT',directory.name,flush=True)
        data = aligned(path,ref)
        if ref is None:
            ref = data
        if len(data['sample_id']) != 2070 or set(data['sample_id']) != set(test_rows):
            raise ValueError('Incomplete official test split')
        p, q = data['target_distribution'].astype(float), data['predicted_distribution'].astype(float)
        saved = json.loads((directory/'metrics.json').read_text())
        metrics, _ = evaluate(p,q)
        valdata = aligned(directory/'val/predictions.npz')
        if len(valdata['sample_id']) != 1035:
            raise ValueError('Incomplete validation split')
        valmetrics, _ = evaluate(valdata['target_distribution'], valdata['predicted_distribution'])
        for split, computed in [('test',metrics), ('val',valmetrics)]:
            for k,v in computed.items():
                np.testing.assert_allclose(v,saved[split][k],atol=1e-10,rtol=0)
        record = {'run_id':directory.name,'name':name,'val_kl':valmetrics['kl'],**metrics,
                  'rare_target_mass':float(p[:,rare].sum(1).mean()),
                  'rare_predicted_mass':float(q[:,rare].sum(1).mean()),
                  'target_entropy':float(entropy(p).mean()), 'predicted_entropy':float(entropy(q).mean()),
                  'test_prediction_sha256':sha(path)}
        runs.append(record)
        if name in DETAIL:
            d = diagnostic(p,q,train_p)
            d.update(run_id=directory.name,label=DETAIL[name],val_kl=valmetrics['kl'])
            d['strata'] = {}
            variables = {'entropy':(entropy(train_p),entropy(p)),
                         'rare_mass':(train_p[:,rare].sum(1),p[:,rare].sum(1))}
            for variable, (train_v,test_v) in variables.items():
                bins = np.quantile(train_v,[.25,.5,.75])
                group = np.searchsorted(bins,test_v,side='right')
                d['strata'][variable] = {'train_boundaries':bins.tolist(),'groups':[]}
                for b in range(4):
                    mask = group == b
                    d['strata'][variable]['groups'].append({'name':f'Q{b+1}','n':int(mask.sum()),
                        'metrics':evaluate(p[mask],q[mask])[0] if mask.any() else None})
            masks = {'rare_present':p[:,rare].sum(1)>0,
                     'rare_mass_at_least_10pct':p[:,rare].sum(1)>=.1,
                     'seen_movie':np.array([v in train_movies for v in data['movie_id']]),
                     'unseen_movie':np.array([v not in train_movies for v in data['movie_id']])}
            for c,label in enumerate(CLASSES):
                masks['dominant_'+label] = topk_mask(p,1)[:,c].astype(bool)
            for key, mask in masks.items():
                d['strata'][key] = {'n':int(mask.sum()),'metrics':evaluate(p[mask],q[mask])[0] if mask.any() else None}
            # Training-derived frame-count boundaries are already saved by each run.
            d['strata']['frame_count'] = json.loads((directory/'test/metrics.json').read_text())['strata']['frame_count']
            perclip = sample_metrics(p,q)
            failures = []
            for i,vid in enumerate(data['sample_id']):
                row = {'sample_id':str(vid),'movie_id':str(data['movie_id'][i]),
                    'movie_name':test_rows[vid]['movie_name'],'clip_name':test_rows[vid]['clip_name'],
                    'frame_count':int(data['frame_count'][i]),'target_argmax':CLASSES[p[i].argmax()],
                    'predicted_argmax':CLASSES[q[i].argmax()], 'target_entropy':float(entropy(p[i])),
                    'predicted_entropy':float(entropy(q[i])), 'rare_target_mass':float(p[i,rare].sum()),
                    'rare_predicted_mass':float(q[i,rare].sum()), **{k:float(v[i]) for k,v in perclip.items()}}
                for k in (1,2,3):
                    row[f'top{k}_overlap_count'] = int((topk_mask(p[i:i+1],k)*topk_mask(q[i:i+1],k)).sum())
                failures.append(row)
            write_csv(out/f'{name}_per_video.csv',sorted(failures,key=lambda r:-r['kl']))
            write_csv(out/f'{name}_per_class.csv',d['class_rows'])
            detail[name], predictions[name], paths[name] = d,q,directory
    if not runs or not set(DETAIL) <= set(detail):
        raise ValueError(f'Missing representative runs: {set(DETAIL)-set(detail)}')
    # Preserve single-seed diagnostics separately from mean-over-seed benchmark scores.
    groups = {}
    for row in runs:
        group = re.sub(r'_s(?:42|43|44)$','',row['name'])
        groups.setdefault(group,[]).append(row)
    group_results = {k:{'n':len(v),'runs':[r['run_id'] for r in v],
        'val_kl_mean':float(np.mean([r['val_kl'] for r in v])),
        'test_mean':{m:float(np.mean([r[m] for r in v])) for m in detail['dino_meanpool_s42']['metrics']}}
        for k,v in groups.items()}
    write_csv(out/'all_previous_runs.csv',runs)
    reference = 'dino_meanpool_s42'
    rng = np.random.default_rng(20261019)
    pairs = {}
    for name,q in predictions.items():
        if name == reference:
            continue
        a,b = sample_metrics(p,q),sample_metrics(p,predictions[reference])
        pairs[name] = {k:clustered_interval(a[k]-b[k],ref['movie_id'],rng) for k in a}
    # Compact plot source contains predictions/labels, never images or 768-D features.
    np.savez_compressed(out/'plot_arrays.npz',sample_id=ref['sample_id'],movie_id=ref['movie_id'],
        target=p,class_order=np.array(CLASSES),train_class_mass=train_p.mean(0),
        rare_indices=rare,**{'prediction_'+k:q for k,q in predictions.items()})
    dump(out/'sample_metadata.json',{v:{k:test_rows[v][k] for k in ['movie_name','clip_name','clip_description']} for v in ref['sample_id']})
    dump(out/'analysis.json',{'created_at_utc':datetime.now(timezone.utc).isoformat(),
        'numpy':np.__version__,'source_sha256':sha(__file__), 'run_count':len(runs),
        'excluded_without_test_predictions':excluded,'class_order':CLASSES,
        'rare_rule':'Six smallest TRAIN soft-label masses; not test-selected',
        'rare_labels':[CLASSES[i] for i in rare], 'test_n':len(p),'train_n':len(train_p),
        'detail':detail,'groups':group_results,'pairs_vs_dino_seed42':pairs,
        'uncertainty':'Paired movie-cluster percentile bootstrap, 3000 draws; fixed seed-42 models. Descriptive, no multiple-comparison correction. Weighted F1 is non-additive and is not bootstrapped here.',
        'paper_predictions':'Not found in inspected project outputs or original data tree. Our text-assisted model is not the paper model.',
        'checks':['All available completed test/validation metrics reproduced to 1e-10 in original NumPy.',
            'Test IDs, targets, class order, movie IDs and frame counts aligned for every run.',
            'No checkpoint training, predictor inference or parameter selection performed.'],
        'limits':['Representative plots use seed 42, not a prediction ensemble.',
            'Error analysis uses the previously exposed test set; subsequent changes require a fresh evaluation plan.',
            'Comment-derived labels are not direct measurements of felt emotion.',
            'No visual cause is inferred solely from a clip title or description.']})
    export_spaces(root,out,train_ids,train_p,ref,paths,predictions)
    dump(out/'file_manifest.json',{str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='file_manifest.json'})
    print('COMPLETE',len(runs),'previous runs',flush=True)


def export_spaces(root,out,train_ids,train_p,ref,paths,predictions):
    """Use actual saved 3-D scene coordinates; PCA is only for 768-D DINO plots."""
    directory = paths['dino_meanpool_s42']
    cache = Path(json.loads((directory/'feature_cache.json').read_text())['path'])
    embeddings, indices, arrays = {},{},{}
    for split in ('train','test'):
        index = json.loads((cache/split/'index.json').read_text())
        manifest = json.loads((cache/split/'complete.json').read_text())
        if sha(cache/split/'features.npy') != manifest['features_sha256'] or sha(cache/split/'index.json') != manifest['index_sha256']:
            raise ValueError('Feature checksum mismatch')
        x = np.load(cache/split/'features.npy',mmap_mode='r')
        z = np.stack([x[a:b].mean(0,dtype=np.float64) for a,b in zip(index['offsets'],index['offsets'][1:])])
        embeddings[split],indices[split],arrays[split] = z,index,x
    np.testing.assert_array_equal([v['sample_id'] for v in indices['train']['videos']],train_ids)
    np.testing.assert_array_equal([v['sample_id'] for v in indices['test']['videos']],ref['sample_id'])
    dino_pca = PCA(3,svd_solver='full').fit(embeddings['train'])
    label_pca = PCA(3,svd_solver='full').fit(train_p)
    p = ref['target_distribution'].astype(float)
    z = dino_pca.transform(embeddings['test'])
    export = {'sample_id':ref['sample_id'],'dino_pca3':z,'label_pca3':label_pca.transform(p)}
    info = {'dino':'Arithmetic mean of frozen frame CLS features; PCA fit on TRAIN only. Not the learned predictor hidden layer.',
            'dino_variance_retained':float(dino_pca.explained_variance_ratio_.sum()),
            'label':'PCA of the 21-D reaction distribution, fit on TRAIN only; axes differ from DINO PCA.',
            'label_variance_retained':float(label_pca.explained_variance_ratio_.sum()),'trajectories':{}}
    for name in ('on_vad_soft_tau010_s42','traj_vad_soft_s42','traj_vad_sparse_s42'):
        directory=paths[name]
        trajectory_path=directory/'test/trajectory.npz'
        summary=json.loads((directory/'test/trajectory_summary.json').read_text())
        if sha(trajectory_path)!=summary['trajectory_sha256']:
            raise ValueError('Trajectory checksum mismatch')
        with np.load(trajectory_path,allow_pickle=False) as f:
            t={k:f[k] for k in f.files}
        np.testing.assert_array_equal(t['sample_id'],ref['sample_id'])
        np.testing.assert_array_equal(t['class_order'],CLASSES)
        np.testing.assert_array_equal(t['offsets'],indices['test']['offsets'])
        if t['vad'].shape!=(len(arrays['test']),3) or not np.isfinite(t['vad']).all():
            raise ValueError('Missing or invalid saved three-dimensional coordinates')
        means=[]; spreads=[]; supports=[]; recon=[]
        for a,b in zip(t['offsets'],t['offsets'][1:]):
            mass=t['contributions'][a:b].sum(1).astype(float)
            recon.append(t['contributions'][a:b].sum(0))
            mean=np.average(t['vad'][a:b],axis=0,weights=mass)
            means.append(mean)
            spreads.append(float(np.average(((t['vad'][a:b]-mean)**2).sum(1),weights=mass)))
            supports.append(float(1/(mass**2).sum()))
        np.testing.assert_allclose(recon,predictions[name],atol=2e-6,rtol=2e-5)
        # Existing checkpoint buffer supplies exact class coordinates; no guessed lexicon values.
        import torch
        state=torch.load(directory/'best.pt',map_location='cpu',weights_only=False)
        model_state=state.get('model',state.get('model_state_dict',state))
        prototypes=model_state['prototypes'].cpu().numpy()
        export['prototypes']=prototypes
        export['label_expected_vad']=p@prototypes
        export[name+'_mean_vad']=np.array(means)
        export[name+'_spread']=np.array(spreads)
        export[name+'_effective_moments']=np.array(supports)
        errors=sample_metrics(p,predictions[name])['kl']
        order=np.argsort(errors)
        chosen=np.unique(np.r_[order[:3],order[-3:],np.argsort(p[:,np.argsort(train_p.mean(0))[:6]].sum(1))[-3:]])
        examples=[]
        for i in chosen:
            a,b=t['offsets'][i:i+2]
            examples.append({'sample_id':str(ref['sample_id'][i]),'kl':float(errors[i]),
                'vad':t['vad'][a:b].tolist(),'weights':t['weights'][a:b].tolist(),
                'dino_pca3':dino_pca.transform(arrays['test'][a:b]).tolist(),
                'frames':indices['test']['videos'][i]['frames']})
        dump(out/(name+'_trajectory_examples.json'),examples)
        eig=np.linalg.eigvalsh(np.cov(t['vad'].astype(float),rowvar=False))
        info['trajectories'][name]={'run_id':directory.name,'frames':len(t['vad']),
            'coordinate_min':t['vad'].min(0).tolist(),'coordinate_max':t['vad'].max(0).tolist(),
            'coordinate_mean':t['vad'].mean(0).tolist(),'coordinate_sd':t['vad'].std(0).tolist(),
            'covariance_eigenvalues':eig.tolist(),'mean_within_clip_variance':float(np.mean(spreads)),
            'median_effective_moments':float(np.median(supports)),
            'contributions_reconstruct_predictions':True,
            'trajectory_sha256':sha(trajectory_path),'checkpoint_sha256':sha(directory/'best.pt')}
    # Similar expected VAD can hide different label distributions (a projection diagnostic).
    expected=export['label_expected_vad']
    distance=cdist(expected,expected)
    np.fill_diagonal(distance,np.inf)
    nearest=distance.argmin(1)
    js=entropy((p+p[nearest])/2)-(entropy(p)+entropy(p[nearest]))/2
    candidates=sorted(range(len(p)),key=lambda i:-float(js[i]))
    info['expected_vad_neighbours']=[{'sample_id':str(ref['sample_id'][i]),
        'neighbour_id':str(ref['sample_id'][nearest[i]]),'vad_distance':float(distance[i,nearest[i]]),
        'js_divergence':float(js[i])} for i in candidates[:20]]
    info['interpretation']='Saved VAD is predicted, not annotated. Label expected VAD is a weighted lexicon centroid, not a measured video emotion. PCA axes are not VAD axes. No VAD-to-DINO mapping is trained in this previous-run analysis.'
    np.savez_compressed(out/'spaces.npz',**export)
    dump(out/'spaces.json',info)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    a=parser.parse_args()
    main(a.root,a.out)
