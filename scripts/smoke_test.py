"""Three-video integration check. Training-set metrics are NOT benchmark results."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import socket
import sys
import time
import traceback
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import yaml
from src.experiments.data import load_metadata, target_distribution, frame_index, read_images
from src.experiments.losses import distribution_loss
from src.experiments.metrics import evaluate
from src.experiments.models import ReactionPredictor
from src.experiments.features import load_frozen_encoder
from src.experiments.attention import AttentionRecorder
from src.experiments.registry import atomic_json, update_registry, utc_now
from src.experiments.taxonomy import REACTION_CLASSES
from check_gpu import gpu_info


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main(args):
    cfg = yaml.safe_load(Path(args.config).read_text())
    if not torch.cuda.is_available():
        raise RuntimeError('Real smoke test requires an allocated CUDA GPU')
    info = gpu_info()
    print(json.dumps(info, indent=2), flush=True)
    seed_everything(cfg['seed'])
    torch.set_num_threads(min(4, int(os.environ.get('SLURM_CPUS_PER_TASK', '4'))))
    out = Path(os.environ.get('V2R_OUTPUT_DIR', 'outputs')) / 'smoke' / args.run_name
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Refusing to overwrite a smoke run: {out}')
    out.mkdir(parents=True, exist_ok=True)
    (out / 'config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
    code = json.loads(Path('code_version.json').read_text())
    atomic_json(out/'environment.json',info)
    atomic_json(out/'code_version.json',code)
    update_registry(args.registry,args.run_name,status='running',start_time=utc_now(),
                    job_id=os.environ.get('SLURM_JOB_ID'),hostname=socket.gethostname(),
                    git_commit=code['git_commit'],source_sha256=code['source_sha256'],
                    config=args.config,hypothesis=cfg['experiment']['hypothesis'],
                    output_dir=str(out),run_type='smoke',benchmark_result=False)
    rows, sha = load_metadata(os.environ['V2R_METADATA_DIR'],cfg['data']['split'])
    if sha != cfg['data']['metadata_sha256']:
        raise ValueError('Official split hash changed')
    ids = cfg['data']['video_ids']
    if not 2 <= len(ids) <= 3 or len(set(ids)) != len(ids):
        raise ValueError('Smoke test must use 2–3 distinct videos')
    selected, counts = [], []
    for vid in ids:
        if vid not in rows:
            raise KeyError(f'Video is not in the official training split: {vid}')
        index, count = frame_index(os.environ['V2R_FRAME_DIR'],vid,cfg['data']['max_frames'])
        selected.append(index)
        counts.append(count)
    atomic_json(out/'sample_manifest.json',dict(split='train',split_sha256=sha,
                class_order=REACTION_CLASSES,videos=[dict(sample_id=v,movie_id=rows[v]['imdbid'],
                total_keyframes=n,selected_frames=index) for v,n,index in zip(ids,counts,selected)],
                encoder=cfg['encoder'],benchmark_result=False))
    device = torch.device('cuda')
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    encoder_cfg = cfg['encoder']
    processor, encoder = load_frozen_encoder(encoder_cfg, device, dtype)
    features=[]
    t0=time.monotonic()
    for vid, index in zip(ids,selected):
        chunks=[]
        for start in range(0,len(index),encoder_cfg['batch_size']):
            images=read_images(index[start:start+encoder_cfg['batch_size']])
            inputs=processor(images=images,return_tensors='pt').to(device)
            inputs={k:v.to(dtype) if v.is_floating_point() else v for k,v in inputs.items()}
            with torch.inference_mode():
                chunks.append(encoder.get_image_features(**inputs).float().cpu())
        features.append(torch.cat(chunks))
        print(f'Encoded {vid}: {tuple(features[-1].shape)}',flush=True)
    encoder_seconds=time.monotonic()-t0
    del encoder
    torch.cuda.empty_cache()
    width=features[0].shape[-1]
    x=torch.zeros(len(ids),max(map(len,features)),width)
    mask=torch.zeros(x.shape[:2],dtype=torch.bool)
    for i,feature in enumerate(features):
        x[i,:len(feature)]=feature
        mask[i,:len(feature)]=True
    torch.save(dict(features=x,mask=mask,ids=ids,encoder=encoder_cfg),out/'smoke_features.pt')
    x,mask=x.to(device),mask.to(device)
    target=torch.tensor(np.stack([target_distribution(rows[v]) for v in ids]),device=device)
    records={}
    training=cfg['training']
    for name,variant in cfg['variants'].items():
        seed_everything(cfg['seed'])
        model=ReactionPredictor(width,training['hidden_dim'],aggregation=variant['aggregation']).to(device)
        optimizer=torch.optim.AdamW(model.parameters(),lr=training['learning_rate'],weight_decay=training['weight_decay'])
        model.eval()
        with torch.no_grad():
            first=distribution_loss(model(x,mask)[0],target,variant['loss']).item()
        losses=[]
        for step in range(training['steps']):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss=distribution_loss(model(x,mask)[0],target,variant['loss'])
            if not torch.isfinite(loss):
                raise FloatingPointError(f'{name}: nonfinite loss')
            loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True)
            optimizer.step()
            losses.append(loss.item())
        model.eval()
        with torch.no_grad():
            logits,attention=model(x,mask)
            final=distribution_loss(logits,target,variant['loss']).item()
            probabilities=logits.softmax(-1)
        if not final < first:
            raise AssertionError(f'{name}: training loss did not decrease ({first} -> {final})')
        path=out/name
        path.mkdir()
        torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),step=training['steps'],
                        seed=cfg['seed'],variant=variant,input_dim=width,config=cfg),path/'checkpoint.pt')
        restored=ReactionPredictor(width,training['hidden_dim'],aggregation=variant['aggregation']).to(device)
        checkpoint=torch.load(path/'checkpoint.pt',map_location=device,weights_only=True)
        restored.load_state_dict(checkpoint['model'])
        restored.eval()
        resumed_optimizer=torch.optim.AdamW(restored.parameters(),lr=training['learning_rate'])
        resumed_optimizer.load_state_dict(checkpoint['optimizer'])
        with torch.no_grad():
            torch.testing.assert_close(restored(x,mask)[0],logits,rtol=0,atol=0)
        # Check a resumed optimizer update is executable and finite.
        resumed_optimizer.zero_grad(set_to_none=True)
        resumed_loss=distribution_loss(restored(x,mask)[0],target,variant['loss'])
        resumed_loss.backward()
        resumed_optimizer.step()
        if not all(torch.isfinite(p).all() for p in restored.parameters()):
            raise FloatingPointError('Nonfinite resumed parameters')
        p=probabilities.detach().cpu().numpy()
        y=target.cpu().numpy()
        metrics,diagnostics=evaluate(y,p)
        np.savez_compressed(path/'predictions.npz',sample_id=np.asarray(ids),target_distribution=y,
                            predicted_distribution=p,target_topk=np.argsort(y,axis=1)[:,-3:][:,::-1],
                            predicted_topk=np.argsort(p,axis=1)[:,-3:][:,::-1],class_order=np.asarray(REACTION_CLASSES))
        if attention is not None:
            inventory = SimpleNamespace(ids=ids, counts=[len(index) for index in selected],
                frame_records=[dict(sample_id=vid, total_frames=count,
                    frames=[{k: v for k, v in frame.items() if k != 'path'} for frame in index])
                    for vid, count, index in zip(ids, counts, selected)])
            recorder = AttentionRecorder(inventory)
            recorder.add(torch.arange(len(ids)), mask, attention)
            recorder.save(path)
        record=dict(initial_loss=first,final_loss=final,loss_decreased=True,checkpoint_reload_exact=True,
                    optimizer_resume_passed=True,steps=training['steps'],metrics=metrics,
                    benchmark_result=False,evaluation_split='same three training videos')
        atomic_json(path/'metrics.json',record)
        atomic_json(path/'diagnostics.json',diagnostics)
        atomic_json(path/'loss_history.json',losses)
        records[name]=record
        print(name,json.dumps(record),flush=True)
    result=dict(status='completed',benchmark_result=False,video_count=len(ids),frame_count=int(mask.sum()),
                encoder_seconds=encoder_seconds,variants=records,environment=info)
    atomic_json(out/'metrics.json',result)
    update_registry(args.registry,args.run_name,status='completed',end_time=utc_now(),metrics=result)
    print(f'SMOKE TEST PASSED: {out}',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',default='configs/experiments/smoke_3videos.yaml')
    parser.add_argument('--run-name',required=True)
    parser.add_argument('--registry',default=os.environ.get('V2R_REGISTRY','results/experiment_registry.json'))
    args=parser.parse_args()
    try:
        main(args)
    except Exception:
        update_registry(args.registry,args.run_name,status='failed',end_time=utc_now(),error=traceback.format_exc())
        raise
