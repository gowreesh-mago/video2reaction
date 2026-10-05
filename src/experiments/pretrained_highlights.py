"""Frozen DSNet pseudo-labels aligned to the existing SigLIP frame inventory."""
import copy
import fcntl
import inspect
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch

from .features import CachedVideos, ImageFiles
from .registry import atomic_json
from .runtime import cache_spec, digest_file, fingerprint


DESCRIPTOR = Path(__file__).resolve().parents[2] / 'assets/highlight_model.json'


def highlight_spec(cfg):
    return {'schema': 1, 'model': json.loads(DESCRIPTOR.read_text()),
        'visual_cache_spec_sha256': fingerprint(cache_spec(cfg)),
        'implementation_sha256': digest_file(Path(__file__)),
        'uv_lock_sha256': digest_file(Path(__file__).resolve().parents[2] / 'uv.lock')}


def highlight_directory(cfg):
    return Path(os.environ['V2R_HIGHLIGHT_FEATURE_DIR']) / fingerprint(highlight_spec(cfg))[:16]


def load_highlight_teacher(device):
    from torchvision import models, transforms
    spec = json.loads(DESCRIPTOR.read_text())
    root = Path(os.environ['V2R_HIGHLIGHT_MODEL_DIR'])
    for path, sha in [(root/'tvsum0.pt', spec['checkpoint_sha256']),
                      (root/'googlenet.pth', spec['backbone_sha256'])] + [
                      (root/'source'/name, sha) for name, sha in spec['source_files'].items()]:
        if digest_file(path) != sha:
            raise ValueError(f'Changed pretrained highlight asset: {path}')
    source = root/'source/src'
    sys.path.insert(0, str(source))
    try:
        from anchor_free.dsnet_af import DSNetAF
    finally:
        sys.path.remove(str(source))
    if Path(inspect.getfile(DSNetAF)).resolve() != (source/'anchor_free/dsnet_af.py').resolve():
        raise ValueError('DSNet import resolved outside the verified source')
    for module in ('anchor_free.dsnet_af', 'anchor_free.anchor_free_helper', 'helpers.bbox_helper', 'modules.models'):
        if Path(sys.modules[module].__file__).resolve() != (source / (module.replace('.', '/') + '.py')).resolve():
            raise ValueError(f'DSNet dependency resolved outside the verified source: {module}')
    teacher = DSNetAF('attention', 1024, 128, 8)
    teacher.load_state_dict(torch.load(root/'tvsum0.pt', map_location='cpu', weights_only=True), strict=True)
    backbone = models.googlenet(weights=None, aux_logits=True, transform_input=False, init_weights=False)
    backbone.load_state_dict(torch.load(root/'googlenet.pth', map_location='cpu', weights_only=True), strict=True)
    backbone.aux_logits = False
    backbone.aux1 = backbone.aux2 = None
    # Match upstream's sequential pool5 extractor, which bypasses transform_input.
    backbone = torch.nn.Sequential(*list(backbone.children())[:-2])
    transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize(mean=[.485, .456, .406], std=[.229, .224, .225])])
    return transform, backbone.to(device).eval().requires_grad_(False), teacher.to(device).eval().requires_grad_(False)


def teacher_scores(images, teacher_parts, device, batch_size=32):
    transform, backbone, teacher = teacher_parts
    features = []
    with torch.inference_mode():
        for start in range(0, len(images), batch_size):
            x = torch.stack([transform(im) for im in images[start:start + batch_size]]).to(device)
            h = backbone(x).flatten(1)
            if h.shape[1] != 1024:
                raise ValueError('DSNet requires its pretrained 1024-dimensional GoogLeNet features')
            features.append(h / (torch.linalg.vector_norm(h, dim=1, keepdim=True) + 1e-10))
        scores, boxes = teacher.predict(torch.cat(features)[None])
    scores, boxes = np.asarray(scores, dtype=np.float32), np.asarray(boxes, dtype=np.float32)
    if (scores.shape != (len(images),) or boxes.shape != (len(images), 2)
            or not np.isfinite(scores).all() or not np.isfinite(boxes).all()
            or (scores < 0).any() or (scores > 1.000001).any()):
        raise ValueError('Invalid DSNet pseudo-labels')
    return scores, boxes


def ranked_positions(scores, k):
    scores = np.asarray(scores)
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError('Expected finite one-dimensional highlight scores')
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError('K must be a positive integer')
    return np.sort(np.argsort(-scores, kind='stable')[:min(k, len(scores))])


def _clip_identity(spec_sha, record, image_hashes):
    return fingerprint({'spec_sha256': spec_sha, 'frames': record, 'images': image_hashes.tolist()})


def _load_clip(path, identity, count, expected_sha=None):
    if expected_sha is not None and digest_file(path) != expected_sha:
        raise ValueError('Corrupted pretrained highlight cache')
    with np.load(path, allow_pickle=False) as f:
        if str(f['identity']) != identity:
            raise ValueError('Highlight cache frame identity mismatch')
        scores, boxes = f['scores'].copy(), f['boxes'].copy()
    if (scores.shape != (count,) or boxes.shape != (count, 2)
            or not np.isfinite(scores).all() or not np.isfinite(boxes).all()
            or (scores < 0).any() or (scores > 1.000001).any()):
        raise ValueError('Invalid cached highlight scores')
    return scores, boxes


def prepare_highlight_cache(cfg, splits, out, device='cuda'):
    root = highlight_directory(cfg); root.mkdir(parents=True, exist_ok=True)
    spec = highlight_spec(cfg); sha = fingerprint(spec)
    with (root/'cache.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (root/'spec.json').exists() and json.loads((root/'spec.json').read_text()) != spec:
            raise ValueError('Pretrained highlight specification mismatch')
        atomic_json(root/'spec.json', spec)
        complete = json.loads((root/'manifest.json').read_text()) if (root/'manifest.json').exists() else None
        if complete and (complete['spec_sha256'] != sha or complete['status'] != 'completed'
                         or set(complete['splits']) != set(splits)):
            raise ValueError('Completed highlight cache cannot change its split inventory')
        teacher_parts = None; records = {}
        for split, rows in splits.items():
            visual = CachedVideos(cfg, split, rows)
            hashes = np.load(visual.directory/'image_sha256.npy', allow_pickle=False)
            directory = root/split; directory.mkdir(exist_ok=True)
            entries = {}
            for i, record in enumerate(visual.frame_records):
                lo,hi=visual.offsets[i:i+2]; vid=record['sample_id']
                identity = _clip_identity(sha, record, hashes[lo:hi].astype(str))
                path = directory/f'{vid}.npz'
                receipt = directory/f'{vid}.json'
                if complete or receipt.exists():
                    previous = (complete['splits'][split]['clips'][vid]['sha256'] if complete
                                else json.loads(receipt.read_text())['sha256'])
                    _load_clip(path, identity, hi-lo, previous)
                else:
                    if complete:
                        raise ValueError('A completed highlight cache lost a clip')
                    folder = Path(os.environ['V2R_FRAME_DIR'])/vid
                    if digest_file(folder/'index.csv') != record['index_sha256']:
                        raise ValueError('Source frame index changed')
                    inputs = ImageFiles([folder/f"{int(frame['scene_number']) + 1:03d}.jpg" for frame in record['frames']])
                    images, actual_hashes = zip(*[inputs[j] for j in range(len(inputs))])
                    if not np.array_equal(np.asarray(actual_hashes, dtype='S64'), hashes[lo:hi]):
                        raise ValueError('Highlight images differ from the visual cache')
                    if teacher_parts is None:
                        teacher_parts = load_highlight_teacher(device)
                    scores, boxes = teacher_scores(images, teacher_parts, device)
                    fd, temporary = tempfile.mkstemp(prefix='.clip-', dir=directory)
                    with os.fdopen(fd, 'wb') as stream:
                        np.savez_compressed(stream, identity=np.asarray(identity), scores=scores, boxes=boxes)
                        stream.flush(); os.fsync(stream.fileno())
                    os.replace(temporary, path)
                    atomic_json(receipt, {'identity':identity,'sha256':digest_file(path)})
                entries[vid]={'sha256':digest_file(path), 'identity':identity, 'frames':int(hi-lo)}
                if (i+1)%100==0 or i+1==len(visual):
                    print(f'HIGHLIGHTS {split}: {i+1}/{len(visual)} clips', flush=True)
            records[split]={'visual_files':visual.manifest['splits'][split], 'clips':entries}
        manifest={'status':'completed','spec_sha256':sha,'spec':spec,'splits':records}
        atomic_json(root/'manifest.json',manifest)
        atomic_json(Path(out)/'highlight_cache.json',{'path':str(root),**manifest})
        return {'benchmark_result':False,'cache_path':str(root),'clips':{s:len(v['clips']) for s,v in records.items()}}


class PretrainedHighlightVideos:
    def __init__(self, cfg, visual):
        self.visual=visual; self.ids=visual.ids; self.targets=visual.targets
        self.stratum_counts=visual.counts.copy(); self.k=cfg['pretrained_highlight']['k']
        root=highlight_directory(cfg); self.manifest=json.loads((root/'manifest.json').read_text())
        sha=fingerprint(highlight_spec(cfg))
        if self.manifest['status']!='completed' or self.manifest['spec_sha256']!=sha:
            raise ValueError('Pretrained highlight cache is incomplete or incompatible')
        split=self.manifest['splits'][visual.split]
        if split['visual_files']!=visual.manifest['splits'][visual.split] or set(split['clips'])!=set(self.ids):
            raise ValueError('Highlight cache differs from the visual-frame inventory')
        hashes=np.load(visual.directory/'image_sha256.npy',allow_pickle=False)
        self.selected=[];self.frame_records=[];self.scores=[];self.boxes=[]
        for i,record in enumerate(visual.frame_records):
            lo,hi=visual.offsets[i:i+2];vid=record['sample_id']; entry=split['clips'][vid]
            identity=_clip_identity(sha,record,hashes[lo:hi].astype(str))
            if entry['identity']!=identity or entry['frames']!=hi-lo:
                raise ValueError('Highlight cache clip identity mismatch')
            scores,boxes=_load_clip(root/visual.split/f'{vid}.npz',identity,hi-lo,entry['sha256'])
            chosen=ranked_positions(scores,self.k)
            self.scores.append(scores);self.boxes.append(boxes);self.selected.append(chosen)
            selected_record=copy.deepcopy(record);selected_record['frames']=[record['frames'][j] for j in chosen]
            self.frame_records.append(selected_record)
        self.counts=np.asarray([len(x) for x in self.selected])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self,i):
        frames,target,index=self.visual[i]
        return frames[self.selected[i]],target,index

    def save_evidence(self,directory):
        directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
        records=[]
        for i,vid in enumerate(self.ids):
            chosen=self.selected[i]
            records.append({'sample_id':vid,'scores':self.scores[i].tolist(),
                'proposal_boxes_keyframe_positions':self.boxes[i].tolist(),
                'selected_positions':chosen.tolist(),'selected_frames':self.frame_records[i]['frames'],
                'frame_count':int(self.stratum_counts[i]),'pooling_weights':[1/len(chosen)]*len(chosen)})
        atomic_json(directory/'pretrained_highlights.json',{'schema':1,'k':self.k,'selection_uses_targets':False,
            'cache_spec_sha256':self.manifest['spec_sha256'],'model':self.manifest['spec']['model'],
            'selection_rule':'Top-K center confidence; ties choose earlier frames; retain chronological order',
            'videos':records})
