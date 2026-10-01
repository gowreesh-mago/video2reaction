"""Pinned EmoSet-trained frame predictor and a cache aligned to visual image hashes."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .features import CachedVideos, ImageFiles
from .registry import atomic_json
from .runtime import cache_spec, digest_file, fingerprint


MODEL_MANIFEST = Path(__file__).resolve().parents[2] / 'assets/frame_emotion_model.json'


def model_descriptor(cfg):
    descriptor = json.loads(MODEL_MANIFEST.read_text())
    for key in ('revision', 'checkpoint_sha256'):
        if cfg['emotion_encoder'][key] != descriptor[key]:
            raise ValueError(f'Frame-emotion model {key} differs from its pinned source')
    return descriptor


def emotion_spec(cfg):
    descriptor = model_descriptor(cfg)
    return {'schema': 1, 'visual_cache_spec_sha256': fingerprint(cache_spec(cfg)),
            'implementation_sha256': digest_file(Path(__file__)),
            'uv_lock_sha256': digest_file(Path(__file__).resolve().parents[2] / 'uv.lock'),
            'model': {key: descriptor[key] for key in ('name', 'architecture', 'class_order',
                'preprocessing', 'inference', 'repository', 'revision', 'checkpoint_sha256')},
            'encoder': cfg['emotion_encoder']}


def emotion_directory(cfg):
    return Path(os.environ['V2R_EMOTION_FEATURE_DIR']) / fingerprint(emotion_spec(cfg))[:16]


def load_emotion_encoder(cfg, device):
    from torchvision import models, transforms
    descriptor = model_descriptor(cfg)
    checkpoint = Path(os.environ['V2R_EMOTION_CHECKPOINT'])
    if (checkpoint.stat().st_size != descriptor['checkpoint_bytes']
            or digest_file(checkpoint) != descriptor['checkpoint_sha256']):
        raise ValueError('Frame-emotion checkpoint checksum mismatch')
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(512, len(descriptor['class_order']))
    # Only tensor weights are accepted. No downloaded Python is executed.
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
    model = model.float().to(device).eval().requires_grad_(False)
    # Matches the author's test.py; adding ImageNet normalization would be wrong.
    transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
        transforms.ToTensor()])
    return transform, model


def probabilities(logits):
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 2 or logits.shape[1] != 8 or not len(logits) or not np.isfinite(logits).all():
        raise ValueError('Expected finite frame-emotion logits with eight columns')
    values = np.exp(logits - logits.max(1, keepdims=True))
    return values / values.sum(1, keepdims=True)


def read_coarse_vad(cfg):
    path = Path(os.environ['V2R_FRAME_VAD_FILE'])
    if digest_file(path) != cfg['emotion_vad_sha256']:
        raise ValueError('Coarse VAD asset checksum mismatch')
    asset = json.loads(path.read_text())
    classes = model_descriptor(cfg)['class_order']
    if asset['class_order'] != classes or set(asset['labels']) != set(classes):
        raise ValueError('Coarse VAD class order differs from the emotion model')
    for label in classes:
        entry = asset['labels'][label]
        if entry['source_word'] != label or entry['approximation'] is not False:
            raise ValueError('Exact sourced VAD words are required')
    values = np.asarray([[asset['labels'][c][key] for key in ('valence', 'arousal', 'dominance')]
                         for c in classes], dtype=np.float64)
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError('NRC v1 VAD must be finite and in [0,1]')
    return values


def _verify_split(directory, complete, visual):
    if complete['visual_files'] != visual.manifest['splits'][visual.split]:
        raise ValueError('Emotion cache refers to a different visual-frame inventory')
    path = directory / 'logits.npy'
    if digest_file(path) != complete['logits_sha256']:
        raise ValueError('Corrupted frame-emotion logits')
    logits = np.load(path, mmap_mode='r', allow_pickle=False)
    if logits.shape != (visual.offsets[-1], 8) or logits.dtype != np.float32 or not np.isfinite(logits).all():
        raise ValueError('Frame-emotion cache has invalid dimensions or values')
    return logits


def load_emotion_logits(cfg, visual):
    root = emotion_directory(cfg)
    manifest = json.loads((root / 'manifest.json').read_text())
    if manifest['status'] != 'completed' or manifest['spec_sha256'] != fingerprint(emotion_spec(cfg)):
        raise ValueError('Frame-emotion cache is incomplete or incompatible')
    return _verify_split(root / visual.split, manifest['splits'][visual.split], visual)


def _prefix_sha(values, cursor):
    return hashlib.sha256(memoryview(np.ascontiguousarray(values[:cursor]))).hexdigest()


def prepare_emotion_cache(cfg, splits, out, device='cuda'):
    """CPU is only for synthetic fixtures; the actual-data CLI requires SLURM/CUDA."""
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('Frame-emotion extraction requires an allocated CUDA GPU')
    root = emotion_directory(cfg)
    root.mkdir(parents=True, exist_ok=True)
    spec = emotion_spec(cfg)
    with (root / 'cache.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        spec_path = root / 'spec.json'
        if spec_path.exists() and json.loads(spec_path.read_text()) != spec:
            raise ValueError('Frame-emotion cache specification mismatch')
        atomic_json(spec_path, spec)
        existing = json.loads((root / 'manifest.json').read_text()) if (root / 'manifest.json').exists() else None
        if existing and (existing['status'] != 'completed' or existing['spec_sha256'] != fingerprint(spec)):
            raise ValueError('Invalid existing frame-emotion manifest')
        if existing and set(existing['splits']) != set(splits):
            raise ValueError('Cannot replace a completed emotion cache with a different split inventory')
        complete = {}
        transform = encoder = None
        started = time.monotonic()
        for split, rows in splits.items():
            visual = CachedVideos(cfg, split, rows)
            directory = root / split
            directory.mkdir(exist_ok=True)
            completion = directory / 'complete.json'
            if existing or completion.exists():
                done = existing['splits'][split] if existing else json.loads(completion.read_text())
                _verify_split(directory, done, visual)
                complete[split] = done
                continue
            expected_images = np.load(visual.directory / 'image_sha256.npy', mmap_mode='r', allow_pickle=False)
            paths = []
            for record in visual.frame_records:
                folder = Path(os.environ['V2R_FRAME_DIR']) / record['sample_id']
                if digest_file(folder / 'index.csv') != record['index_sha256']:
                    raise ValueError('Source keyframe index changed after visual feature extraction')
                paths.extend(folder / f"{int(frame['scene_number']) + 1:03d}.jpg" for frame in record['frames'])
            progress_path = directory / 'progress.json'
            progress = json.loads(progress_path.read_text()) if progress_path.exists() else {'next_frame': 0}
            cursor = progress['next_frame']
            if not 0 <= cursor <= len(paths):
                raise ValueError('Invalid frame-emotion resume cursor')
            path = directory / 'logits.npy'
            if cursor:
                logits = np.load(path, mmap_mode='r+')
                if (logits.shape != (len(paths), 8) or logits.dtype != np.float32
                        or _prefix_sha(logits, cursor) != progress['prefix_sha256']):
                    raise ValueError('Committed frame-emotion cache prefix is corrupted')
            else:
                logits = np.lib.format.open_memmap(path, mode='w+', dtype='float32', shape=(len(paths), 8))
            if cursor < len(paths) and encoder is None:
                transform, encoder = load_emotion_encoder(cfg, device)
            options = cfg['emotion_encoder']
            loader = DataLoader(ImageFiles(paths[cursor:]), batch_size=options['batch_size'],
                num_workers=options['workers'], collate_fn=list, persistent_workers=options['workers'] > 0)
            for batch_index, batch in enumerate(loader):
                images, hashes = zip(*batch)
                end = cursor + len(batch)
                if not np.array_equal(np.asarray(hashes, dtype='S64'), expected_images[cursor:end]):
                    raise ValueError('Image content differs from the frozen visual cache')
                inputs = torch.stack([transform(image) for image in images]).to(device)
                with torch.inference_mode():
                    values = encoder(inputs).float().cpu().numpy()
                if values.shape != (len(batch), 8) or not np.isfinite(values).all():
                    raise ValueError('Invalid frame-emotion output')
                logits[cursor:end] = values
                cursor = end
                if (batch_index + 1) % options.get('flush_batches', 128) == 0 or cursor == len(paths):
                    logits.flush()
                    with path.open('rb') as stream:
                        os.fsync(stream.fileno())
                    atomic_json(progress_path, {'next_frame': cursor, 'total_frames': len(paths),
                                              'prefix_sha256': _prefix_sha(logits, cursor)})
                    print(f'EMOTION {split}: {cursor}/{len(paths)}; elapsed={time.monotonic()-started:.1f}s', flush=True)
            if cursor != len(paths) or not np.isfinite(logits).all():
                raise ValueError('Incomplete frame-emotion cache')
            complete[split] = {'clips': len(visual), 'frames': len(paths), 'logits_sha256': digest_file(path),
                               'visual_files': visual.manifest['splits'][split]}
            atomic_json(completion, complete[split])
        manifest = {'status': 'completed', 'spec': spec, 'spec_sha256': fingerprint(spec), 'splits': complete}
        atomic_json(root / 'manifest.json', manifest)
        atomic_json(Path(out) / 'emotion_cache.json', {'path': str(root), **manifest})
        return {'benchmark_result': False, 'cache_path': str(root), 'splits': complete}
