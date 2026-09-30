"""Resumable full-frame feature extraction and verified cache loading."""
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset

from .data import frame_index, target_distribution
from .registry import atomic_json
from .runtime import cache_directory, cache_spec, digest_file, fingerprint


class ImageFiles(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        raw = Path(self.paths[index]).read_bytes()
        with Image.open(io.BytesIO(raw)) as image:
            rgb = image.convert('RGB')
        return rgb, hashlib.sha256(raw).hexdigest()


def prepare_features(cfg, splits, out):
    from transformers import AutoModel, AutoProcessor
    if not torch.cuda.is_available():
        raise RuntimeError('Feature preparation requires an allocated CUDA GPU')
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError('The pinned feature cache requires bf16 support')
    root = cache_directory(cfg)
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'cache.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        spec = cache_spec(cfg)
        spec_path = root / 'spec.json'
        if spec_path.exists() and json.loads(spec_path.read_text()) != spec:
            raise ValueError('Cache specification mismatch')
        atomic_json(spec_path, spec)
        encoder_cfg = cfg['encoder']
        kwargs = {'revision': encoder_cfg['revision'], 'local_files_only': encoder_cfg['local_files_only']}
        processor = AutoProcessor.from_pretrained(encoder_cfg['model'], use_fast=encoder_cfg['use_fast'], **kwargs)
        encoder = AutoModel.from_pretrained(encoder_cfg['model'], torch_dtype=torch.bfloat16, **kwargs).cuda().eval()
        encoder.requires_grad_(False)
        atomic_json(root / 'processor.json', processor.image_processor.to_dict())
        complete = {}
        started = time.monotonic()
        for split, rows in splits.items():
            directory = root / split
            directory.mkdir(exist_ok=True)
            videos, paths, offsets = [], [], [0]
            for vid in sorted(rows):
                index, count = frame_index(os.environ['V2R_FRAME_DIR'], vid, cfg['data']['max_frames'])
                paths.extend(row['path'] for row in index)
                offsets.append(len(paths))
                videos.append({'sample_id': vid, 'total_frames': count,
                               'frames': [{k: v for k, v in row.items() if k != 'path'} for row in index],
                               'index_sha256': digest_file(Path(os.environ['V2R_FRAME_DIR']) / vid / 'index.csv')})
            index_info = {'videos': videos, 'offsets': offsets, 'split_sha256': cfg['data']['split_sha256'][split]}
            index_path = directory / 'index.json'
            if index_path.exists() and json.loads(index_path.read_text()) != index_info:
                raise ValueError(f'{split}: frame index changed since cache creation')
            atomic_json(index_path, index_info)
            progress_path = directory / 'progress.json'
            progress = json.loads(progress_path.read_text()) if progress_path.exists() else {'next_frame': 0}
            next_frame = progress['next_frame']
            if not 0 <= next_frame <= len(paths):
                raise ValueError('Invalid feature cache progress')
            shape = (len(paths), encoder_cfg['feature_dim'])
            feature_path, hashes_path = directory / 'features.npy', directory / 'image_sha256.npy'
            if next_frame:
                features = np.load(feature_path, mmap_mode='r+')
                image_hashes = np.load(hashes_path, mmap_mode='r+')
                if features.shape != shape or image_hashes.shape != (len(paths),):
                    raise ValueError('Cache dimensions do not match frame inventory')
            else:
                features = np.lib.format.open_memmap(feature_path, mode='w+', dtype='float32', shape=shape)
                image_hashes = np.lib.format.open_memmap(hashes_path, mode='w+', dtype='S64', shape=(len(paths),))
            loader = DataLoader(ImageFiles(paths[next_frame:]), batch_size=encoder_cfg['batch_size'],
                                num_workers=encoder_cfg['workers'], collate_fn=list,
                                persistent_workers=encoder_cfg['workers'] > 0)
            cursor = next_frame
            for batch_index, batch in enumerate(loader):
                images, hashes = zip(*batch)
                inputs = processor(images=list(images), return_tensors='pt').to('cuda')
                inputs = {k: v.to(torch.bfloat16) if v.is_floating_point() else v for k, v in inputs.items()}
                with torch.inference_mode():
                    encoded = encoder.get_image_features(**inputs).float().cpu().numpy()
                if encoded.shape != (len(batch), shape[1]) or not np.isfinite(encoded).all():
                    raise ValueError('Invalid frozen features')
                end = cursor + len(batch)
                features[cursor:end] = encoded
                image_hashes[cursor:end] = hashes
                cursor = end
                if (batch_index + 1) % 128 == 0 or cursor == len(paths):
                    features.flush()
                    image_hashes.flush()
                    atomic_json(progress_path, {'next_frame': cursor, 'total_frames': len(paths)})
                    print(f'FEATURES {split}: {cursor}/{len(paths)} frames; elapsed={time.monotonic()-started:.1f}s', flush=True)
            if cursor != len(paths):
                raise ValueError('Incomplete feature extraction')
            if not np.isfinite(features).all():
                raise ValueError('Nonfinite cached features')
            complete[split] = {'clips': len(videos), 'frames': len(paths), 'feature_dim': shape[1],
                               'features_sha256': digest_file(feature_path),
                               'image_hashes_sha256': digest_file(hashes_path),
                               'index_sha256': digest_file(index_path)}
            atomic_json(directory / 'complete.json', complete[split])
        manifest = {'status': 'completed', 'spec_sha256': fingerprint(spec), 'spec': spec,
                    'splits': complete, 'processor_sha256': digest_file(root / 'processor.json')}
        atomic_json(root / 'manifest.json', manifest)
        atomic_json(out / 'feature_cache.json', {'path': str(root), **manifest})
        return {'cache_path': str(root), 'splits': complete, 'benchmark_result': False}


class CachedVideos(Dataset):
    def __init__(self, cfg, split, rows):
        root = cache_directory(cfg)
        manifest = json.loads((root / 'manifest.json').read_text())
        if manifest.get('status') != 'completed' or manifest['spec_sha256'] != fingerprint(cache_spec(cfg)):
            raise ValueError('Feature cache is incomplete or incompatible')
        directory = root / split
        for filename, field in [('features.npy', 'features_sha256'), ('image_sha256.npy', 'image_hashes_sha256'), ('index.json', 'index_sha256')]:
            if digest_file(directory / filename) != manifest['splits'][split][field]:
                raise ValueError(f'{split}: corrupted cache file {filename}')
        index = json.loads((directory / 'index.json').read_text())
        if index['split_sha256'] != cfg['data']['split_sha256'][split]:
            raise ValueError('Cached metadata split mismatch')
        self.ids = [video['sample_id'] for video in index['videos']]
        if self.ids != sorted(rows):
            raise ValueError('Cache sample IDs differ from official split')
        self.offsets = index['offsets']
        self.features = np.load(directory / 'features.npy', mmap_mode='r')
        if self.offsets[0] != 0 or self.offsets[-1] != len(self.features) or any(a >= b for a, b in zip(self.offsets, self.offsets[1:])):
            raise ValueError('Invalid feature offsets')
        self.targets = np.stack([target_distribution(rows[vid]) for vid in self.ids])
        self.counts = np.diff(self.offsets)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        features = np.array(self.features[self.offsets[i]:self.offsets[i + 1]], copy=True)
        return torch.from_numpy(features), torch.from_numpy(self.targets[i]), i


def collate_videos(batch):
    frames, targets, indices = zip(*batch)
    x = torch.nn.utils.rnn.pad_sequence(frames, batch_first=True)
    lengths = torch.tensor([len(frame) for frame in frames])
    mask = torch.arange(x.shape[1])[None, :] < lengths[:, None]
    return x, mask, torch.stack(targets), torch.tensor(indices)
