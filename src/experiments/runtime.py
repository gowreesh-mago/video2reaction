"""Configuration, provenance and deterministic state for cluster experiments."""
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile

import numpy as np
import torch
import yaml

from .data import load_metadata, target_distribution
from .taxonomy import REACTION_CLASSES


def digest_file(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def load_config(path):
    path = Path(path)
    cfg = yaml.safe_load(path.read_text())
    base = cfg.pop('base', None)
    if base:
        result = load_config(path.parent / base)
        def merge(left, right):
            for key, value in right.items():
                if isinstance(value, dict) and isinstance(left.get(key), dict):
                    merge(left[key], value)
                else:
                    left[key] = copy.deepcopy(value)
        merge(result, cfg)
        return result
    return cfg


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda']])


def atomic_torch(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    os.close(fd)
    try:
        torch.save(value, name)
        with open(name, 'rb') as stream:
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def official_splits(cfg):
    splits, hashes = {}, {}
    for split in ('train', 'val', 'test'):
        rows, sha = load_metadata(os.environ['V2R_METADATA_DIR'], split)
        if sha != cfg['data']['split_sha256'][split]:
            raise ValueError(f'{split}: official metadata hash mismatch')
        for row in rows.values():
            target_distribution(row)
        splits[split], hashes[split] = rows, sha
    names = list(splits)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if set(splits[a]) & set(splits[b]):
                raise ValueError(f'Video overlap between {a} and {b}')
    return splits, hashes


def cache_spec(cfg):
    return {'schema': 1, 'encoder': cfg['encoder'], 'data': cfg['data'],
            'features': 'SigLIP2 get_image_features, eval, bf16 -> float32',
            'class_order': REACTION_CLASSES}


def cache_directory(cfg):
    return Path(os.environ['V2R_FEATURE_DIR']) / fingerprint(cache_spec(cfg))[:16]


def verify_code(root):
    root = Path(root)
    manifest = json.loads((root / 'code_version.json').read_text())
    if manifest['dirty']:
        raise ValueError('Experiments require a committed source snapshot')
    for name, expected in manifest['files'].items():
        if digest_file(root / name) != expected:
            raise ValueError(f'Source snapshot mismatch: {name}')
    return manifest
