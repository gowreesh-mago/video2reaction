"""Frozen, split-verified description features and controlled multimodal inputs."""
import fcntl
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from .registry import atomic_json
from .runtime import digest_file, fingerprint


def description_spec(cfg):
    return {'schema': 1, 'encoder': cfg['description_encoder'],
            'split_sha256': cfg['data']['split_sha256'],
            'field': 'clip_description', 'lowercase': True,
            'pooling': 'content-token-weighted mean of nonoverlapping chunks; no L2 normalization',
            'implementation_sha256': digest_file(Path(__file__)),
            'uv_lock_sha256': digest_file(Path(__file__).resolve().parents[2] / 'uv.lock')}


def description_directory(cfg):
    return Path(os.environ['V2R_DESCRIPTION_FEATURE_DIR']) / fingerprint(description_spec(cfg))[:16]


def description_inventory(rows, split_sha):
    records = []
    for vid in sorted(rows):
        text = rows[vid].get('clip_description')
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f'{vid}: missing or empty clip_description')
        records.append({'sample_id': vid, 'description_sha256': hashlib.sha256(text.encode()).hexdigest()})
    return {'split_sha256': split_sha, 'videos': records}


def load_text_encoder(options, device):
    from transformers import AutoModel, AutoTokenizer
    kwargs = {'revision': options['revision'], 'local_files_only': options['local_files_only']}
    tokenizer = AutoTokenizer.from_pretrained(options['model'], use_fast=True, **kwargs)
    if (not tokenizer.is_fast or tokenizer.add_bos_token or not tokenizer.add_eos_token
            or tokenizer.padding_side != 'right' or tokenizer.model_input_names != ['input_ids']):
        raise ValueError('Pinned SigLIP2 tokenizer conventions changed')
    dtype = torch.bfloat16 if device == 'cuda' else torch.float32
    model = AutoModel.from_pretrained(options['model'], torch_dtype=dtype, **kwargs).to(device).eval()
    model.requires_grad_(False)
    if model.config.text_config.max_position_embeddings != options['max_length']:
        raise ValueError('Text context length differs from the pinned encoder')
    return tokenizer, model


def encode_descriptions(texts, tokenizer, model, options, device):
    """Retain every content token; never silently truncate a long description."""
    capacity = options['max_length'] - tokenizer.num_special_tokens_to_add(pair=False)
    if capacity <= 0:
        raise ValueError('No room for text tokens')
    chunks, owners, weights, records = [], [], [], []
    for row, text in enumerate(texts):
        tokens = tokenizer.encode(text.lower(), add_special_tokens=False)
        if not tokens:
            raise ValueError('Description produced no content tokens')
        start = len(chunks)
        for offset in range(0, len(tokens), capacity):
            content = tokens[offset:offset + capacity]
            chunk = tokenizer.build_inputs_with_special_tokens(content)
            if len(chunk) > options['max_length']:
                raise ValueError('Tokenizer special-token count is inconsistent')
            chunks.append({'input_ids': chunk})
            owners.append(row)
            weights.append(len(content))
        records.append({'content_tokens': len(tokens), 'chunks': len(chunks) - start})
    features = np.zeros((len(texts), options['feature_dim']), dtype=np.float64)
    for start in range(0, len(chunks), options['chunk_batch_size']):
        end = min(start + options['chunk_batch_size'], len(chunks))
        # The pinned model was trained on 64 tokens including right padding;
        # its tokenizer supplies only input_ids, without a padding attention mask.
        inputs = tokenizer.pad(chunks[start:end], padding='max_length',
            max_length=options['max_length'], return_attention_mask=False, return_tensors='pt')
        with torch.inference_mode():
            values = model.get_text_features(input_ids=inputs['input_ids'].to(device)).float().cpu().numpy()
        if values.shape != (end - start, options['feature_dim']) or not np.isfinite(values).all():
            raise ValueError('Invalid frozen text-encoder output')
        np.add.at(features, np.asarray(owners[start:end]), values.astype(np.float64) * np.asarray(weights[start:end])[:, None])
    features /= np.asarray([record['content_tokens'] for record in records])[:, None]
    return features.astype(np.float32), records


def _prefix_sha(values, cursor):
    return hashlib.sha256(memoryview(np.ascontiguousarray(values[:cursor]))).hexdigest()


def _verify_split(directory, done, inventory, options):
    for name in ('index.json', 'features.npy', 'tokens.json'):
        if digest_file(directory / name) != done['sha256'][name]:
            raise ValueError(f'Corrupted description cache: {name}')
    if json.loads((directory / 'index.json').read_text()) != inventory:
        raise ValueError('Description cache IDs, text, or split changed')
    features = np.load(directory / 'features.npy', mmap_mode='r', allow_pickle=False)
    records = json.loads((directory / 'tokens.json').read_text())
    if (features.shape != (len(inventory['videos']), options['feature_dim'])
            or features.dtype != np.float32 or not np.isfinite(features).all()
            or len(records) != len(features) or any(r['content_tokens'] < 1 or r['chunks'] < 1 for r in records)):
        raise ValueError('Invalid description feature/token inventory')
    return features


def load_description_features(cfg, split, rows):
    root = description_directory(cfg)
    manifest = json.loads((root / 'manifest.json').read_text())
    if manifest['status'] != 'completed' or manifest['spec_sha256'] != fingerprint(description_spec(cfg)):
        raise ValueError('Description cache is incomplete or incompatible')
    inventory = description_inventory(rows, cfg['data']['split_sha256'][split])
    return _verify_split(root / split, manifest['splits'][split], inventory, cfg['description_encoder'])


def prepare_description_cache(cfg, splits, out, device='cuda'):
    """CPU is only for synthetic fixtures; actual data runs use the guarded SLURM CLI."""
    if device == 'cuda' and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()):
        raise RuntimeError('Text extraction requires a bf16-capable allocated GPU')
    root, spec, options = description_directory(cfg), description_spec(cfg), cfg['description_encoder']
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'cache.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        spec_path = root / 'spec.json'
        if spec_path.exists() and json.loads(spec_path.read_text()) != spec:
            raise ValueError('Description cache specification mismatch')
        atomic_json(spec_path, spec)
        manifest_path = root / 'manifest.json'
        existing = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
        if existing and (existing['status'] != 'completed' or existing['spec_sha256'] != fingerprint(spec)
                         or set(existing['splits']) != set(splits)):
            raise ValueError('Cannot replace a completed description cache with another inventory')
        complete, tokenizer, model = {}, None, None
        for split, rows in splits.items():
            inventory = description_inventory(rows, cfg['data']['split_sha256'][split])
            directory = root / split
            directory.mkdir(exist_ok=True)
            completion = directory / 'complete.json'
            if existing or completion.exists():
                done = existing['splits'][split] if existing else json.loads(completion.read_text())
                _verify_split(directory, done, inventory, options)
                complete[split] = done
                continue
            index_path = directory / 'index.json'
            if index_path.exists() and json.loads(index_path.read_text()) != inventory:
                raise ValueError('Description inventory changed during extraction')
            atomic_json(index_path, inventory)
            path, progress_path = directory / 'features.npy', directory / 'progress.json'
            progress = json.loads(progress_path.read_text()) if progress_path.exists() else {'next_video': 0, 'tokens': []}
            cursor, records = progress['next_video'], progress['tokens']
            shape = (len(rows), options['feature_dim'])
            if not 0 <= cursor <= len(rows) or len(records) != cursor:
                raise ValueError('Invalid description-cache progress')
            if cursor:
                features = np.load(path, mmap_mode='r+')
                if (features.shape != shape or features.dtype != np.float32
                        or _prefix_sha(features, cursor) != progress['prefix_sha256']):
                    raise ValueError('Committed description-cache prefix is corrupted')
            else:
                features = np.lib.format.open_memmap(path, mode='w+', dtype='float32', shape=shape)
            if cursor < len(rows) and model is None:
                tokenizer, model = load_text_encoder(options, device)
            ids = sorted(rows)
            while cursor < len(rows):
                end = min(cursor + options['video_batch_size'], len(rows))
                values, token_records = encode_descriptions([rows[vid]['clip_description'] for vid in ids[cursor:end]],
                                                           tokenizer, model, options, device)
                features[cursor:end] = values
                records.extend(token_records)
                cursor = end
                features.flush()
                with path.open('rb') as stream:
                    os.fsync(stream.fileno())
                atomic_json(progress_path, {'next_video': cursor, 'tokens': records,
                                            'prefix_sha256': _prefix_sha(features, cursor)})
                print(f'DESCRIPTION {split}: {cursor}/{len(rows)}', flush=True)
            atomic_json(directory / 'tokens.json', records)
            complete[split] = {'clips': len(rows), 'chunks': sum(r['chunks'] for r in records),
                'multi_chunk_clips': sum(r['chunks'] > 1 for r in records),
                'max_content_tokens': max(r['content_tokens'] for r in records),
                'sha256': {name: digest_file(directory / name) for name in ('index.json', 'features.npy', 'tokens.json')}}
            _verify_split(directory, complete[split], inventory, options)
            atomic_json(completion, complete[split])
        manifest = {'status': 'completed', 'spec': spec, 'spec_sha256': fingerprint(spec), 'splits': complete}
        atomic_json(manifest_path, manifest)
        atomic_json(Path(out) / 'description_cache.json', {'path': str(root), **manifest})
        return {'benchmark_result': False, 'cache_path': str(root), 'splits': complete}


class DescriptionVideos:
    def __init__(self, cfg, visual, rows):
        self.visual, self.mode = visual, cfg['description']['mode']
        if self.mode not in {'text', 'visual_text', 'visual_visual', 'text_text'}:
            raise ValueError('Unknown description input comparison')
        self.ids, self.targets = visual.ids, visual.targets
        if self.ids != sorted(rows):
            raise ValueError('Description and visual sample order differ')
        self.text = load_description_features(cfg, visual.split, rows)
        self.stratum_counts = visual.counts
        self.counts = np.ones(len(visual), dtype=np.int64) if self.mode in {'text', 'text_text'} else visual.counts

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        target = torch.from_numpy(self.targets[i])
        text = torch.from_numpy(np.array(self.text[i:i+1], copy=True))
        if self.mode == 'text':
            return text, target, i
        if self.mode == 'text_text':
            return torch.cat([text, text], dim=-1), target, i
        visual, _, _ = self.visual[i]
        second = text.expand(len(visual), -1) if self.mode == 'visual_text' else visual
        return torch.cat([visual, second], dim=-1), target, i
