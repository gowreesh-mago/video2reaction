"""Controlled frozen-feature training with validation-only checkpoint selection."""
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .features import collate_videos
from .losses import distribution_loss
from .metrics import evaluate
from .models import ReactionPredictor
from .registry import atomic_json
from .runtime import atomic_torch, fingerprint, restore_rng, rng_state, seed_everything


def loader(dataset, cfg, epoch=None):
    generator = torch.Generator().manual_seed(cfg['seed'] + (epoch or 0))
    return DataLoader(dataset, batch_size=cfg['training']['batch_size'], shuffle=epoch is not None,
                      generator=generator, collate_fn=collate_videos, num_workers=0)


def predict(model, dataset, cfg, device, shuffle_frames=False):
    model.eval()
    prediction = np.empty((len(dataset), 21), dtype=np.float64)
    with torch.no_grad():
        for x, mask, _, indices in loader(dataset, cfg):
            if shuffle_frames:
                for row, index in enumerate(indices.tolist()):
                    count = int(mask[row].sum())
                    rng = np.random.default_rng(cfg['evaluation']['permutation_seed'] + index)
                    x[row, :count] = x[row, rng.permutation(count)].clone()
            logits, _ = model(x.to(device), mask.to(device))
            prediction[indices.numpy()] = logits.softmax(-1).cpu().numpy()
    return prediction


def fit(cfg, train, val, out, device, source_sha, resume_from=None):
    """CPU is supported for synthetic tests; the real CLI requires CUDA."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    seed_everything(cfg['seed'])
    model = ReactionPredictor(**cfg['model']).to(device)
    options = cfg['training']
    if options['selection_metric'] != 'kl' or options['scheduler'] != 'constant':
        raise ValueError('This runner supports validation KL and a constant learning rate')
    optimizer = torch.optim.AdamW(model.parameters(), lr=options['learning_rate'], weight_decay=options['weight_decay'])
    best, best_epoch, stale, start, best_state, history = float('inf'), 0, 0, 0, None, []
    identity = {'config_sha256': fingerprint(cfg), 'source_sha256': source_sha}
    if resume_from:
        # Only load checkpoints produced by this project in the user's private output directory.
        state = torch.load(resume_from, map_location=device, weights_only=False)
        if state['identity'] != identity:
            raise ValueError('Resume requires the same resolved config and source snapshot')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        start, best, best_epoch, stale = state['epoch'], state['best_kl'], state['best_epoch'], state['stale']
        best_state, history = state['best_model'], state['history']
        restore_rng(state['rng'])
    if stale < options['patience']:
        for epoch in range(start, options['epochs']):
            model.train()
            total = 0.
            for x, mask, target, _ in loader(train, cfg, epoch=epoch):
                optimizer.zero_grad(set_to_none=True)
                logits, _ = model(x.to(device), mask.to(device))
                loss = distribution_loss(logits, target.to(device), **cfg['loss'])
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite training objective')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), options['clip_grad_norm'], error_if_nonfinite=True)
                optimizer.step()
                total += loss.item() * len(x)
            validation, _ = evaluate(val.targets, predict(model, val, cfg, device))
            improved = validation['kl'] < best - options['min_delta']
            if improved:
                best, best_epoch, stale = validation['kl'], epoch + 1, 0
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            else:
                stale += 1
            history.append({'epoch': epoch + 1, 'train_objective': total / len(train), 'val': validation})
            atomic_torch(out / 'last.pt', {'identity': identity, 'model': model.state_dict(),
                         'optimizer': optimizer.state_dict(), 'epoch': epoch + 1, 'best_kl': best,
                         'best_epoch': best_epoch, 'best_model': best_state, 'stale': stale,
                         'history': history, 'rng': rng_state()})
            atomic_json(out / 'history.json', history)
            print(json.dumps({'epoch': epoch + 1, 'train_objective': total / len(train),
                              'val_kl': validation['kl'], 'best_epoch': best_epoch, 'patience_used': stale}), flush=True)
            if stale >= options['patience']:
                break
    if best_state is None:
        raise ValueError('Training produced no selected checkpoint')
    model.load_state_dict(best_state)
    atomic_torch(out / 'best.pt', {'model': best_state, 'config': copy.deepcopy(cfg), 'identity': identity,
                                 'best_epoch': best_epoch, 'val_kl': best})
    # Verify serialized selected weights before evaluation on the test split.
    reloaded = ReactionPredictor(**cfg['model']).to(device)
    reloaded.load_state_dict(torch.load(out / 'best.pt', map_location=device, weights_only=True)['model'])
    model.eval()
    reloaded.eval()
    x, mask, _, _ = next(iter(loader(val, cfg)))
    with torch.no_grad():
        torch.testing.assert_close(model(x.to(device), mask.to(device))[0],
                                   reloaded(x.to(device), mask.to(device))[0], rtol=0, atol=0)
    return model, {'best_epoch': best_epoch, 'val_selection_kl': best, 'epochs_run': len(history),
                   'trainable_parameters': sum(p.numel() for p in model.parameters()), 'checkpoint_reload_exact': True}
