"""Verify the E pair against Tier 0 and inspect saved reaction-attention maps."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.analyze_tier0 import RUNS, analyze
from src.experiments.runtime import digest_file
from src.experiments.taxonomy import REACTION_CLASSES


ATTENTION_RUNS = {'e_reaction_query': 'e_reaction_query_27438329',
                  'e_shared_query_control': 'e_shared_query_control_27438330'}
PAIRS = [('e_reaction_query', 'b1_meanpool'), ('e_shared_query_control', 'b1_meanpool'),
         ('e_reaction_query', 'e_shared_query_control')]


def attention_summary(directory, shared=False):
    frame_path, path = directory / 'attention_frames.json', directory / 'attention.npz'
    frames = json.loads(frame_path.read_text())
    if frames['attention_sha256'] != digest_file(path) or frames['class_order'] != REACTION_CLASSES:
        raise ValueError('Attention checksum or taxonomy mismatch')
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    with np.load(directory / 'predictions.npz', allow_pickle=False) as archive:
        ids, counts = archive['sample_id'], archive['frame_count']
    np.testing.assert_array_equal(data['sample_id'], ids)
    np.testing.assert_array_equal(data['class_order'], REACTION_CLASSES)
    np.testing.assert_array_equal(data['offsets'], np.r_[0, counts.cumsum()])
    if data['weights'].shape != (counts.sum(), 21) or not np.isfinite(data['weights']).all():
        raise ValueError('Invalid attention dimensions or values')
    if (data['weights'] < 0).any() or len(frames['videos']) != len(ids):
        raise ValueError('Invalid attention or frame inventory')
    normalized_entropies, effective_fractions, js_values, peak_agreements = [], [], [], []
    for i, (start, end) in enumerate(zip(data['offsets'], data['offsets'][1:])):
        record = frames['videos'][i]
        if record['sample_id'] != ids[i] or len(record['frames']) != counts[i]:
            raise ValueError('Attention frame order differs from prediction inventory')
        weights = data['weights'][start:end].astype(np.float64)
        np.testing.assert_allclose(weights.sum(0), 1, atol=1e-6, rtol=0)
        logs = np.log(np.maximum(weights, 1e-30))
        entropy = -(weights * logs).sum(0)
        js = (weights * (logs - np.log(np.maximum(weights.mean(1, keepdims=True), 1e-30)))).sum(0).mean()
        peaks = weights.argmax(0)
        np.testing.assert_allclose(entropy, data['entropy'][i], atol=1e-12, rtol=0)
        np.testing.assert_allclose(js, data['class_map_js'][i], atol=1e-12, rtol=0)
        np.testing.assert_array_equal(peaks, data['top_frame_position'][i])
        if shared:
            np.testing.assert_array_equal(weights, np.repeat(weights[:, :1], 21, axis=1))
        normalized_entropies.extend(entropy / np.log(end-start) if end-start > 1 else np.zeros(21))
        effective_fractions.extend(np.exp(entropy) / (end-start))
        js_values.append(js)
        peak_agreements.append(len(set(peaks.tolist())) == 1)
    return {'clips': len(ids), 'frames': int(counts.sum()),
            'mean_normalized_attention_entropy': float(np.mean(normalized_entropies)),
            'mean_effective_frame_fraction': float(np.mean(effective_fractions)),
            'mean_class_map_js': float(np.mean(js_values)),
            'clips_with_identical_top_frame_for_all_classes_fraction': float(np.mean(peak_agreements)),
            'attention_sha256': digest_file(path), 'frame_inventory_sha256': digest_file(frame_path),
            'interpretation': 'Attention is a model diagnostic; it does not identify causal audience-reaction evidence.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outputs', type=Path, default=Path('outputs/experiments'))
    parser.add_argument('--output', type=Path, default=Path('research_notes/attention_results.json'))
    parser.add_argument('--resamples', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=20260930)
    args = parser.parse_args()
    if args.resamples < 2:
        parser.error('--resamples must be at least 2')
    result = analyze(args.outputs, args.resamples, args.seed, {**RUNS, **ATTENTION_RUNS}, PAIRS)
    result['attention_diagnostics'] = {name: {split: attention_summary(args.outputs / run / split,
        shared=name == 'e_shared_query_control') for split in ('val', 'test')}
        for name, run in ATTENTION_RUNS.items()}
    result['checks'].append('Both E runs have checksum-verified, normalized attention maps aligned with every validation/test clip and its original frames; shared maps are exactly equal across reactions.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'paired_kl': {f'{a} minus {b}': result['paired_kl'][f'{a} minus {b}'] for a, b in PAIRS},
                      'attention_diagnostics': result['attention_diagnostics']}, indent=2))
