"""Verify recommended runs and predeclared controls in the locked cluster environment.

Read saved outputs and private VAD assets only; no model fitting or test-based
selection. Pass one or more submission JSON files to include completed batches.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.analyze_tier0 import RUNS, analyze
from scripts.analyze_attention import ATTENTION_RUNS, PAIRS, attention_summary
from src.experiments.emotion import model_descriptor, probabilities, read_coarse_vad
from src.experiments.evidence import peak_scores, select_frames
from src.experiments.runtime import digest_file, load_config
from src.experiments.vad import reaction_vad


def comparisons(runs):
    pairs = list(PAIRS)
    for name in runs:
        if name not in RUNS and name not in ATTENTION_RUNS:
            pairs.append((name, 'b1_meanpool'))
    pairs += [('b_vad_aux', 'b_vad_aux_permuted'),
              ('b_vad_geometry', 'b_vad_geometry_permuted'),
              ('f_global_peak', 'f_global_control'), ('f_global_peak', 'f_peak_control'),
              ('visual_description', 'description_only'),
              ('visual_description', 'description_visual_control'),
              ('visual_description', 'description_text_control')]
    for k in (1, 2, 4, 8):
        for method in ('arousal', 'distance', 'confidence'):
            pairs.extend((f'd_{method}_k{k}', f'd_{control}_k{k}') for control in ('uniform', 'random'))
    return [(a, b) for a, b in dict.fromkeys(pairs) if a in runs and b in runs]


def verify_evidence(directory, cfg, reference_frames):
    selected_path = directory / 'selected_frames.json'
    record = json.loads(selected_path.read_text())
    assert record['selection_uses_targets'] is False
    assert record['evidence'] == cfg['evidence'] and record['fusion'] == cfg['model'].get('fusion')
    assert record['source_frame_inventory'] == reference_frames
    with np.load(directory / 'predictions.npz', allow_pickle=False) as data:
        ids, counts = data['sample_id'], data['frame_count']
    with np.load(directory / 'frame_evidence.npz', allow_pickle=False) as data:
        evidence = {key: data[key] for key in data.files}
    np.testing.assert_array_equal(evidence['sample_id'], ids)
    np.testing.assert_array_equal(evidence['offsets'], np.r_[0, counts.cumsum()])
    np.testing.assert_array_equal(evidence['coarse_class_order'], model_descriptor(cfg)['class_order'])
    q = probabilities(evidence['emotion_logits'])
    np.testing.assert_allclose(evidence['emotion_probability'], q, rtol=0, atol=1e-12)
    vad, scores = peak_scores(q, read_coarse_vad(cfg), cfg['evidence']['neutral'])
    np.testing.assert_allclose(evidence['expected_vad'], vad, rtol=0, atol=1e-12)
    for saved, method in [('arousal', 'arousal'), ('distance', 'distance'), ('confidence_strength', 'confidence')]:
        np.testing.assert_allclose(evidence[saved], scores[method], rtol=0, atol=1e-12)
    assert len(record['videos']) == len(ids)
    for i, (start, end) in enumerate(zip(evidence['offsets'], evidence['offsets'][1:])):
        item = record['videos'][i]
        assert item['sample_id'] == ids[i] and item['total_frames'] == counts[i]
        assert reference_frames[i]['sample_id'] == ids[i]
        chosen = select_frames(end-start, cfg['evidence']['selector'], cfg['evidence']['k'],
                               {key: values[start:end] for key, values in scores.items()}, ids[i], cfg['seed'])
        np.testing.assert_array_equal(item['selected_positions'], chosen)
        assert item['selected_frames'] == [reference_frames[i]['frames'][j] for j in chosen]
        peak = np.zeros(end-start, dtype=np.float32)
        peak[chosen] = 1 / len(chosen)
        np.testing.assert_array_equal(evidence['peak_weights'][start:end], peak)
        np.testing.assert_array_equal(evidence['global_weights'][start:end],
                                      np.full(end-start, 1/(end-start), dtype=np.float32))
    return {'clips': len(ids), 'frames': int(counts.sum()),
            'verified': 'All scores, selected positions, original frame records, and pooling weights recompute.',
            'evidence_sha256': digest_file(directory / 'frame_evidence.npz'),
            'selected_frames_sha256': digest_file(selected_path)}


def main(args):
    runs = {**RUNS, **ATTENTION_RUNS}
    submission_hashes = {}
    for path in args.submissions:
        submission_hashes[str(path)] = digest_file(path)
        for name, job in json.loads(path.read_text()).items():
            if name.endswith('_cache'):
                continue
            if name in runs:
                raise ValueError(f'Duplicate experiment in supplied batches: {name}')
            runs[name] = f'{name}_{job}'
    result = analyze(args.outputs, args.resamples, 20260930, runs, comparisons(runs))
    result['submission_sha256'] = submission_hashes
    result['analysis_script_sha256'] = digest_file(Path(__file__))
    result['attention_diagnostics'] = {name: {split: attention_summary(args.outputs / run / split,
        shared=name == 'e_shared_query_control') for split in ('val', 'test')}
        for name, run in ATTENTION_RUNS.items()}
    frames = {split: json.loads((args.outputs / ATTENTION_RUNS['e_reaction_query'] / split /
              'attention_frames.json').read_text())['videos'] for split in ('val', 'test')}
    result['frame_diagnostics'] = {}
    sourced_cfg = load_config(args.outputs / runs['b_vad_geometry'] / 'config.yaml')
    matrix, provenance = reaction_vad(sourced_cfg['vad_objective'], 'cpu')
    matrix = matrix.numpy().astype(np.float64)
    result['expected_vad_diagnostic'] = {'definition': 'Mean squared Euclidean distance between target@E and prediction@E, using the same sourced 21-label E for every model. Sum over three axes, then mean over clips. This uses the predicted reaction distribution, not the auxiliary head.',
        'asset_sha256': provenance['asset_sha256'], 'limits': 'Post-hoc diagnostic; not used for model selection. Word ratings are not human ratings of these clips.', 'runs': {}}
    for name, run in runs.items():
        directory = args.outputs / run
        cfg = load_config(directory / 'config.yaml')
        if 'vad_objective' in cfg:
            _, expected = reaction_vad(cfg['vad_objective'], 'cpu')
            assert json.loads((directory / 'vad_provenance.json').read_text()) == expected
        if 'evidence' in cfg:
            result['frame_diagnostics'][name] = {split: verify_evidence(directory / split, cfg, frames[split])
                                                for split in ('val', 'test')}
        values = {}
        for split in ('val', 'test'):
            with np.load(directory / split / 'predictions.npz', allow_pickle=False) as data:
                delta = (data['predicted_distribution'].astype(np.float64) - data['target_distribution']) @ matrix
                values[split] = float(np.square(delta).sum(1).mean())
        result['expected_vad_diagnostic']['runs'][name] = values
    result['checks'] += ['VAD coordinate assignments match each saved run provenance.',
        'All saved peak scores and frame selections match the declared selector and original frame inventory.']
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'runs_verified': len(runs), 'frame_runs_verified': len(result['frame_diagnostics']),
                      'paired_kl': result['paired_kl'], 'output': str(args.output)}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--submissions', type=Path, nargs='+', required=True)
    parser.add_argument('--outputs', type=Path, default=Path('outputs/experiments'))
    parser.add_argument('--output', type=Path, default=Path('research_notes/recommended_results.json'))
    parser.add_argument('--resamples', type=int, default=10000)
    args = parser.parse_args()
    if args.resamples < 2:
        parser.error('--resamples must be at least 2')
    main(args)
