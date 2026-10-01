"""Build tables, scientific figures, and 20 examples from saved run outputs.

This is report preparation, not model training or checkpoint selection. Official
scores and class ranks are read from cluster outputs; no local argsort replaces
the locked NumPy evaluation. All source data remains in the existing run files.
"""
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/v2r-report-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'research_notes/latex'
TABLES = DEST / 'tables'
FIGURES = DEST / 'figures'
ALIASES = {'b0_prior': 'B0', 'b1_meanpool': 'B1', 'b2_temporal': 'B2',
           'b2_set_control': 'Set', 'a5_distribution': 'A5'}
COLORS = ['#82919e', '#087f8c', '#b65336', '#785a9b', '#ba8d27']


def tex(value):
    replacements = {'\\': r'\textbackslash{}', '&': r'\&', '%': r'\%', '$': r'\$',
                    '#': r'\#', '_': r'\_', '{': r'\{', '}': r'\}',
                    '~': r'\textasciitilde{}', '^': r'\textasciicircum{}'}
    return ''.join(replacements.get(c, c) for c in str(value))


def table(filename, headers, rows, caption, label, align=None, size=r'\small'):
    align = align or 'l' + 'r' * (len(headers) - 1)
    text = [r'\begin{table}[!htbp]', r'\centering', size,
            r'\setlength{\tabcolsep}{5pt}', r'\begin{tabular}{' + align + '}',
            r'\toprule', ' & '.join(headers) + r' \\', r'\midrule']
    text.extend(' & '.join(str(v) for v in row) + r' \\' for row in rows)
    text.extend([r'\bottomrule', r'\end{tabular}', r'\caption{' + caption + '}',
                 r'\label{' + label + '}', r'\end{table}', ''])
    (TABLES / filename).write_text('\n'.join(text))


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    report_path = ROOT / 'research_notes/tier0_results.json'
    report = json.loads(report_path.read_text())
    runs = report['runs']
    audit = json.loads((ROOT / 'research_notes/dataset_audit.json').read_text())
    classes = audit['class_order']
    per_class = {}
    predictions = {}
    for name, run in runs.items():
        directory = ROOT / 'outputs/experiments' / run['run_id']
        per_class[name] = json.loads((directory / 'test/per_class.json').read_text())
        with np.load(directory / 'test/predictions.npz', allow_pickle=False) as data:
            predictions[name] = {k: data[k] for k in data.files}
        for k in ('sample_id', 'movie_id', 'target_distribution', 'class_order', 'frame_count'):
            np.testing.assert_array_equal(predictions[name][k], predictions['b0_prior'][k])
        for topk in ('top1', 'top2', 'top3'):
            s = per_class[name]['scores'][topk]
            weighted = np.dot(s['f1'], s['target_support']) / sum(s['target_support'])
            np.testing.assert_allclose(weighted, run['test']['f1_' + topk], rtol=0, atol=1e-12)

    primary = ['kl', 'cosine', 'mrr', 'f1_top1', 'f1_top3']
    table('primary.tex', ['Model', r'KL $\downarrow$', r'Cosine $\uparrow$', r'MRR $\uparrow$',
          r'F1@1 $\uparrow$', r'F1@3 $\uparrow$'],
          [[ALIASES[n]] + [f'{r["test"][m]:.6f}' for m in primary] for n, r in runs.items()],
          'Test results on all 2,070 official test clips. Scores are proportions, not percentages. '
          'Lower KL is better; larger cosine, MRR, and F1 are better. Set denotes B2 without positions.', 'tab:primary')
    secondary = ['chebyshev', 'clark', 'cad', 'intersection', 'tpe', 'f1_top2']
    table('secondary.tex', ['Model', r'Cheb $\downarrow$', r'Clark $\downarrow$', r'CAD $\downarrow$',
          r'Inter $\uparrow$', r'TPE $\downarrow$', r'F1@2 $\uparrow$'],
          [[ALIASES[n]] + [f'{r["test"][m]:.6f}' for m in secondary] for n, r in runs.items()],
          'Remaining official test metrics. CAD follows the fixed class order; it is not a '
          'three-dimensional VAD distance.', 'tab:secondary', size=r'\footnotesize')
    table('selection.tex', ['Model', 'Parameters', 'Epoch selected', 'Epochs run', r'Val KL $\downarrow$'],
          [[ALIASES[n], f'{r["training"]["trainable_parameters"]:,}',
            r['training'].get('best_epoch', '--'), r['training'].get('epochs_run', '--'),
            f'{r["val"]["kl"]:.6f}'] for n, r in runs.items()],
          'Validation-only checkpoint selection. B0 has no optimized parameters or checkpoint selection.', 'tab:selection')
    labels = ['B1 minus B0', 'B2 minus B1', 'B2 minus Set', 'Set minus B1', 'A5 minus B1', 'B2 shuffled minus ordered']
    table('bootstrap.tex', ['Comparison', r'$\Delta$ KL', '95\% interval'],
          [[label, f'{v["difference"]:+.6f}',
            f'[{v["percentile_95_ci"][0]:+.6f}, {v["percentile_95_ci"][1]:+.6f}]']
           for label, v in zip(labels, report['paired_kl'].values())],
          'Paired movie-cluster bootstrap, 10,000 resamples of 1,183 test movies. Negative differences '
          'favor the first model. These are marginal percentile intervals, conditional on trained weights.',
          'tab:bootstrap', align='lrl')
    frames = {'train': 317950, 'val': 45964, 'test': 91312}
    table('dataset.tex', ['Split', 'Clips', 'Frames', 'Mean / median frames', 'Mean entropy'],
          [[s, f'{a["size"]:,}', f'{frames[s]:,}', f'{a["keyframes"]["mean"]:.2f} / {a["keyframes"]["median"]:g}',
            f'{a["entropy_nats"]["mean"]:.4f}'] for s, a in audit['splits'].items()],
          'Audited official splits. Target entropy is measured in nats. All referenced frame paths '
          'exist and every frame was subsequently decoded during completed feature extraction.', 'tab:dataset', align='lrrcr')

    macro_rows = []
    for n in runs:
        s1, s3 = [per_class[n]['scores'][k] for k in ('top1', 'top3')]
        macro_rows.append([ALIASES[n], f'{np.mean(s1["f1"]):.4f}', sum(x > 0 for x in s1['f1']),
                          sum(x == 0 for x in s1['prediction_support']), f'{np.mean(s3["f1"]):.4f}',
                          sum(x > 0 for x in s3['f1'])])
    table('rare_classes.tex', ['Model', 'Macro F1@1', 'Nonzero F1@1', 'Unpredicted@1', 'Macro F1@3', 'Nonzero F1@3'],
          macro_rows, 'Additional class-balanced diagnostics over all 21 classes. Undefined class scores '
          'are zero; classes absent from the target top-k sets are included. These are not replacements '
          'for official support-weighted F1.', 'tab:rare', size=r'\footnotesize')
    for n in runs:
        s1, s3 = [per_class[n]['scores'][k] for k in ('top1', 'top3')]
        rows = []
        for i, c in enumerate(classes):
            rows.append([tex(c)] + [f'{s[key][i]:.3f}' for s in (s1, s3) for key in ('precision', 'recall', 'f1')]
                        + [s1['target_support'][i], s1['prediction_support'][i], s3['target_support'][i], s3['prediction_support'][i]])
        table(f'classes_{n}.tex', ['Class', '$P_1$', '$R_1$', '$F_1$', '$P_3$', '$R_3$', '$F_3$', '$S_1$', '$N_1$', '$S_3$', '$N_3$'],
              rows, f'{ALIASES[n]} test per-class scores. Subscripts denote top-k. $P$: precision; $R$: recall; '
              '$F$: F1; $S$: target support; $N$: prediction support. The support counts follow locked '
              'NumPy top-k ties and need not match metadata dominant-label counts.', f'tab:classes-{n}', size=r'\scriptsize')
    strata_rows = []
    for variable, title in [('entropy', 'Entropy'), ('frame_count', 'Frames'), ('dominant_probability', 'Dominant mass')]:
        for quartile in ('1', '2', '3', '4'):
            ref = runs['b1_meanpool']['diagnostics']['strata'][variable][quartile]
            strata_rows.append([f'{title} Q{quartile}', ref['n']] +
                               [f'{r["diagnostics"]["strata"][variable][quartile]["metrics"]["kl"]:.4f}' for r in runs.values()])
    table('strata.tex', ['Group', '$n$'] + list(ALIASES.values()), strata_rows,
          'Test KL by training-derived bins. Equal quantile cuts do not guarantee equal test counts; '
          'values equal to a cut enter the upper bin. Bins are fixed before test evaluation.', 'tab:strata')
    class_rows = [[tex(c), f'{audit["splits"]["train"]["per_label_mean_probability"][c]:.6f}',
                   audit['splits']['train']['dominant_label_frequency'].get(c, 0),
                   audit['splits']['test']['dominant_label_frequency'].get(c, 0)] for c in classes]
    table('taxonomy.tex', ['Class (fixed order)', 'Mean train mass', 'Train dominant', 'Test dominant'], class_rows,
          'Class order, imbalance, and metadata dominant-label frequencies from the dataset audit. '
          'These last two columns use metadata labels; official top-k support is reported separately.', 'tab:taxonomy')

    # Select post-hoc error illustrations without changing any model or metric.
    base = predictions['b0_prior']
    p = base['target_distribution'].astype(np.float64)
    entropy = -(p * np.log(np.maximum(p, 1e-12))).sum(1)
    losses = {}
    for name, data in predictions.items():
        q = data['predicted_distribution'].astype(np.float64)
        losses[name] = (p * np.log(p / (q + 1e-10) + 1e-10)).sum(1)
        np.testing.assert_allclose(losses[name].mean(), runs[name]['test']['kl'], rtol=0, atol=1e-10)
    gain = losses['b0_prior'] - losses['b1_meanpool']
    selections = [('improvements', -gain), ('failures', gain), ('high_entropy', -entropy), ('low_entropy', entropy)]
    used, used_movies, examples = set(), set(), []
    for group, order in selections:
        chosen = []
        for i in sorted(range(len(p)), key=lambda j: (float(order[j]), str(base['sample_id'][j]))):
            movie = str(base['movie_id'][i])
            if i in used or movie in used_movies:
                continue
            used.add(i)
            used_movies.add(movie)
            chosen.append(i)
            if len(chosen) == 5:
                break
        assert len(chosen) == 5
        for i in chosen:
            if group == 'improvements':
                assert gain[i] > 0
            if group == 'failures':
                assert gain[i] < 0
            examples.append({'group': group, 'sample_id': str(base['sample_id'][i]),
                             'movie_id': str(base['movie_id'][i]), 'frames': int(base['frame_count'][i]),
                             'entropy_nats': float(entropy[i]), 'b1_kl_improvement_over_b0': float(gain[i]),
                             'target_distribution': p[i].tolist(),
                             'target_top3': [classes[j] for j in base['target_topk'][i]],
                             'models': {n: {'kl': float(losses[n][i]),
                                           'predicted_distribution': d['predicted_distribution'][i].tolist(),
                                           'predicted_top3': [classes[j] for j in d['predicted_topk'][i]]}
                                        for n, d in predictions.items()}})
    metadata_path = ROOT / 'results/report_examples_metadata.json'
    existing_examples = ROOT / 'research_notes/qualitative_examples.json'
    metadata = {}
    if existing_examples.exists():
        metadata = {e['sample_id']: {'movie_name': e['movie_name']}
                    for e in json.loads(existing_examples.read_text())['examples'] if e.get('movie_name')}
    if metadata_path.exists():
        metadata.update(json.loads(metadata_path.read_text()))
    for example in examples:
        example.update(metadata.get(example['sample_id'], {}))
    evidence = {'selection': 'Five largest B1-vs-B0 KL improvements, five largest KL regressions, then five '
                'highest-entropy and five lowest-entropy remaining test clips. Unique sample/movie IDs '
                'across all groups; ties by sample ID. Post-hoc examples, not a representative sample.',
                'class_order': classes, 'examples': examples}
    (ROOT / 'research_notes/qualitative_examples.json').write_text(json.dumps(evidence, indent=2) + '\n')
    (ROOT / 'results/report_example_ids.json').write_text(json.dumps([e['sample_id'] for e in examples]) + '\n')
    for group, _ in selections:
        content = []
        for number, e in enumerate(examples, 1):
            if e['group'] != group:
                continue
            target = dict(zip(classes, e['target_distribution']))
            predicted = dict(zip(classes, e['models']['b1_meanpool']['predicted_distribution']))
            top = lambda labels, values: ', '.join(f'{tex(c)} {values[c]:.3f}' for c in labels)
            content.extend([
                r'\par\noindent\begin{minipage}{\textwidth}\small',
                r'\textbf{' + f'Q{number:02d}' + r'}\quad\href{https://www.youtube.com/watch?v=' + e['sample_id'] + r'}{\texttt{' + tex(e['sample_id']) + r'}}'
                + r'\quad IMDb: \texttt{' + tex(e['movie_id']) + r'}\par',
                (r'\textit{' + tex(e['movie_name']) + r'}\par') if e.get('movie_name') else '',
                f'Frames: {e["frames"]}; target entropy: {e["entropy_nats"]:.3f} nats; '
                f'KL(B0): {e["models"]["b0_prior"]["kl"]:.3f}; KL(B1): {e["models"]["b1_meanpool"]["kl"]:.3f}.\\par',
                'Target top 3: ' + top(e['target_top3'], target) + r'.\par',
                'B1 top 3: ' + top(e['models']['b1_meanpool']['predicted_top3'], predicted) + r'.\par',
                r'\end{minipage}\par\medskip',
            ])
        (TABLES / f'examples_{group}.tex').write_text('\n'.join(content) + '\n')

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False,
                         'axes.spines.right': False, 'pdf.fonttype': 42, 'axes.titleweight': 'bold'})
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.25), gridspec_kw={'width_ratios': [1, 1.45]})
    vals = [r['test']['kl'] for r in runs.values()]
    axes[0].bar(list(ALIASES.values()), vals, color=COLORS, width=.65)
    axes[0].set(ylabel='Test KL (lower is better)', ylim=(0, .8), title='Official test results')
    for i, value in enumerate(vals):
        axes[0].text(i, value + .02, f'{value:.3f}', ha='center', fontsize=8)
    contrasts = list(report['paired_kl'].values())[1:]
    contrast_names = ['B2 - B1', 'B2 - Set', 'Set - B1', 'A5 - B1', 'B2 shuffled - ordered']
    for i, v in enumerate(contrasts):
        lo, hi = v['percentile_95_ci']
        axes[1].plot([lo, hi], [i, i], color='#087f8c', linewidth=2)
        axes[1].scatter(v['difference'], i, color='#087f8c', s=24, zorder=3)
    axes[1].axvline(0, color='#67757d', linestyle='--', linewidth=1)
    axes[1].set(yticks=range(5), yticklabels=contrast_names, xticks=[-.004, 0, .004, .008],
                xlabel='Difference in test KL', title='95% paired intervals')
    axes[1].invert_yaxis()
    fig.tight_layout(w_pad=2)
    fig.savefig(FIGURES / 'results_and_uncertainty.pdf', bbox_inches='tight')
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    for (name, run), color in zip(runs.items(), COLORS):
        group = run['diagnostics']['strata']['entropy']
        axes[0].plot(range(1, 5), [group[str(i)]['metrics']['kl'] for i in range(1, 5)],
                     marker='o', markersize=4, label=ALIASES[name], color=color)
        if name != 'b0_prior':
            history = json.loads((ROOT / 'outputs/experiments' / run['run_id'] / 'history.json').read_text())
            axes[1].plot([h['epoch'] for h in history], [h['val']['kl'] for h in history], label=ALIASES[name], color=color)
            chosen = run['training']['best_epoch']
            axes[1].scatter(chosen, run['val']['kl'], s=30, color=color)
    axes[0].set(xticks=[1, 2, 3, 4], xlabel='Target-entropy bin (cuts from training)', ylabel='Test KL', title='Entropy-stratified results')
    axes[0].legend(frameon=False, fontsize=7, ncol=2)
    axes[1].set(xlabel='Epoch', ylabel='Validation KL', title='Training and checkpoint selection')
    axes[1].legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout(w_pad=2)
    fig.savefig(FIGURES / 'diagnostics_and_training.pdf', bbox_inches='tight')
    plt.close(fig)
    manifest = {'source_results_sha256': hashlib.sha256(report_path.read_bytes()).hexdigest(),
                'run_ids': {n: r['run_id'] for n, r in runs.items()}, 'models': len(runs),
                'examples': len(examples), 'unique_example_movies': len(used_movies),
                'checks': ['Per-class scores reproduce saved weighted F1.', 'Per-clip KL reproduces saved test KL.',
                           'Sample IDs, movie IDs, targets, class order, and frame counts agree across models.',
                           'Twenty unique example clips and movies; five clips per requested group.']}
    (DEST / 'report_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
