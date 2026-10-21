"""Collect completed projection runs and render a portable results supplement."""
import argparse
import html
import json
from pathlib import Path
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.analyze_previous_runs import dump, write_csv

NAMES = {'original_dino': 'Original DINO', 'train_mean': 'Training mean', 'dino_pca3': 'DINO: 3 components',
         'dino_pca32': 'DINO: 32 components', 'vad_linear': 'VAD: linear', 'vad_quadratic': 'VAD: quadratic',
         'vad_rbf': 'VAD: radial basis', 'shuffled_vad_rbf': 'Shuffled VAD control', 'original_vad': 'Original VAD'}


def main(args):
    out = args.out; out.mkdir(parents=True, exist_ok=True)
    runs, paths = {}, {}
    pattern = f'seed*_{args.array_id}/results.json' if args.array_id else 'seed*/results.json'
    for p in sorted(args.input.glob(pattern)):
        value = json.loads(p.read_text())
        if value['smoke']:
            continue
        seed = value['seed']
        if seed in runs:
            raise ValueError(f'Duplicate completed seed {seed}; provide a folder with one attempt per seed')
        runs[seed], paths[seed] = value, p.parent
    missing = sorted(set([42, 43, 44]) - set(runs))
    if missing:
        dump(out / 'status.json', {'status': 'incomplete', 'completed_seeds': sorted(runs), 'missing_seeds': missing})
        if not args.allow_partial or not runs:
            raise RuntimeError(f'Incomplete experiment: missing seeds {missing}')
    summary, rows = {}, []
    for name in NAMES:
        per_seed = [runs[s]['results'][name] for s in sorted(runs)]
        metrics = {k: {'mean': float(np.mean([d['metrics'][k] for d in per_seed])),
                       'std': float(np.std([d['metrics'][k] for d in per_seed], ddof=1)) if len(per_seed)>1 else 0.}
                   for k in per_seed[0]['metrics']}
        reconstruction = {k: float(np.mean([d['reconstruction'][k] for d in per_seed]))
                          for k in per_seed[0].get('reconstruction', {})}
        summary[name] = {'metrics': metrics, 'reconstruction': reconstruction,
            'rare_target_mass': float(np.mean([d['groups']['rare']['target_mass'] for d in per_seed])),
            'rare_predicted_mass': float(np.mean([d['groups']['rare']['predicted_mass'] for d in per_seed]))}
        for seed in sorted(runs):
            d = runs[seed]['results'][name]
            rows.append({'seed': seed, 'method': name, **d['metrics'],
                         **{f'reconstruction_{k}': v for k,v in d.get('reconstruction', {}).items()},
                         'rare_target_mass': d['groups']['rare']['target_mass'],
                         'rare_predicted_mass': d['groups']['rare']['predicted_mass']})
    fields = sorted(set().union(*(r.keys() for r in rows)))
    write_csv(out / 'all_seed_metrics.csv', [{k:r.get(k) for k in fields} for r in rows])
    # Choose the mapping from validation reconstruction scores, never test outcomes.
    val = {name: float(np.mean([r['selected'][name]['val_mse'] for r in runs.values()]))
           for name in ('vad_linear', 'vad_quadratic', 'vad_rbf')}
    chosen = min(val, key=val.get)
    top3_sets = {}
    for seed in sorted(runs):
        with np.load(paths[seed]/'plot_arrays.npz',allow_pickle=False) as f:
            q=f['prediction_'+chosen];classes=f['class_order']
            sets=np.unique(np.sort(np.argsort(q,axis=1)[:,-3:],axis=1),axis=0)
            top3_sets[str(seed)]=[[str(classes[i]) for i in row] for row in sets]
    report = {'status': 'incomplete' if missing else 'completed', 'seeds': sorted(runs),
              'missing_seeds': missing, 'selected_mapping_by_validation': chosen,
              'mean_validation_reconstruction_mse': val, 'summary': summary, 'preferred_mapping_top3_sets_by_seed':top3_sets,
              'paper_paired_comparison': 'blocked: no fine-tuned paper checkpoint or per-video predictions available'}
    dump(out / 'summary.json', report)
    dump(out / 'status.json', {'status': report['status'], 'completed_seeds': sorted(runs), 'missing_seeds': missing})
    figures = out / 'figures'; figures.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size': 13, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, ax = plt.subplots(figsize=(10, 2.7), constrained_layout=True)
    ax.set(xlim=(0, 12), ylim=(0, 2.6)); ax.axis('off')
    boxes = [('Scene DINO\n768 values', '#e5eff5'), ('Frozen trajectory\nmodel: 3 VAD values', '#e5eff5'),
             ('Learned reverse map\n768 DINO values', '#f6dfcf'), ('Frozen DINO\nreaction head', '#e5eff5'),
             ('21 reaction\nprobabilities', '#e4f1e9')]
    for i,(label,color) in enumerate(boxes):
        x0 = .05 + 2.4*i
        ax.add_patch(FancyBboxPatch((x0,.95),2.05,1.0,boxstyle='round,pad=.05',facecolor=color,edgecolor='#466273'))
        ax.text(x0+1.025,1.45,label,ha='center',va='center',fontsize=11.5)
        if i<4: ax.annotate('',xy=(x0+2.35,1.45),xytext=(x0+2.08,1.45),arrowprops={'arrowstyle':'->','color':'#466273'})
    ax.text(6,2.35,'VAD-to-DINO experiment: scene reconstruction followed by reaction prediction',ha='center',weight='bold',fontsize=12)
    ax.text(6,.45,'Only the reverse map is fitted. Training DINO features are its targets; validation error selects its parameters.',ha='center',fontsize=10)
    ax.text(6,.10,'At test time, inputs are visual. The original DINO prediction path and component bottlenecks are controls.',ha='center',fontsize=10)
    fig.savefig(figures/'method_schematic.png',dpi=180);fig.savefig(figures/'method_schematic.pdf');fig.savefig(figures/'method_schematic.svg');plt.close(fig)
    labels = [NAMES[n] for n in NAMES]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for ax, metric, title in [(axes[0], 'kl', 'Reaction-distribution KL (lower is better)'),
                              (axes[1], 'f1_top1', 'Dominant-reaction F1 (higher is better)')]:
        ax.barh(x, [summary[n]['metrics'][metric]['mean'] for n in NAMES],
                xerr=[summary[n]['metrics'][metric]['std'] for n in NAMES], color=['#2a6f97' if 'vad' not in n else '#b85c38' for n in NAMES])
        ax.set_yticks(x, labels); ax.invert_yaxis(); ax.set_title(title); ax.grid(axis='x', alpha=.2)
    fig.savefig(figures/'reaction_scores.png', dpi=180); fig.savefig(figures/'reaction_scores.pdf'); plt.close(fig)
    methods = [n for n in NAMES if n != 'original_vad']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    for ax, metric, title in [(axes[0], 'r2_vs_train_mean', 'Reconstruction: explained error reduction'),
                              (axes[1], 'centered_cosine', 'Cosine after subtracting training mean')]:
        ax.barh(np.arange(len(methods)), [summary[n]['reconstruction'][metric] for n in methods], color='#3b7c6a')
        ax.set_yticks(np.arange(len(methods)), [NAMES[n] for n in methods]); ax.invert_yaxis(); ax.set_title(title); ax.axvline(0,color='black',lw=.7)
    fig.savefig(figures/'reconstruction.png', dpi=180); fig.savefig(figures/'reconstruction.pdf'); plt.close(fig)
    seed = min(runs)
    with np.load(paths[seed]/'plot_arrays.npz',allow_pickle=False) as f:
        arrays = {k:f[k] for k in f.files}
    classes = arrays['class_order'].tolist()
    selected = ['original_dino', chosen, 'original_vad']
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), constrained_layout=True)
    axes[0].plot(arrays['target'].mean(0), 'ko-', label='Observed audience distribution')
    for name in selected:
        q = arrays['prediction_'+name]
        axes[0].plot(q.mean(0), '.-', label=NAMES[name])
        ratio = q.std(0)/np.maximum(arrays['target'].std(0),1e-12)
        axes[1].plot(ratio, '.-', label=NAMES[name])
    for ax in axes:
        ax.set_xticks(np.arange(len(classes)), classes, rotation=45, ha='right'); ax.grid(alpha=.2)
    axes[0].set_title(f'Class means: representative seed {seed}'); axes[0].set_ylabel('Probability'); axes[0].legend(fontsize=11)
    axes[1].axhline(1,color='black',linestyle='--',label='Matches observed variation')
    axes[1].set_ylabel('Predicted / observed standard deviation'); axes[1].set_title('Does the model vary enough between clips?')
    fig.savefig(figures/'distribution_differences.png',dpi=180);fig.savefig(figures/'distribution_differences.pdf');plt.close(fig)
    coords = arrays['original_dino_pca3']
    mapped = arrays[chosen+'_pca3']
    dominant = arrays['target'].argmax(1)
    hover = [f'{sid}<br>Observed dominant reaction: {classes[c]}' for sid,c in zip(arrays['sample_id'],dominant)]
    graph = go.Figure()
    graph.add_trace(go.Scatter3d(x=coords[:,0].tolist(),y=coords[:,1].tolist(),z=coords[:,2].tolist(),mode='markers',name='Original video DINO mean',
        marker={'size':3,'color':dominant.tolist(),'colorscale':'Turbo','opacity':.55},text=hover,hovertemplate='%{text}<extra></extra>'))
    graph.add_trace(go.Scatter3d(x=mapped[:,0].tolist(),y=mapped[:,1].tolist(),z=mapped[:,2].tolist(),mode='markers',name='Reconstructed video DINO mean',
        marker={'size':3,'color':dominant.tolist(),'colorscale':'Turbo','symbol':'diamond'},text=hover,hovertemplate='%{text}<extra></extra>'))
    graph.update_layout(title={'text':f'VAD to DINO: actual and reconstructed video features, seed {seed}<br>'
        '<sup>Shared axes fitted on sampled training scenes; points are video means. Toggle traces with the legend.</sup>'},
        scene={'xaxis_title':'DINO component 1','yaxis_title':'DINO component 2','zaxis_title':'DINO component 3'},height=800)
    graph.write_html(out/'projection_3d.html',include_plotlyjs=True,full_html=True)
    best = summary[chosen]; original = summary['original_dino']
    rare_note=(f"The six training-defined rare reactions carry {100*best['rare_target_mass']:.4f}% of observed probability. "
               f"Original DINO predicts {100*original['rare_predicted_mass']:.4f}%; the selected reconstruction predicts "
               f"{100*best['rare_predicted_mass']:.4f}%. Clark distance improves from "
               f"{original['metrics']['clark']['mean']:.4f} to {best['metrics']['clark']['mean']:.4f}, while KL and dominant-reaction F1 worsen. "
               "A better Clark score alone does not establish better rare-reaction prediction.")
    if all(len(v)==1 for v in top3_sets.values()) and len({tuple(v[0]) for v in top3_sets.values()})==1:
        rare_note += ' Every test clip has the same three highest-ranked reactions in every seed: '+', '.join(next(iter(top3_sets.values()))[0])+'.'
    conclusion = (f"The validation-selected mapping is {NAMES[chosen]}. Its three-seed mean test reconstruction "
                  f"explained error reduction is {best['reconstruction']['r2_vs_train_mean']:.4f}. "
                  f"The frozen DINO reaction head gives KL {best['metrics']['kl']['mean']:.4f}, compared with "
                  f"{original['metrics']['kl']['mean']:.4f} on original DINO features. "
                  f"Lower KL is better. These results test retained visual information and compatibility with the frozen head; "
                  f"they do not test a newly trained reaction head or prove that all possible VAD models fail.")
    lines = ['# VAD to DINO: results supplement', '', f"Status: {report['status']}. Seeds: {sorted(runs)}.", '', conclusion, '',
             '## Mean results', '', '| Method | KL ↓ | Top-1 F1 ↑ | Reconstruction R² ↑ |', '|---|---:|---:|---:|']
    for n,s in summary.items():
        r2 = s['reconstruction'].get('r2_vs_train_mean')
        lines.append(f"| {NAMES[n]} | {s['metrics']['kl']['mean']:.4f} | {s['metrics']['f1_top1']['mean']:.4f} | {r2:.4f} |" if r2 is not None else
                     f"| {NAMES[n]} | {s['metrics']['kl']['mean']:.4f} | {s['metrics']['f1_top1']['mean']:.4f} | n/a |")
    lines += ['', '## Rare reactions and ranking collapse', '', rare_note, '', '## What was tested', '',
              'Three learned scene VAD coordinates were mapped to 768 DINO dimensions using linear, quadratic and radial-basis ridge regression. '
              'A shuffled-clip control breaks the training relationship between VAD and visual features. Models use at most eight scenes per training clip. '
              'Validation reconstruction error selects hyperparameters. Every test scene is evaluated after choices are frozen.', '',
              'The reaction head is the original frozen DINO mean-pooling model: per-scene layer normalization, linear projection, '
              'mean pooling, and its existing nonlinear classifier. DINO principal-component controls isolate a general low-dimensional bottleneck.', '',
              'R² here means one minus reconstruction error divided by training-mean prediction error, averaged with equal clip weight. '
              'It can be negative. Centered cosine removes the common feature mean. Retrieval asks whether reconstructed features identify '
              'their own clip among all test clips. All eleven reaction metrics and rare-class tables are in the CSV and per-seed diagnostics.', '',
              '## Limits', '',
              '- The original VAD model was trained on the training clips; decoder fitting uses those in-sample coordinates.',
              '- Held-out test reaction labels are used only for final scoring, never to construct coordinates or select mappings.',
              '- Three-seed error bars are standard deviations, not confidence intervals. Per-seed KL intervals resample movies.',
              '- No scene-level emotion annotations establish whether the saved coordinates are psychologically accurate.',
              '- The exact paired paper-model comparison remains blocked by missing fine-tuned predictions/checkpoints.', '',
              '## Files', '', '[All metrics](all_seed_metrics.csv) · [Machine-readable summary](summary.json) · [Interactive 3D projection](projection_3d.html)',
              '', '[Original paper](https://arxiv.org/html/2607.06875v1). Published aggregates cannot replace paired predictions.']
    (out/'README.md').write_text('\n'.join(lines)+'\n')
    # LaTeX source stays editable and builds independently from these local figures.
    tex_rows = '\n'.join(NAMES[n].replace('&',r'\&') + f" & {s['metrics']['kl']['mean']:.4f} & {s['metrics']['f1_top1']['mean']:.4f}" + r'\\' for n,s in summary.items())
    metric_names={'chebyshev':r'Largest class error $\downarrow$', 'kl':r'KL divergence $\downarrow$',
        'clark':r'Clark distance $\downarrow$', 'cad':r'Cumulative absolute distance $\downarrow$',
        'cosine':r'Cosine similarity $\uparrow$', 'intersection':r'Probability overlap $\uparrow$',
        'mrr':r'Mean reciprocal rank $\uparrow$', 'tpe':r'Top-reaction probability error $\downarrow$',
        'f1_top1':r'Weighted Top-1 F1 $\uparrow$', 'f1_top2':r'Weighted Top-2 F1 $\uparrow$', 'f1_top3':r'Weighted Top-3 F1 $\uparrow$'}
    metric_rows='\n'.join(label+' & '+' & '.join(f"{summary[n]['metrics'][metric]['mean']:.4f}" for n in ['original_dino',chosen,'original_vad'])+r'\\' for metric,label in metric_names.items())
    tex = r'''\documentclass[11pt,a4paper]{article}
\usepackage[margin=22mm]{geometry}\usepackage{graphicx,booktabs,hyperref}
\title{VAD to DINO: Reconstruction and Reaction Prediction}
\author{Video2Reaction experiment supplement}\date{22 October 2026}
\begin{document}\maketitle
\section{Question and result}
Can the three learned valence, arousal and dominance coordinates retain enough visual information to recover useful DINO embeddings?
''' + conclusion.replace('_',r'\_') + r'''
\section{Execution and model details}
The three-video smoke test was Snellius job 28448335. The full seed jobs were 28448337\_0, 28448337\_1 and 28448337\_2, for seeds 42, 43 and 44. All completed with exit code 0:0 on 22 October 2026. They used frozen source commit \texttt{81b34098f278}, H100 allocations and an isolated uv environment with PyTorch 2.8.0 and NumPy 2.2.6. The official splits contain 7,243 training, 1,035 validation and 2,070 test clips.

The cached visual encoder is DINOv2-base: each scene keyframe has a 768-dimensional visual feature. The existing VAD model projects this to 128 dimensions and applies two Transformer layers, with four attention heads per layer. This lets each scene use context from other scenes. A small nonlinear output layer produces three coordinates in $[-1,1]$. They were learned from video-level reaction labels with fixed NRC emotion coordinates, not from observed scene-level emotion labels. The original VAD predictor additionally uses scene relevance and duration to combine reaction probabilities.

The separate DINO baseline applies layer normalization to each 768-dimensional scene feature, maps it linearly to 128 dimensions, averages the scene representations, and applies a 128-unit hidden layer with a GELU activation followed by 21 output logits and a softmax. This entire predictor stays frozen when evaluating reconstructed features.

\section{Experiment}
\includegraphics[width=\textwidth]{figures/method_schematic.pdf}
Each saved visual trajectory model maps scene DINO features to three VAD coordinates. We freeze it and learn the reverse mapping to 768 DINO dimensions. The mappings are linear ridge regression, quadratic ridge regression, and radial-basis ridge regression. Radial basis functions respond to coordinates near 128 training centers. Ridge regularization limits coefficient magnitude.

We use up to eight evenly spaced scenes per training clip, with equal total clip weight. Training data alone determine normalization, centers and coefficients. Validation reconstruction mean squared error selects ridge strength from 0.0001, 0.01 and 1.0, and radial-basis bandwidth from 0.5, 1 and 2. All selections are frozen before test evaluation. The three existing paired checkpoint seeds are 42, 43 and 44.

The controls are the training-mean feature, DINO reconstruction through three or 32 principal components, and a radial-basis mapping with training trajectories shuffled between clips. The test covers every scene of all 2,070 official test clips. No comment, description, text embedding or test reaction label enters the visual mapping.

\section{Reaction prediction}
Reconstructed scene features are passed through the unchanged DINO predictor: scene layer normalization, a linear projection, mean pooling, then its nonlinear reaction classifier. We do not train a new reaction head. KL measures distribution mismatch and lower is better. Top-1 F1 combines precision and recall for the dominant reaction and higher is better.
\begin{center}\begin{tabular}{lrr}\toprule Method & Mean KL & Mean Top-1 F1\\\midrule
''' + tex_rows + r'''\bottomrule\end{tabular}\end{center}
\includegraphics[width=\textwidth]{figures/reaction_scores.pdf}
\section{Reconstruction and distribution diagnostics}
Explained error reduction is one minus reconstruction error divided by the error of predicting the training-mean feature. It can be negative. Both errors average scenes within clips, then average clips. Centered cosine subtracts the training mean so a common feature offset cannot dominate similarity. Three-seed error bars show standard deviation, not confidence intervals.
\includegraphics[width=\textwidth]{figures/reconstruction.pdf}
\includegraphics[width=\textwidth]{figures/distribution_differences.pdf}
\section{All metrics and rare reactions}
\begin{center}\small\begin{tabular}{lrrr}\toprule Metric & DINO & Reconstructed & Original VAD\\\midrule
''' + metric_rows + r'''\bottomrule\end{tabular}\end{center}
KL penalizes missing target probability. Largest class error measures the worst class discrepancy. Clark distance measures relative errors; the implementation adds $10^{-8}$ to its squared denominator. Cumulative absolute distance compares cumulative probabilities in the fixed taxonomy order. Cosine compares distribution direction, and probability overlap sums the shared mass. Mean reciprocal rank rewards placing the true dominant reaction near the top. Top-reaction probability error measures its probability mismatch. Weighted Top-$k$ F1 combines precision and recall for the selected $k$ reactions, weighted by their target support.

''' + rare_note.replace('%',r'\%') + r'''

In this implementation, very small predictions on absent classes contribute little to Clark distance because of its denominator constant. That can coexist with severe underprediction when a rare reaction is present. The per-class support and probability errors therefore matter alongside the aggregate score.
\section{Interpretation and limits}
This experiment tests information retained by learned VAD coordinates and compatibility with a frozen DINO head. A reverse mapping cannot create information absent from its input. Poor results can reflect information loss, a limited mapping family, or a mismatch with the frozen reaction head; they do not rule out every VAD method. No scene-level ground truth verifies the emotional interpretation of the coordinates. The original VAD model's training coordinates are in-sample; validation and test remain held out for the new decoder.

The machine-readable supplement includes every benchmark metric, per-video errors, per-class support, rare-class diagnostics and movie-cluster bootstrap intervals for paired KL differences. The interactive 3D file shows original and reconstructed video features on the same training-fitted DINO axes.
\section{Paper-model comparison}
The exact paired comparison with the paper's fine-tuned LLaVA or Qwen model is blocked because its per-video predictions and fine-tuned checkpoint were not found in the project or public model account. Our visual-description baseline is a different model and is not substituted. Published aggregate scores do not permit paired distribution analysis.
\url{https://arxiv.org/html/2607.06875v1}
\end{document}
'''
    (out/'projection_report.tex').write_text(tex)
    # Cluster fallback PDF: self-contained summary and figures. Local LaTeX build replaces it.
    import textwrap
    with PdfPages(out/'projection_report.pdf') as pdf:
        fig = plt.figure(figsize=(8.27,11.69)); fig.text(.09,.93,'VAD to DINO: results supplement',fontsize=18,weight='bold')
        paragraphs = [conclusion,
            'Protocol: three paired seeds, training-only scene sampling and fitting, validation-only hyperparameter selection, all 2,070 test clips. '
            'Inputs are visually predicted scene coordinates; no held-out reaction label is used as an input.',
            'The following pages show reaction scores, feature reconstruction, and differences in class probability and variation. '
            'All eleven benchmark metrics and rare-class diagnostics are supplied in the companion files.',
            'Exact paper-model paired analysis is blocked by missing fine-tuned predictions/checkpoints. '
            'Source paper: https://arxiv.org/html/2607.06875v1']
        y=.86
        for paragraph in paragraphs:
            text='\n'.join(textwrap.wrap(paragraph,90));fig.text(.09,y,text,fontsize=11,va='top',linespacing=1.5);y-=.027*(text.count('\n')+1)+.05
        pdf.savefig(fig);plt.close(fig)
        for name in ['method_schematic','reaction_scores','reconstruction','distribution_differences']:
            fig,ax=plt.subplots(figsize=(11.69,8.27));ax.imshow(plt.imread(figures/(name+'.png')));ax.axis('off');pdf.savefig(fig,bbox_inches='tight');plt.close(fig)
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>VAD to DINO results</title>'
        '<style>body{max-width:1050px;margin:40px auto;font:17px/1.6 system-ui;padding:0 20px;color:#193244}img{width:100%}a{color:#05668d}</style>'
        '<h1>VAD to DINO: results</h1><p>'+html.escape(conclusion)+'</p><p>'+html.escape(rare_note)+'</p>'
        '<p><a href="projection_report.pdf">PDF report</a> · <a href="projection_report.tex">LaTeX source</a> · '
        '<a href="projection_3d.html">Interactive 3D plot</a> · <a href="all_seed_metrics.csv">All metrics</a> · '
        '<a href="summary.json">JSON results</a></p>' + ''.join(f'<img src="figures/{n}.png" alt="{n}">' for n in ['method_schematic','reaction_scores','reconstruction','distribution_differences']) +
        '<h2>Paper-model comparison</h2><p>The exact paired comparison requires the paper’s fine-tuned predictions or checkpoint. It remains missing.</p>')
    print(json.dumps({'status':report['status'],'selected':chosen,'output':str(out)}),flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--array-id');p.add_argument('--allow-partial',action='store_true');main(p.parse_args())
