"""Render the compact cluster audit locally; no dataset/features are required."""
import argparse
import csv
import html
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ORDER = ['dino_meanpool_s42','on_dino_meanpool_wide_s42','on_vad_soft_tau010_s42',
         'traj_vad_soft_s42','traj_vad_sparse_s42','on_vad_class_peak_tau010_s42',
         'on_vad_duration_free_s42','traj_free_soft_s42','traj_free_sparse_s42',
         'b1_meanpool','e_shared_query_control','visual_description','b0_prior']
SHORT = ['DINO baseline','DINO wide','VAD best','VAD soft','VAD sparse','VAD class peak',
         'Free duration','Free soft','Free sparse','SigLIP baseline','SigLIP shared',
         'Visual + text','Training prior']
FOCUS = [0,2,10,11]


def esc(s):
    s=str(s)
    return ''.join({'\\':r'\textbackslash{}','&':r'\&','%':r'\%','$':r'\$',
        '#':r'\#','_':r'\_','{':r'\{','}':r'\}','~':r'\textasciitilde{}','^':r'\textasciicircum{}'}.get(c,c) for c in s)


def save(fig,out,name):
    fig.savefig(out/f'{name}.png',dpi=180,bbox_inches='tight')
    fig.savefig(out/f'{name}.pdf',bbox_inches='tight')
    plt.close(fig)


def table(headers,rows,columns=None):
    columns=columns or 'l'+'r'*(len(headers)-1)
    return '\n'.join([r'\begin{center}\small',r'\begin{tabular}{'+columns+'}',r'\toprule',
        ' & '.join(esc(x) for x in headers)+r' \\ \midrule',
        *[' & '.join(esc(x) for x in row)+r' \\' for row in rows],
        r'\bottomrule\end{tabular}\end{center}'])


def main(out):
    a=json.loads((out/'analysis.json').read_text())
    spaceinfo=json.loads((out/'spaces.json').read_text())
    metadata=json.loads((out/'sample_metadata.json').read_text())
    with np.load(out/'plot_arrays.npz') as f:
        data={k:f[k] for k in f.files}
    with np.load(out/'spaces.npz') as f:
        spaces={k:f[k] for k in f.files}
    p=data['target']; classes=list(data['class_order']); rare=data['rare_indices']
    q=[data['prediction_'+n] for n in ORDER]
    details=[a['detail'][n] for n in ORDER]
    figures=out/'figures';figures.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'font.family':'DejaVu Sans','savefig.facecolor':'white'})
    colors=['#2463A1','#D47720','#3A8765','#925B9B']
    labels=[c+(' *' if i in rare else '') for i,c in enumerate(classes)]
    fig,ax=plt.subplots(2,1,figsize=(12,9),gridspec_kw={'height_ratios':[1,1.5]},layout='constrained')
    x=np.arange(21)
    ax[0].bar(x-.3,p.mean(0),.15,label='Ground truth',color='#202b35')
    for j,k in enumerate(FOCUS):
        ax[0].bar(x-.15+j*.15,q[k].mean(0),.15,label=SHORT[k],color=colors[j])
    ax[0].set(xticks=x,xticklabels=labels,ylabel='Mean probability',title='Population reaction mass: average over 2,070 test videos')
    ax[0].tick_params(axis='x',rotation=60);ax[0].legend(ncol=3,fontsize=9)
    bias=np.array([(v-p).mean(0) for v in q])
    im=ax[1].imshow(bias,cmap='RdBu_r',vmin=-abs(bias).max(),vmax=abs(bias).max(),aspect='auto')
    ax[1].set(xticks=x,xticklabels=labels,yticks=np.arange(len(q)),yticklabels=SHORT,
        title='Signed bias: prediction minus target; red = too much probability')
    ax[1].tick_params(axis='x',rotation=60);fig.colorbar(im,ax=ax[1],label='Probability difference')
    save(fig,figures,'distribution_bias')

    fig,axs=plt.subplots(2,2,figsize=(11,8),layout='constrained')
    hp=-(p*np.log(np.maximum(p,1e-30))).sum(1)
    for ax,k in zip(axs.flat,FOCUS):
        hq=-(q[k]*np.log(np.maximum(q[k],1e-30))).sum(1)
        im=ax.hexbin(hp,hq,gridsize=35,mincnt=1,cmap='Blues',extent=(0,3.05,0,3.05))
        ax.plot([0,3.05],[0,3.05],':',c='#B34535')
        ax.set(title=f'{SHORT[k]}: mean {hq.mean():.3f} vs target {hp.mean():.3f}',
               xlabel='Ground-truth entropy (nats)',ylabel='Predicted entropy (nats)',xlim=(0,3.05),ylim=(0,3.05))
        fig.colorbar(im,ax=ax,label='Videos')
    save(fig,figures,'entropy')

    variation={}
    fig,axs=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    for j,k in enumerate(FOCUS):
        ratio=q[k].std(0)/p.std(0)
        correlation=float(np.corrcoef(q[k][:,rare].sum(1),p[:,rare].sum(1))[0,1])
        variation[ORDER[k]]={'median_class_sd_ratio':float(np.median(ratio)),
            'median_rare_class_sd_ratio':float(np.median(ratio[rare])),
            'rare_mass_pearson_r':correlation}
        axs[0].plot(x,ratio,'o-',label=SHORT[k],c=colors[j],markersize=3)
        axs[1].scatter(p[:,rare].sum(1),q[k][:,rare].sum(1),s=5,alpha=.15,c=colors[j],label=SHORT[k])
    axs[0].axhline(1,c='gray',linestyle=':')
    axs[0].set(xticks=x,xticklabels=labels,ylabel='Predicted SD / target SD',title='Variation across videos, for each reaction')
    axs[0].tick_params(axis='x',rotation=65,labelsize=8);axs[0].legend(fontsize=8)
    axs[1].plot([0,.5],[0,.5],':',c='gray')
    axs[1].set(xlabel='Ground-truth rare mass',ylabel='Predicted rare mass',title='Does rare probability follow the right videos?')
    save(fig,figures,'variation')
    (out/'descriptive_correlations.json').write_text(json.dumps(variation,indent=2)+'\n')

    fig,axs=plt.subplots(2,1,figsize=(12,9),layout='constrained')
    for ax,k in zip(axs,[1,3]):
        vals=np.array([[r[f'top{k}_recall'] for r in d['class_rows']] for d in details])
        im=ax.imshow(vals,aspect='auto',vmin=0,vmax=1,cmap='YlGnBu')
        ax.set(xticks=x,xticklabels=labels,yticks=np.arange(len(q)),yticklabels=SHORT,title=f'Per-reaction Top-{k} recall (* = rare in training)')
        ax.tick_params(axis='x',rotation=60)
        for row in range(len(q)):
            for c in range(21):
                if details[row]['class_rows'][c][f'top{k}_target_support']==0:
                    ax.text(c,row,'-',ha='center',va='center',fontsize=9)
        fig.colorbar(im,ax=ax,label='Recall; dash = no target support')
    save(fig,figures,'rare_recall')

    metrics=list(details[0]['metrics'])
    values=np.array([[d['metrics'][m] for m in metrics] for d in details])
    oriented=values.copy()
    for j,m in enumerate(metrics):
        if m in {'kl','chebyshev','clark','cad','tpe'}:
            oriented[:,j]*=-1
    normalized=(oriented-oriented.min(0))/(np.ptp(oriented,axis=0)+1e-12)
    fig,ax=plt.subplots(figsize=(12,6.5),layout='constrained')
    ax.imshow(normalized,aspect='auto',cmap='RdYlGn',vmin=0,vmax=1)
    ax.set(xticks=np.arange(len(metrics)),xticklabels=[m.replace('f1_top','F1@') for m in metrics],yticks=np.arange(len(q)),yticklabels=SHORT,
        title='All official metrics: values shown; greener = better within this column')
    for i in range(len(q)):
        for j in range(len(metrics)):
            ax.text(j,i,f'{values[i,j]:.3f}',ha='center',va='center',fontsize=8)
    save(fig,figures,'all_metrics')

    fig,axs=plt.subplots(2,2,figsize=(11,8),layout='constrained')
    stratanames=['all','rare_present','rare_mass_at_least_10pct','dominant_outside_train_top3','unseen_movie']
    stratalabels=['All videos','Any rare mass','Rare >=10%','Outside top 3','Unseen movie']
    for ax,metric in zip(axs.flat,['kl','cosine','mrr','f1_top1']):
        for j,k in enumerate(FOCUS):
            vals=[details[k]['metrics'][metric]]+[details[k]['strata'][s]['metrics'][metric] for s in stratanames[1:]]
            ax.plot(range(5),vals,'o-',label=SHORT[k],c=colors[j])
        ns=[len(p)]+[details[0]['strata'][s]['n'] for s in stratanames[1:]]
        ax.set(xticks=range(5),xticklabels=[f'{l}\nn={n}' for l,n in zip(stratalabels,ns)],ylabel=metric,title=metric.replace('f1_top1','Top-1 weighted F1'))
        ax.tick_params(axis='x',labelsize=8);ax.grid(axis='y',alpha=.2)
    axs[0,0].legend(fontsize=8)
    save(fig,figures,'strata')

    fig,axs=plt.subplots(2,2,figsize=(11,8),layout='constrained')
    for ax,k in zip(axs.flat,FOCUS):
        for group,label,color in [(np.arange(21),'All reactions','#2463A1'),(rare,'Rare reactions','#D47720')]:
            xx,yy,ss=[],[],[]
            for b in range(10):
                mask=np.minimum((q[k][:,group]*10).astype(int),9)==b
                if mask.any():
                    xx.append(q[k][:,group][mask].mean());yy.append(p[:,group][mask].mean());ss.append(mask.sum())
            ax.plot(xx,yy,'o-',label=label,c=color)
        ax.plot([0,1],[0,1],':',c='gray');ax.set(title=SHORT[k],xlabel='Mean predicted probability in bin',ylabel='Mean ground-truth probability in bin',xlim=(0,1),ylim=(0,1))
        ax.legend(fontsize=8)
    save(fig,figures,'calibration')

    # Numeric error examples are selected by a declared rule, not by semantic anecdotes.
    failures=list(csv.DictReader((out/'dino_meanpool_s42_per_video.csv').open()))
    chosen_ids=[r['sample_id'] for r in failures[:3]]
    rare_fail=sorted(failures,key=lambda r:-float(r['rare_target_mass']))
    for r in rare_fail:
        if r['sample_id'] not in chosen_ids: chosen_ids.append(r['sample_id'])
        if len(chosen_ids)>=6: break
    fig,axs=plt.subplots(3,2,figsize=(12,11),layout='constrained')
    examples=[]
    for ax,vid in zip(axs.flat,chosen_ids):
        i=list(data['sample_id']).index(vid)
        shown=np.argsort(p[i]+q[0][i]+q[2][i])[-7:]
        for j,(v,name,col) in enumerate([(p,'Target','#202b35'),(q[0],'DINO','#2463A1'),(q[2],'VAD','#D47720')]):
            ax.bar(np.arange(7)+(j-1)*.23,v[i,shown],.23,label=name,color=col)
        ax.set(xticks=np.arange(7),xticklabels=[classes[c] for c in shown],ylabel='Probability',
            title=f'{vid}\n{metadata[vid]["clip_name"][:48]}')
        ax.tick_params(axis='x',rotation=55,labelsize=8);ax.legend(fontsize=8)
        examples.append({'sample_id':vid,'youtube_url':'https://www.youtube.com/watch?v='+vid,
            'selection':'Largest DINO KL' if vid in chosen_ids[:3] else 'Largest rare target mass',**metadata[vid]})
    save(fig,figures,'failure_examples')
    (out/'example_selection.json').write_text(json.dumps(examples,indent=2)+'\n')

    fig=plt.figure(figsize=(12,4.6),layout='constrained')
    space_axes=[]
    for j,(key,title) in enumerate([('dino_pca3','DINO video means: training-fit PCA'),('label_pca3','21-label distributions: training-fit PCA'),('label_expected_vad','Label-weighted lexicon coordinates')]):
        ax=fig.add_subplot(1,3,j+1,projection='3d')
        space_axes.append(ax)
        v=spaces[key]
        points=ax.scatter(*v.T,c=p[:,rare].sum(1),cmap='viridis',vmin=0,vmax=p[:,rare].sum(1).max(),s=5,alpha=.55)
        ax.set_title(title,fontsize=9)
        ax.set_xlabel('Valence' if j==2 else 'PC 1',fontsize=8);ax.set_ylabel('Arousal' if j==2 else 'PC 2',fontsize=8);ax.set_zlabel('Dominance' if j==2 else 'PC 3',fontsize=8)
        ax.tick_params(labelsize=7)
    fig.colorbar(points,ax=space_axes,orientation='horizontal',fraction=.045,pad=.06,label='Rare ground-truth probability')
    save(fig,figures,'spaces3d')
    interactive(out,data,spaces,metadata,a)

    traj=json.loads((out/'on_vad_soft_tau010_s42_trajectory_examples.json').read_text())
    traj=sorted(traj,key=lambda v:v['kl'])
    fig=plt.figure(figsize=(12,8),layout='constrained')
    for j,ex in enumerate([traj[0],traj[-1]]):
        for col,key in enumerate(['vad','dino_pca3']):
            ax=fig.add_subplot(2,2,j*2+col+1,projection='3d')
            v=np.asarray(ex[key]);w=np.asarray(ex['weights'])
            ax.plot(*v.T,c='#697d8d',alpha=.6,lw=1)
            ax.scatter(*v.T,c=np.arange(len(v)),cmap='viridis',s=15+150*w/max(w.max(),1e-8))
            ax.set_title(f'{ex["sample_id"]}, KL={ex["kl"]:.3f}: '+('saved VAD' if col==0 else 'DINO PCA'),fontsize=10)
            for dim,setlabel in enumerate([ax.set_xlabel,ax.set_ylabel,ax.set_zlabel]):
                setlabel(['Valence','Arousal','Dominance'][dim] if col==0 else f'PC {dim+1}',fontsize=8)
            ax.tick_params(labelsize=7)
    save(fig,figures,'trajectories')
    write_report(out,a,spaceinfo,details,examples)
    print('Rendered figures, interactive plots, and report source:',out)


def interactive(out,data,spaces,metadata,a):
    p=data['target'];rare=data['rare_indices'];classes=list(data['class_order'])
    hover=[]
    for i,v in enumerate(data['sample_id']):
        top=np.argsort(p[i])[-5:][::-1]
        hover.append(html.escape(str(v))+'<br>'+html.escape(metadata[v]['clip_name'])+'<br>'+', '.join(f'{classes[c]}: {p[i,c]:.2f}' for c in top))
    fig=make_subplots(rows=1,cols=3,specs=[[{'type':'scene'}]*3],subplot_titles=['DINO mean embeddings (PCA)','Reaction distributions (PCA)','Label-weighted VAD'])
    for j,key in enumerate(['dino_pca3','label_pca3','label_expected_vad']):
        v=spaces[key]
        fig.add_trace(go.Scatter3d(x=v[:,0],y=v[:,1],z=v[:,2],mode='markers',text=hover,hovertemplate='%{text}<extra></extra>',
            marker=dict(size=3,color=p[:,rare].sum(1),colorscale='Viridis',cmin=0,cmax=1,showscale=j==2,colorbar=dict(title='Probability')),showlegend=False),row=1,col=j+1)
    buttons=[]
    for label,values in [('Rare reaction mass',p[:,rare].sum(1))]+[(c,p[:,i]) for i,c in enumerate(classes)]:
        buttons.append(dict(label=label,method='restyle',args=[{'marker.color':[values.tolist()]*3}]))
    fig.update_layout(height=680,title='Same 2,070 test videos; colour by ground-truth reaction probability',
        updatemenus=[dict(buttons=buttons,x=0,y=1.12,xanchor='left')],margin=dict(t=120,b=10,l=0,r=0))
    for j in range(3):
        scene='scene'+(str(j+1) if j else '')
        fig.update_layout(**{scene:dict(xaxis_title='Valence' if j==2 else 'PC 1',yaxis_title='Arousal' if j==2 else 'PC 2',zaxis_title='Dominance' if j==2 else 'PC 3')})
    prefix='''<!doctype html><html><head><meta charset="utf-8"><title>Video2Reaction: previous-run error analysis</title><style>body{font:16px system-ui;margin:30px;color:#1b2c3d}p{max-width:1100px;line-height:1.5}</style></head><body><h1>Video embeddings and reaction distributions</h1><p>Rotate and zoom each panel. Select a reaction to colour the same videos by its full ground-truth probability. Hover for video ID, title, and the five largest label probabilities. DINO and label PCA use separate axes fitted on training data; their orientations cannot be compared directly. The VAD panel is the expectation of fixed lexicon coordinates under each label distribution, not a measured emotion trajectory.</p>'''
    (out/'embeddings_3d.html').write_text(prefix+fig.to_html(full_html=False,include_plotlyjs=True)+'</body></html>')
    # Saved VAD trajectories: choose a video, retain its actual three coordinates.
    examples=json.loads((out/'on_vad_soft_tau010_s42_trajectory_examples.json').read_text())
    fig=go.Figure()
    for j,ex in enumerate(examples):
        v=np.array(ex['vad']);w=np.array(ex['weights'])
        fig.add_trace(go.Scatter3d(x=v[:,0],y=v[:,1],z=v[:,2],mode='lines+markers',visible=j==0,
            marker=dict(size=3+10*w/max(w.max(),1e-8),color=np.arange(len(v)),colorscale='Viridis',showscale=True,colorbar=dict(title='Scene index')),
            name=ex['sample_id'],text=[f'Scene {i}, weight {weight:.4f}' for i,weight in enumerate(w)],hovertemplate='%{text}<br>V=%{x:.3f}, A=%{y:.3f}, D=%{z:.3f}<extra></extra>'))
    buttons=[dict(label=f'{ex["sample_id"]}: KL {ex["kl"]:.3f}',method='update',args=[{'visible':[i==j for i in range(len(examples))]}]) for j,ex in enumerate(examples)]
    fig.update_layout(height=750,title='Actual saved VAD coordinates: best VAD condition, seed 42',updatemenus=[dict(buttons=buttons,x=0,y=1.1)],scene=dict(xaxis_title='Valence',yaxis_title='Arousal',zaxis_title='Dominance'))
    (out/'vad_trajectories_3d.html').write_text('<!doctype html><meta charset="utf-8"><title>Saved VAD trajectories</title><h1 style="font-family:system-ui">Saved per-scene VAD trajectories</h1><p style="font-family:system-ui">Marker size reflects model relevance; colour shows scene order. These are predictions from a model with full-video context, not independently annotated emotional states. Examples include low error, high error and high rare-reaction mass.</p>'+fig.to_html(full_html=False,include_plotlyjs=True))


def write_report(out,a,spaceinfo,details,examples):
    dino=details[0];vad=details[2];prior=details[-1]
    rare_names=', '.join(a['rare_labels'])
    means=a['groups']
    variation=json.loads((out/'descriptive_correlations.json').read_text())
    metricheaders=['Model','KL','Cheb.','Clark','CAD','Cosine','Inter.']
    metricrows=[[s]+[f'{d["metrics"][m]:.4f}' for m in ['kl','chebyshev','clark','cad','cosine','intersection']] for s,d in zip(SHORT,details)]
    rankrows=[[s]+[f'{d["metrics"][m]:.4f}' for m in ['mrr','tpe','f1_top1','f1_top2','f1_top3']] for s,d in zip(SHORT,details)]
    rare_rows=[[s,f'{d["groups"]["rare"]["predicted_mass"]:.4f}',f'{d["groups"]["rare"]["macro_f1"]["1"]:.4f}',f'{d["groups"]["rare"]["macro_f1"]["3"]:.4f}'] for s,d in zip(SHORT,details)]
    class_rows=[[r['label'],f'{r["train_mass"]:.4f}',f'{r["target_mass"]:.4f}',str(int(r['top1_target_support'])),str(int(r['top3_target_support'])),f'{r["top1_recall"]:.3f}',f'{r["top3_recall"]:.3f}'] for r in dino['class_rows']]
    fig=lambda name,caption: '\n'+r'\begin{figure}[H]\centering\includegraphics[width=\linewidth]{figures/'+name+'.pdf}'+r'\caption{'+esc(caption)+r'}\end{figure}'+'\n'
    text=r'''\documentclass[10pt,a4paper]{article}
\usepackage[margin=18mm]{geometry}
\usepackage[T1]{fontenc}\usepackage{lmodern}\usepackage{graphicx,booktabs,float,longtable,xcolor,hyperref}
\usepackage[font=small,labelfont=bf]{caption}
\hypersetup{colorlinks=true,linkcolor=blue,urlcolor=blue}
\setlength{\parindent}{0pt}\setlength{\parskip}{6pt}
\begin{document}
\begin{center}{\LARGE Video2Reaction: error analysis of previous runs}\\[8pt]
{\large DINO embeddings, saved VAD trajectories, and reaction distributions}\\[5pt]
October 19, 2026\end{center}
\section{What this analysis answers}
This report checks where the previously trained predictors succeed and fail. It covers every available completed prediction run, all eleven official metrics, class frequency, distribution shape, and examples of large errors. No new predictor has been trained. The advertisement proposal is outside the scope.

The ground truth is a distribution of reactions extracted from comments. It is not a direct measurement of what each viewer felt. A video can have several plausible reactions, and rare reactions can matter even when they are not the dominant label.
'''
    text+=f"We independently recomputed the validation and test metrics for {a['run_count']} previous runs. Each test result uses the same 2,070 videos and 21 reactions. The original cluster environment (NumPy {a['numpy']}) preserves the benchmark's tie ordering. All recomputed values match the saved metrics within $10^{{-10}}$.\n"
    text+=r'''\subsection{Main findings}
'''
    text+=f"The DINO baseline's mean test KL over its three original seeds is {means['dino_meanpool']['test_mean']['kl']:.6f}; the wide DINO condition is {means['on_dino_meanpool_wide']['test_mean']['kl']:.6f}. The best VAD condition by the earlier validation ranking is {means['on_vad_soft_tau010']['test_mean']['kl']:.6f}. Lower KL is better. These are averages of separate model scores, not scores of an ensemble.\n"
    text+=f"The detailed plots below use seed 42 consistently. DINO assigns {dino['groups']['rare']['predicted_mass']:.4f} probability to the six rare reactions in total, against {dino['groups']['rare']['target_mass']:.4f} in the labels. The best VAD condition assigns {vad['groups']['rare']['predicted_mass']:.4f}. All six rare classes have zero dominant-label support in this test set, so their zero Top-1 macro F1 is uninformative. Their Top-3 recall is also zero for both models despite nonzero Top-3 support. Adding rare probability has not produced correct rare-reaction retrieval in these representative runs.\n"
    predicted_classes=sum(r['top1_prediction_support']>0 for r in dino['class_rows'])
    missed=[r['label'] for r in dino['class_rows'] if r['top1_target_support']>0 and r['top1_recall']==0]
    text+=f"DINO predicts only {predicted_classes} of the 21 classes as its top reaction. Supported classes with zero Top-1 recall are {esc(', '.join(missed))}. This is a stronger diagnostic of narrow recognition than a good aggregate KL score alone.\n"
    tail=dino['strata']['dominant_outside_train_top3']
    text+=f"Outside the three most common training reactions ({esc(', '.join(a['most_common_train_labels']))}), {tail['n']} test videos remain. DINO has Top-1 F1 {tail['metrics']['f1_top1']:.4f} on that subset, compared with {dino['metrics']['f1_top1']:.4f} overall. These class-frequency effects should be central to any next hypothesis.\n"
    text+=f"The mean target entropy is {dino['distribution']['target_entropy_mean']:.3f} nats. DINO predicts {dino['distribution']['prediction_entropy_mean']:.3f}; the best VAD condition predicts {vad['distribution']['prediction_entropy_mean']:.3f}. Larger entropy means probability is spread over more reactions. The plots test whether models are too diffuse, too concentrated, or fail differently across videos.\n"
    text+=r'''\section{Models and interpretation of comparisons}
All DINO runs use frozen DINOv2-base frame embeddings (768 numbers per scene keyframe). The baseline projects each frame and averages it before classifying the reaction distribution. The wide condition increases capacity. The VAD models use a temporal Transformer and predict three numbers per scene: valence, arousal and dominance. Distances to fixed NRC reaction coordinates produce per-scene probabilities; the pooling rule combines scenes. The free models predict 21 reaction logits without that three-dimensional distance constraint.

The original soft and sparse VAD conditions are shown as failure references. ``VAD best'' is the previously validation-selected low-temperature soft-pooling condition; it is not selected by this analysis. ``Class peak'' selects each reaction's strongest raw affinity across scenes. SigLIP2 baselines use a different, language-aligned image encoder. Visual + text uses the supplied clip description and is explicitly a text-assisted reference, not a pure-vision result. It is our trained model, not the paper authors' best model.

Each row in the detailed tables is one saved run (seed 42 for the DINO family). The full CSV contains every run. No three-seed mean is silently substituted for a single-seed curve.
\section{All official metrics}
KL is forward Kullback--Leibler divergence from the target distribution to the prediction. Chebyshev is the largest per-video probability error. Clark sums relative class errors, so very small probabilities matter. CAD compares cumulative probabilities in the benchmark's fixed class order; that order must not be changed. Cosine measures directional agreement; intersection measures shared probability mass. Lower is better for KL, Chebyshev, Clark and CAD; higher is better for cosine and intersection.
'''
    text+=table(metricheaders,metricrows)
    text+=r'''MRR is mean reciprocal rank of the target's most probable reaction. TPE is probability error at that target reaction. F1 at Top-1, Top-2 and Top-3 uses support-weighted multilabel F1, not per-video accuracy. Higher MRR and F1 are better; lower TPE is better. Official Top-k scoring can include zero-probability labels when fewer than k target reactions are nonzero. Argmax and Top-1 argsort use different tie conventions; both are preserved.
'''
    text+=table(['Model','MRR','TPE','F1@1','F1@2','F1@3'],rankrows)
    text+=fig('all_metrics','Green means better within a metric column among these representatives. Colours are scaled separately per column; numbers are the actual official scores. This is not a significance test.')
    text+=r'''\section{Predicted versus ground-truth distributions}
Population bias and individual-video error answer different questions. A model can reproduce the average class frequencies while assigning the wrong reactions to individual clips. Conversely, an occasional rare reaction can be predicted correctly while the total rare probability is inflated. We therefore inspect both average mass and per-video distributions.
'''
    text+=fig('distribution_bias','Asterisks mark the six rarest reactions by training probability mass. Red in the lower panel means too much predicted probability; blue means too little. Class order follows the official evaluator.')
    text+=fig('entropy','Entropy compares how spread out the distributions are. Points above the dotted diagonal are more diffuse than the corresponding labels. These are separate predictions for the same test clips.')
    text+=fig('variation','A ratio below one means the predictions vary less between videos than the target probabilities do. Matching the total rare mass is insufficient if that mass does not track which videos actually contain rare reactions. Standard deviations and Pearson correlations here are descriptive test-set summaries.')
    text+=f"DINO's median predicted-to-target standard-deviation ratio across the 21 classes is {variation['dino_meanpool_s42']['median_class_sd_ratio']:.3f}; for the six rare classes it is {variation['dino_meanpool_s42']['median_rare_class_sd_ratio']:.3f}. Its rare-mass correlation with the target is {variation['dino_meanpool_s42']['rare_mass_pearson_r']:.3f}, compared with {variation['on_vad_soft_tau010_s42']['rare_mass_pearson_r']:.3f} for the best VAD condition. Thus the VAD model assigns substantially more rare mass without tracking the videos containing that mass. The DINO model better matches the population total but still strongly reduces variation between videos.\n"
    text+=fig('calibration','For each probability bin, compare mean predicted mass with mean label mass. The diagonal indicates agreement in that bin. These are descriptive calibration curves against comment-derived soft labels, not calibration against direct human emotion measurements. Bin counts are saved in analysis.json.')
    text+=r'''\section{Rare and long-tail reactions}
'''
    text+='The rare set is fixed using training data only: '+esc(rare_names)+'. The following table reports total predicted rare mass and the unweighted average F1 across those six classes. The latter is a diagnostic and differs from the official support-weighted score. Unsupported classes receive zero by convention; their supports are shown separately.\n'
    text+=f"True rare probability mass on the test set is {dino['groups']['rare']['target_mass']:.6f}.\n"
    text+=table(['Model','Predicted rare mass','Rare macro F1@1','Rare macro F1@3'],rare_rows)
    text+=fig('rare_recall','Recall is shown for every class and model. Dashes mark classes with zero ground-truth Top-k support, for which a zero score is not evidence of measured recall. Rare classes are marked with an asterisk.')
    text+=table(['Reaction','Train mass','Test mass','Top-1 n','Top-3 n','DINO R@1','DINO R@3'],class_rows)
    text+=r'''\section{Where models fail}
Subsets include any positive rare-reaction mass, at least ten percent rare mass, and movies absent from training. The same clips are used across models within each subset. Entropy quartiles, training-defined rare-mass quartiles, frame-count quartiles and every dominant class also have all eleven metrics in the JSON. Repeated zero quantiles can create empty groups; these remain explicitly empty rather than being assigned fabricated scores.
'''
    text+=fig('strata','Subset performance is descriptive. Small groups, especially unseen movies and rare dominant reactions, support less precise conclusions. The official split has substantial movie overlap; these plots do not establish generalization to a new movie distribution.')
    strata_rows=[]
    for title,key in [('Any rare mass','rare_present'),('Rare mass at least 10%','rare_mass_at_least_10pct'),('Outside common three','dominant_outside_train_top3'),('Unseen movie','unseen_movie')]:
        strata_rows.append([title,str(dino['strata'][key]['n'])]+[f'{details[k]["strata"][key]["metrics"]["kl"]:.4f}' for k in FOCUS])
    text+=table(['Subset','Videos','DINO KL','VAD KL','SigLIP KL','Text KL'],strata_rows)
    interval=a['pairs_vs_dino_seed42']['on_vad_soft_tau010_s42']['kl']
    text+=f"Across all test clips, the best VAD condition's seed-42 KL exceeds DINO's by {interval['difference']:.4f}; the paired movie-bootstrap interval is [{interval['ci95'][0]:.4f}, {interval['ci95'][1]:.4f}]. This is a clear degradation for these saved models. It does not prove that every VAD-based model must fail. The current error pattern is excess probability spread and poor localization of rare mass to the right videos.\n"
    text+=f"DINO's lowest target-entropy quartile has KL {dino['strata']['entropy']['groups'][0]['metrics']['kl']:.4f}, versus {dino['strata']['entropy']['groups'][3]['metrics']['kl']:.4f} in the highest quartile. Confident, concentrated label distributions are therefore not automatically easier for this model: its more diffuse output can miss the dominant reaction.\n"
    text+=r'''\subsection{Concrete failures for inspection}
The first three examples have the largest DINO baseline KL. The next three have the largest rare-reaction mass among videos not already selected. Each plot shows the seven largest reactions under the sum of target, DINO and VAD probabilities; omitted classes remain in the numerical files. Titles come from metadata and are not a verified explanation of the failure. Following each video link supports a separate manual content review.
'''
    text+=fig('failure_examples','Target and prediction distributions for reproducibly selected failures. Large KL often reflects assigning too little mass to reactions that dominate the actual labels, which a population-average plot can hide.')
    for ex in examples:
        text+=r'\href{'+ex['youtube_url']+'}{'+esc(ex['sample_id'])+'}: '+esc(ex['movie_name']+' / '+ex['clip_name'])+'. Selection: '+esc(ex['selection'])+'.\n\par\n'
    text+=r'''\section{Three-dimensional embeddings and saved VAD}
There are three distinct spaces. DINO video embeddings are 768-dimensional; a three-component principal component analysis (PCA), fitted on training video means only, provides a view of their largest visual variations. Reaction labels are 21-dimensional probability vectors; a separate training-fit PCA displays their dominant variations. Neither PCA has valence, arousal and dominance axes.

The third view is the expected lexicon coordinate under a video's ground-truth reaction distribution. This makes a three-dimensional label plot but loses information: different distributions can have similar or identical centroids. It must not be interpreted as an independently annotated emotional trajectory.
'''
    text+=f"The first three DINO components retain {100*spaceinfo['dino_variance_retained']:.1f}\\% of training embedding variance. The first three label components retain {100*spaceinfo['label_variance_retained']:.1f}\\% of training label variance. A visually weak separation in three dimensions therefore cannot by itself disprove separability in the full feature space.\n"
    text+=fig('spaces3d','The same test videos appear in all three panels; colour encodes rare-reaction probability. The interactive HTML allows rotation and selection of each of the 21 reaction probabilities. PCA coordinate systems are separate, so positions across panels are not directly comparable.')
    text+=r'''\subsection{The checkpoint coordinates already exist}
We inspected the saved trajectory arrays and read the fixed prototype asset after verifying its checksum and class mapping against each run's provenance. No model inference is needed. For three representative VAD runs, all 91,312 test-scene coordinates are finite and have three dimensions. Their saved contributions independently reconstruct the final video predictions within export tolerance. Low-error, high-error and high-rare-mass trajectory examples are exported with scene order and model weights.
'''
    text+=fig('trajectories','Two saved example trajectories from the best VAD condition, beside the corresponding DINO frame features in the fixed training-fit PCA view. Colour indicates scene order; marker size indicates model relevance. Lines join scene keyframes, not dense observed motion.')
    geometry=[]
    for name,label in [('on_vad_soft_tau010_s42','VAD best'),('traj_vad_soft_s42','VAD soft'),('traj_vad_sparse_s42','VAD sparse')]:
        row=spaceinfo['trajectories'][name]
        geometry.append([label]+[f'{v:.3f}' for v in row['coordinate_sd']]+[f'{row["mean_within_clip_variance"]:.4f}',f'{row["median_effective_moments"]:.2f}'])
    text+=table(['Model','Valence SD','Arousal SD','Dominance SD','Within-clip var.','Eff. scenes'],geometry)
    text+='Coordinate standard deviations pool all test scenes, so longer clips contribute more scenes. Within-clip variance is contribution-weighted and then averaged over videos. Effective scenes is the median inverse concentration of each scene\'s total probability contribution. It is a model summary, not a ground-truth count of emotional events.\n\n'
    text+='All three inspected models have nonzero spread on every VAD axis and positive covariance eigenvalues. Their errors therefore cannot be explained by predicting one constant VAD point for all scenes. This does not validate the semantic meaning of the learned coordinates; there are no independent scene-level VAD annotations in this evaluation.\n'
    text+=r'''\subsection{VAD to DINO: what these results can establish}
The requested direction is VAD to DINO. The previous models predict VAD from visual features; they do not contain a learned inverse map from VAD back to DINO. A three-number bottleneck cannot uniquely recover every 768-dimensional visual feature. Many visually different scenes can share similar affective coordinates. A learned map could estimate a conditional visual prototype, but demonstrating that would require a separately specified reconstruction probe and held-out evaluation. This report does not train such a map or claim that visual proximity proves emotional agreement.

The saved coordinate ranges, covariance eigenvalues, within-video spread, and effective number of contributing scenes are in spaces.json. These permit checking collapse or excessive concentration without rerunning a model. The nearest-neighbour examples in label-expected VAD also show why a low-dimensional centroid can hide differences between full reaction distributions.
\section{Paper comparison and limitations}
The paper's fine-tuned LLaVA/Qwen per-video predictions were not found in the inspected Snellius outputs or original data tree. Therefore this report cannot compute paired errors, class biases or significance against those models. It does not relabel our visual-plus-description model as the paper model. The earlier paper comparison remains an aggregate comparison: SA-BFGS visual plus text has KL 0.588, while the fine-tuned LLaVA-Next row has MRR 0.7833, Top-1 F1 0.6521 and Top-3 F1 0.7672. These references use different methods and are not one combined target row. The DINO KL is numerically competitive, while its ranking and classification scores are lower.

Source: \url{https://arxiv.org/html/2607.06875v1}, Tables 5--7. Exact paired error analysis needs sample-aligned predictions from the desired paper checkpoint.

DINOv3 was not used in the previous experiments. Changing the encoder requires a new experiment and is outside this analysis. The \href{https://github.com/facebookresearch/dinov3}{official DINOv3 implementation} provides the model and integration details.

The paired confidence intervals saved in analysis.json resample movies, retaining all clips from each sampled movie. They describe fixed seed-42 models, use 3,000 draws, and are not corrected for multiple comparisons. They do not include training variability. Weighted F1 is non-additive and is reported directly rather than assigned an invalid average of per-video F1 values.

The test set has already been used repeatedly. These results identify failure modes; using them to change a model and reporting improvements on the same test set alone would overstate independent evidence. No claim of a new method, overall paper win, or causal emotional explanation follows from these diagnostic plots.
\section{Files and reproducibility}
\begin{itemize}
\item analysis.json: audited scores, per-class errors, calibration, subsets and paired intervals.
\item all\_previous\_runs.csv: all audited runs with every official test metric and validation KL.
\item Per-model per\_video.csv files: sortable failures and decomposable metrics. Top-k overlap counts are diagnostics, not official weighted F1.
\item Per-model per\_class.csv files: probability bias, error, support, precision, recall and F1.
\item embeddings\_3d.html: standalone interactive DINO and label plots.
\item vad\_trajectories\_3d.html: standalone interactive saved trajectories.
\item spaces.json / spaces.npz: compact projections and trajectory diagnostics. No raw images or full DINO feature cache are copied locally.
\item file\_manifest.json: checksums of the cluster analysis exports. Report source and plots are generated locally by scripts/render\_previous\_run\_analysis.py.
\end{itemize}
'''
    text+=table(['Representative','Original run ID'],[[s,d['run_id']] for s,d in zip(SHORT,details)],'lp{10cm}')
    text+=r'\end{document}'
    # Each generated prose line is a paragraph; keep table and figure markup intact.
    text=re.sub(r'(?<=\.)\n(?=[A-Za-z])','\n\n',text)
    (out/'error_analysis.tex').write_text(text)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    main(parser.parse_args().input)
