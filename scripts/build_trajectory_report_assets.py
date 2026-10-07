"""Build LaTeX tables and figures from verified aggregate results, without video data."""
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.configure_trajectories import VARIANTS

DEST = ROOT / 'research_notes/latex/trajectory_report'
TABLES, FIGURES = DEST / 'tables', DEST / 'figures'
for path in (TABLES, FIGURES):
    path.mkdir(parents=True, exist_ok=True)
R = json.loads((ROOT / 'research_notes/trajectory_results.json').read_text())
D = json.loads((ROOT / 'research_notes/dataset_audit.json').read_text())
M = json.loads((ROOT / 'research_notes/trajectory_morning_status.json').read_text())
O = json.loads((ROOT / 'configs/trajectory_overnight.json').read_text())
H = json.loads((ROOT / 'research_notes/trajectory_report_inputs.json').read_text())
G = R['groups']
NAMES = list(VARIANTS) + O['variants']
IDS = {n: f'{"P" if i < 16 else "N"}{i+1 if i < 16 else i-15:02d}' for i,n in enumerate(NAMES)}
assert len(NAMES) == len(set(NAMES)) == 88
assert R['benchmark_run_count'] == 264

LABELS = {
    'dino_meanpool': 'DINO mean pooling', 'on_dino_meanpool_wide': 'DINO mean pooling, width 256',
    'traj_vad_duration': 'VAD duration', 'traj_vad_peak': 'VAD single peak',
    'traj_vad_class_peak': 'VAD peak per emotion', 'traj_vad_soft': 'VAD soft relevance',
    'traj_vad_sparse': 'VAD sparse relevance', 'traj_vad_power': 'VAD power pooling',
    'traj_vad_sparse_proximity': 'VAD raw proximity', 'traj_vad_sparse_permuted': 'VAD sparse, permuted labels',
    'traj_vad_duration_rare': 'VAD duration, rare sampling', 'traj_vad_sparse_rare': 'VAD sparse, rare sampling',
    'traj_vad_duration_importance': 'VAD duration, corrected sampling',
    'traj_vad_sparse_importance': 'VAD sparse, corrected sampling',
    'traj_free_soft': 'Free decoder, soft relevance', 'traj_free_sparse': 'Free decoder, sparse relevance',
    'traj_vad_sparse_no_time': 'VAD sparse, no timestamps',
    'on_vad_duration_free': 'Free decoder, duration', 'on_vad_power_free': 'Free decoder, power',
    'on_vad_soft_tau010': 'VAD soft, initial temperature 0.1',
    'on_free_sparse_shallow': 'Free sparse, one context layer',
    'on_free_sparse_rare': 'Free sparse, rare sampling',
    'on_free_sparse_rare_ipw': 'Free sparse, corrected sampling',
}


def pathtex(value):
    return r'\path{' + value + '}'


def table(name, caption, columns, headers, rows, small=True):
    size = r'\small' if small else r'\footnotesize'
    n = len(headers)
    head = ' & '.join(headers) + r' \\'
    lines = [r'\begingroup', size, r'\setlength{\tabcolsep}{4pt}',
             r'\begin{longtable}{' + columns + '}',
             r'\caption{' + caption + r'}\label{tab:' + name + r'}\\',
             r'\toprule', head, r'\midrule\endfirsthead',
             r'\multicolumn{' + str(n) + r'}{l}{\textit{Table \ref{tab:' + name + r'} continued}}\\',
             r'\toprule', head, r'\midrule\endhead',
             r'\midrule\multicolumn{' + str(n) + r'}{r}{\textit{Continued on the next page}}\\\endfoot',
             r'\bottomrule\endlastfoot']
    lines += [' & '.join(map(str, row)) + r' \\' for row in rows]
    lines += [r'\end{longtable}', r'\endgroup']
    (TABLES / f'{name}.tex').write_text('\n'.join(lines) + '\n')


def num(value):
    return f'{value:.4f}'


primary = []
for name in VARIANTS:
    row = G[name]
    primary.append([IDS[name], LABELS[name], num(row['val_mean']['kl']),
                    f"${row['test_mean']['kl']:.4f} \\pm {row['test_sd']['kl']:.4f}$",
                    num(row['test_mean']['mrr']), num(row['test_mean']['f1_top1'])])
table('primary', 'All 16 primary conditions. Test KL shows mean and sample standard deviation across three seeds.',
      'lp{5.6cm}rrrr', ['ID', 'Condition', 'Val. KL', 'Test KL', 'MRR', 'Top-1 F1'], primary)

selected = ['on_dino_meanpool_wide','dino_meanpool','on_vad_duration_free','on_vad_power_free',
            'traj_free_soft','traj_free_sparse','on_free_sparse_shallow','on_vad_soft_tau010']
rows = []
for name in selected:
    row = G[name]
    rows.append([LABELS[name], num(row['val_mean']['kl']),
                 f"${row['test_mean']['kl']:.4f} \\pm {row['test_sd']['kl']:.4f}$",
                 num(row['test_mean']['mrr']),num(row['test_mean']['f1_top1']),num(row['test_mean']['f1_top3'])])
table('selected','Selected comparisons, using means across seeds 42, 43 and 44.',
      'p{5.6cm}rrrrr',['Model','Val. KL','Test KL','MRR','F1@1','F1@3'],rows)

pair_labels = {
 'on_dino_meanpool_wide minus dino_meanpool': 'Wide minus standard mean pooling',
 'on_vad_duration_free minus dino_meanpool': 'Free duration minus mean pooling',
 'traj_free_soft minus on_vad_duration_free': 'Free soft minus free duration',
 'traj_free_sparse minus traj_free_soft': 'Free sparse minus free soft',
 'on_free_sparse_shallow minus traj_free_sparse': 'One-layer minus two-layer free sparse',
 'on_vad_soft_tau010 minus dino_meanpool': 'Best VAD minus mean pooling',
 'on_vad_soft_tau010 minus traj_vad_soft': 'VAD soft: initial 0.1 minus initial 0.25',
 'traj_vad_duration_rare minus traj_vad_duration': 'Rare-sampled minus natural VAD duration',
 'on_free_sparse_rare minus traj_free_sparse': 'Rare-sampled minus natural free sparse',
 'traj_vad_sparse_permuted minus traj_vad_sparse': 'Permuted minus sourced sparse VAD',
}
rows=[]
for key,row in R['seed_averaged_paired_kl'].items():
    lo,hi=row['percentile_95_ci']
    rows.append([pair_labels[key], f"{row['difference']:+.6f}", f'[{lo:+.6f}, {hi:+.6f}]'])
table('paired','Paired movie-bootstrap comparisons. Negative differences favor the first model.',
      'p{8.8cm}rr',['Comparison','KL difference','95\% interval'],rows)

rare_names=['dino_meanpool','on_dino_meanpool_wide','traj_free_sparse','on_free_sparse_rare','on_free_sparse_rare_ipw',
            'traj_vad_duration','traj_vad_duration_rare','traj_vad_sparse','traj_vad_sparse_rare','on_vad_soft_tau010']
rare_values={}
rows=[]
for name in rare_names:
    values=[R['runs'][f'{name}_s{s}']['rare_diagnostics'] for s in (42,43,44)]
    rare_values[name]={k:float(np.mean([v['groups']['rare'][k] for v in values]))
                       for k in ['mean_class_probability_mae','target_mass','predicted_mass']}
    rare_values[name]['brier']=float(np.mean([v['brier_sum'] for v in values]))
    v=rare_values[name]
    rows.append([LABELS[name],num(G[name]['test_mean']['kl']),f"{v['mean_class_probability_mae']:.6f}",
                 f"{v['predicted_mass']:.5f}",f"{v['brier']:.5f}"])
table('rare','Probability diagnostics, averaged across three seeds. Rare MAE and predicted mass use the six rare classes; Brier covers all 21 classes. Actual rare-group mass is 0.018256.',
      'p{6.5cm}rrrr',['Model','KL','Rare MAE','Predicted mass','Brier'],rows)

rows=[]
for name in ['dino_meanpool','on_dino_meanpool_wide','traj_free_soft','traj_free_sparse','on_vad_duration_free',
             'on_vad_soft_tau010','traj_vad_duration','traj_vad_peak','traj_vad_sparse']:
    runs=[R['runs'][f'{name}_s{s}'] for s in (42,43,44)]
    rows.append([LABELS[name],f"{runs[0]['training']['trainable_parameters']:,}",
                 '/'.join(str(x['training']['epochs_run']) for x in runs),
                 '/'.join(str(x['training']['best_epoch']) for x in runs)])
table('training','Declared trainable parameter counts and training duration. Epoch triplets follow seeds 42/43/44.',
      'p{7cm}rrr',['Model','Parameters','Epochs run','Selected epochs'],rows)

rows=[]
for name in ['traj_vad_duration','traj_vad_peak','traj_vad_soft','traj_vad_sparse','on_vad_soft_tau010',
             'traj_free_soft','traj_free_sparse']:
    v=[R['runs'][f'{name}_s{s}']['trajectory_summary'] for s in (42,43,44)]
    support=np.mean([x['mean_effective_contributing_moments'] for x in v])
    frac=np.mean([x['mean_contributing_fraction'] for x in v])
    tau='--' if v[0]['temperature'] is None else f"{np.mean([x['temperature'] for x in v]):.4f}"
    rows.append([LABELS[name],f'{support:.2f}',f'{frac:.3f}',tau])
table('support','Scene contribution diagnostics. The support fraction counts exactly nonzero scene contributions.',
      'p{7.3cm}rrr',['Model','Effective scenes','Support fraction','Fitted temperature'],rows)

rows=[]
for name in R['validation_ranking']:
    x=G[name]
    rows.append([IDS[name],pathtex(name),num(x['val_mean']['kl']),
                 f"${x['test_mean']['kl']:.4f} \\pm {x['test_sd']['kl']:.4f}$",num(x['test_mean']['mrr']),num(x['test_mean']['f1_top1'])])
table('all_conditions','Complete 88-condition results, ordered by mean validation KL. Configuration names are exact repository identifiers.',
      'lp{6.8cm}rrrr',['ID','Configuration','Val. KL','Test KL','MRR','F1@1'],rows,small=False)
rows=[]
for name in NAMES:
    x=G[name]
    rows.append([IDS[name],pathtex(name),*[f"{x['test_kl_per_seed'][str(s)]:.6f}" for s in (42,43,44)]])
table('seeds','Test KL for every trained seed; no seed is selected by its test score.',
      'lp{7cm}rrr',['ID','Configuration','Seed 42','Seed 43','Seed 44'],rows,small=False)
rows=[]
for name in NAMES:
    rows.append([IDS[name],pathtex(name),*[R['runs'][f'{name}_s{s}']['run_id'].rsplit('_',1)[1] for s in (42,43,44)]])
table('jobs','Complete job ledger. All 264 predictor jobs completed with exit code 0:0.',
      'lp{7cm}rrr',['ID','Configuration','Job: seed 42','Job: seed 43','Job: seed 44'],rows,small=False)

rows=[]
rare_labels=set(R['runs']['dino_meanpool_s42']['rare_diagnostics']['groups']['rare']['labels'])
for i,c in enumerate(D['class_order']):
    a=D['splits']['train'];b=D['splits']['test']
    rows.append([str(i+1),c,f"{a['per_label_mean_probability'][c]:.6f}",
                 str(a['dominant_label_frequency'].get(c,0)),str(b['dominant_label_frequency'].get(c,0)),
                 'Yes' if c in rare_labels else ''])
table('classes','Fixed class order and label imbalance. Dominant counts use metadata labels, which can differ from metric tie-breaking.',
      'rlrrrr',['Index','Reaction','Train probability','Train dominant','Test dominant','Rare'],rows)

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                     'axes.spines.right':False,'axes.labelcolor':'#183444','text.color':'#183444',
                     'savefig.bbox':'tight','pdf.fonttype':42,'ps.fonttype':42})
teal, orange, slate = '#087F8C','#C76435','#61778A'


def save(fig,name):
    fig.savefig(FIGURES/f'{name}.pdf')
    fig.savefig(FIGURES/f'{name}.png',dpi=150)
    plt.close(fig)


names=['on_dino_meanpool_wide','dino_meanpool','on_vad_duration_free','on_vad_power_free','traj_free_soft','traj_free_sparse',
       'on_vad_soft_tau010','traj_vad_duration','traj_vad_sparse_permuted','traj_vad_power','traj_vad_soft',
       'traj_vad_sparse','traj_vad_class_peak','traj_vad_peak']
fig,ax=plt.subplots(figsize=(9.3,6.6))
y=np.arange(len(names))
vals=[G[n]['test_mean']['kl'] for n in names]
colors=[teal if 'free' in n or 'meanpool' in n else orange for n in names]
ax.barh(y,vals,xerr=[G[n]['test_sd']['kl'] for n in names],color=colors,alpha=.85,capsize=2)
ax.set_yticks(y,[LABELS[n] for n in names]);ax.invert_yaxis();ax.set_xlabel('Test KL divergence (lower is better)')
ax.axvline(G['dino_meanpool']['test_mean']['kl'],color=slate,linestyle='--',linewidth=1)
for yy,v in zip(y,vals): ax.text(v+.04,yy,f'{v:.4f}',va='center',fontsize=9)
ax.set_xlim(0,2.04);ax.grid(axis='x',alpha=.15);fig.tight_layout();save(fig,'overview')

keys=list(R['seed_averaged_paired_kl'])
small=[k for k in keys if abs(R['seed_averaged_paired_kl'][k]['difference'])<.03]
large=[k for k in keys if k not in small]
fig,axs=plt.subplots(2,1,figsize=(9.5,7.2),gridspec_kw={'height_ratios':[len(small),len(large)]})
for ax,subset,title in zip(axs,[small,large],['Small differences: intervals often include zero','Large differences involving the VAD decoder']):
    for j,k in enumerate(subset):
        row=R['seed_averaged_paired_kl'][k];lo,hi=row['percentile_95_ci'];v=row['difference']
        ax.errorbar(v,j,xerr=[[v-lo],[hi-v]],fmt='o',color=teal if hi<0 else orange if lo>0 else slate,capsize=3)
    ax.set_yticks(range(len(subset)),[pair_labels[k] for k in subset],fontsize=9)
    ax.invert_yaxis();ax.axvline(0,color='black',linewidth=.8,linestyle='--');ax.set_title(title,loc='left',fontsize=11)
    ax.grid(axis='x',alpha=.15);ax.set_xlabel('Difference in test KL: first model minus second')
fig.tight_layout();save(fig,'paired')

poolings=['duration','peak','class_peak','soft','sparse','power','sparse_proximity']
data=[]
for p in poolings:
    names=[f'on_vad_{p}_tau010',f'traj_vad_{p}',f'on_vad_{p}_tau060',f'on_vad_{p}_tau120']
    data.append([G[n]['test_mean']['kl'] for n in names])
data=np.asarray(data)
fig,ax=plt.subplots(figsize=(7.8,4.5))
im=ax.imshow(data,cmap='YlOrBr',vmin=.58,vmax=2.1,aspect='auto')
ax.set_xticks(range(4),['0.10','0.25 (primary)','0.60','1.20']);ax.set_xlabel('Initial VAD distance temperature (learned thereafter)')
ax.set_yticks(range(7),['Duration','Single peak','Peak per emotion','Soft relevance','Sparse relevance','Power','Raw proximity'])
for i in range(7):
    for j in range(4):ax.text(j,i,f'{data[i,j]:.3f}',ha='center',va='center',color='white' if data[i,j]>1.55 else '#183444')
fig.colorbar(im,ax=ax,label='Mean test KL');fig.tight_layout();save(fig,'temperature')

fig,axs=plt.subplots(1,2,figsize=(10.2,4.3))
shown=['dino_meanpool_s42','on_dino_meanpool_wide_s42','traj_free_soft_s42','on_vad_soft_tau010_s42','traj_vad_duration_s42','traj_vad_peak_s42']
for idx,name in enumerate(shown):
    history=H[name]['history'];base=name.rsplit('_s',1)[0]
    ax=axs[0] if idx<3 else axs[1]
    ax.plot([x['epoch'] for x in history],[x['val']['kl'] for x in history],label=LABELS[base],linewidth=1.7)
for ax in axs:
    ax.set_xlabel('Training epoch');ax.set_ylabel('Validation KL (lower is better)');ax.grid(alpha=.15)
    ax.legend(fontsize=8,loc='best')
axs[0].set_title('Unrestricted visual predictors',fontsize=11);axs[1].set_title('Fixed VAD predictors',fontsize=11)
fig.tight_layout();save(fig,'learning_curves')

names=['dino_meanpool','traj_free_sparse','on_free_sparse_rare','on_free_sparse_rare_ipw','traj_vad_duration','traj_vad_duration_rare','on_vad_soft_tau010']
fig,axs=plt.subplots(1,2,figsize=(10,4.4),sharey=True)
y=np.arange(len(names))
axs[0].barh(y,[rare_values[n]['mean_class_probability_mae'] for n in names],color=[teal]*4+[orange]*3)
axs[1].barh(y,[rare_values[n]['predicted_mass'] for n in names],color=[teal]*4+[orange]*3)
axs[1].axvline(rare_values[names[0]]['target_mass'],color='black',linestyle='--',label='Actual group mass')
axs[0].set_yticks(y,[LABELS[n] for n in names],fontsize=9);axs[0].invert_yaxis()
axs[0].set_xlabel('Rare-class mean absolute probability error');axs[1].set_xlabel('Predicted probability mass of six rare classes')
axs[1].legend(fontsize=8)
for ax in axs:ax.grid(axis='x',alpha=.15)
fig.tight_layout();save(fig,'rare')

source_files=['research_notes/trajectory_results.json','research_notes/trajectory_morning_status.json',
              'research_notes/trajectory_report_inputs.json','research_notes/dataset_audit.json',
              'research_notes/trajectory_time_audit.json','configs/trajectory_overnight.json']
manifest={'benchmark_runs':264,'conditions':88,'seeds':[42,43,44],
          'audit_job':'27711358','audit_completed_at_utc':R['checked_at_utc'],
          'source_sha256':{f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in source_files},
          'tables':sorted(p.name for p in TABLES.glob('*.tex')),
          'figures':sorted(p.name for p in FIGURES.glob('*.pdf')),
          'scope':'Derived tables/plots from saved aggregate results; no benchmark inference or metric recomputation on Mac.'}
(DEST/'report_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(manifest,indent=2))
