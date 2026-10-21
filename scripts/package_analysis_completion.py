"""Build a local landing page/checklist for existing analysis and new experiments."""
import argparse
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import shutil


def main(args):
    root, out = args.root.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    previous = root/'outputs/analysis/2026-10-19-error-analysis'
    shutil.copytree(previous, out/'01_previous_run_analysis', dirs_exist_ok=True)
    videos = out/'02_video_explorers'; videos.mkdir(exist_ok=True)
    for folder, count in [('2026-10-19-interactive-trajectories',10),('2026-10-19-best-worst-trajectories',20)]:
        source=root/'outputs/analysis'/folder
        for name in [f'video_trajectories_{count}.html', f'trajectory_examples_{count}.json',
                     'comment_availability.json', 'selection_verification.json']:
            if (source/name).exists(): shutil.copy2(source/name,videos/name)
    reports=out/'05_reports';reports.mkdir(exist_ok=True)
    for name in ['video2reaction_error_analysis.pdf','video2reaction_trajectory_report.pdf','video2reaction_trajectory_report_source.zip']:
        shutil.copy2(root/'output/pdf'/name,reports/name)
    shutil.copy2(root/'research_notes/error_analysis/completion_plan.md',out/'EXPERIMENT_PLAN.md')
    shutil.copy2(root/'research_notes/error_analysis/projection_review.md',out/'HYPOTHESIS_REVIEW.md')
    launch=root/'research_notes/error_analysis/projection_launch.json'
    if launch.exists(): shutil.copy2(launch,out/'launch_record.json')
    summary_path=out/'03_vad_to_dino/summary/summary.json'
    complete=summary_path.exists() and json.loads(summary_path.read_text())['status']=='completed'
    status='Completed across all three seeds' if complete else 'Submitted / awaiting verified complete results'
    rows=[('Video embeddings in 3D','Complete','01_previous_run_analysis/embeddings_3d.html'),
          ('Video reaction-label distributions in 3D','Complete','01_previous_run_analysis/embeddings_3d.html'),
          ('Predicted versus observed distributions: DINO and our VAD models','Complete','01_previous_run_analysis/figures/distribution_bias.png'),
          ('All eleven metrics and per-video failure analysis','Complete','01_previous_run_analysis/all_previous_runs.csv'),
          ('Rare / long-tail reaction error analysis','Complete','01_previous_run_analysis/figures/rare_recall.png'),
          ('VAD to DINO experiment',status,'03_vad_to_dino/summary/index.html' if complete else 'EXPERIMENT_PLAN.md'),
          ('Exact paired comparison with the paper’s fine-tuned model','Blocked: predictions/checkpoint missing','04_paper_comparison/status.json')]
    now=datetime.now(timezone.utc).isoformat()
    lines=['# Video2Reaction: analysis and experiment outputs','',f'Updated {now}. Deadline: 22 October 2026, 06:00 Amsterdam.','',
           'Open **index.html** for the browsable results. Full videos and the full dataset remain on Snellius.','',
           '## Checklist','']
    for title,state,link in rows:
        lines.append(f"- [{'x' if state.startswith('Complete') else ' '}] [{title}]({link}) — {state}.")
    lines+=['','## Folder guide','',
        '- `01_previous_run_analysis`: numerical analysis of 308 previous runs, plots, CSV/JSON files and report source.',
        '- `02_video_explorers`: offline 3D trajectories, scene frames and all-emotion timelines for ten examples and the 20 best/worst cases.',
        '- `03_vad_to_dino`: new reconstruction experiments, per-seed outputs, diagnostics, models, logs and summary.',
        '- `04_paper_comparison`: exact comparison dependency and search record.',
        '- `05_reports`: compiled reports for the previous experiments.',
        '', 'The paper-model row is intentionally incomplete: aggregate paper scores cannot supply its per-video distributions.']
    (out/'README.md').write_text('\n'.join(lines)+'\n')
    (out/'STATUS.md').write_text('\n'.join(lines[:len(rows)+9])+'\n')
    cards=[('Results report','05_reports/video2reaction_error_analysis.pdf','Twelve-page error analysis across previous runs.'),
           ('Experiment report','05_reports/video2reaction_trajectory_report.pdf','Hypotheses, model details and earlier trajectory experiments.'),
           ('20 best / worst videos','02_video_explorers/video_trajectories_20.html','Scene frames, emotion timelines, distributions and 3D playback.'),
           ('Video and label embeddings','01_previous_run_analysis/embeddings_3d.html','Interactive three-dimensional views of video features and labels.')]
    if complete:cards.insert(0,('New: VAD to DINO','03_vad_to_dino/summary/index.html','Three-seed reconstruction experiment, all metrics and interactive projection.'))
    content=''.join(f'<a class="card" href="{url}"><h2>{html.escape(title)}</h2><p>{html.escape(text)}</p></a>' for title,url,text in cards)
    table=''.join(f'<tr><td><a href="{url}">{html.escape(title)}</a></td><td>{html.escape(state)}</td></tr>' for title,state,url in rows)
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Video2Reaction · Analysis outputs</title><style>
body{font:16px/1.6 system-ui,sans-serif;color:#213343;background:#f5f7f8;margin:0}main{max-width:1120px;margin:50px auto;padding:0 24px}
h1{font-size:36px;line-height:1.2}h2{font-size:21px}.eyebrow{color:#356f75;font-weight:700}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:18px;margin:30px 0}
.card{display:block;background:white;border:1px solid #d9e1e5;border-radius:12px;padding:22px;text-decoration:none;color:inherit}.card:hover{border-color:#2b7180}.card h2{margin:0 0 10px}.card p{margin:0;color:#526878}
table{border-collapse:collapse;width:100%;background:white}td,th{padding:13px 16px;border-bottom:1px solid #e1e8eb;text-align:left}a{color:#12647b}.note{background:#fff4d9;border-left:4px solid #c58c20;padding:15px 20px}footer{color:#61717a;margin-top:35px;font-size:14px}
</style><main><div class="eyebrow">VIDEO2REACTION · RESEARCH OUTPUTS</div><h1>Experiments and error analysis</h1>
<p>One folder for the previous-run analysis, interactive video examples, and the new VAD-to-DINO experiment.</p><div class="grid">'''+content+'''</div>
<h2>Requested work</h2><table><thead><tr><th>Item</th><th>Status</th></tr></thead><tbody>'''+table+'''</tbody></table>
<p class="note"><strong>Paper-model comparison:</strong> the fine-tuned checkpoint or per-video predictions are still missing. Our visual-description model is a separate baseline. No paired comparison is claimed.</p>
<p><a href="EXPERIMENT_PLAN.md">Experiment design</a> · <a href="HYPOTHESIS_REVIEW.md">Hypothesis and code review</a> · <a href="README.md">Folder guide</a> · <a href="STATUS.md">Markdown checklist</a> · <a href="01_previous_run_analysis/analysis.json">Previous results JSON</a></p>
<footer>Updated '''+html.escape(now)+'''. Full videos, the dataset, and complete DINO feature arrays remain on Snellius.</footer></main></html>'''
    (out/'index.html').write_text(page)
    manifest=[]
    for path in sorted(out.rglob('*')):
        if not path.is_file() or path.name in {'manifest.json','manifest.sha256'}:continue
        digest=hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda:f.read(8*1024*1024),b''):digest.update(block)
        manifest.append({'path':str(path.relative_to(out)),'bytes':path.stat().st_size,'sha256':digest.hexdigest()})
    (out/'manifest.json').write_text(json.dumps({'updated_at':now,'files':manifest,'new_projection_complete':complete},indent=2)+'\n')
    print(json.dumps({'folder':str(out),'files':len(manifest),'projection_complete':complete,'paper_paired_complete':False}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path.cwd());p.add_argument('--out',type=Path,required=True)
    main(p.parse_args())
