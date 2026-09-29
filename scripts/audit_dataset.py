import collections, csv, hashlib, json, math, statistics, struct
from pathlib import Path
import argparse, os
parser=argparse.ArgumentParser()
parser.add_argument('--frames', default=os.environ.get('V2R_FRAME_DIR'), required='V2R_FRAME_DIR' not in os.environ)
parser.add_argument('--metadata', default=os.environ.get('V2R_METADATA_DIR'), required='V2R_METADATA_DIR' not in os.environ)
parser.add_argument('--output', type=Path, required=True)
args=parser.parse_args()

root = Path(args.frames)
meta = Path(args.metadata)
classes = ['sadness','disgust','grief','fear','disapproval','disappointment','embarrassment','nervousness','annoyance','anger','confusion','realization','caring','curiosity','relief','approval','surprise','excitement','amusement','admiration','joy']
def stats(a):
    a = sorted(a)
    def pct(q):
        pos = (len(a)-1)*q/100; lo = math.floor(pos); hi = math.ceil(pos)
        return a[lo] + (a[hi]-a[lo])*(pos-lo)
    return dict(n=len(a), min=min(a), max=max(a), mean=statistics.mean(a), median=statistics.median(a), percentiles={str(p):pct(p) for p in [1,5,25,50,75,95,99]})
def jpeg_size(path):
    with path.open('rb') as f:
        if f.read(2) != b'\xff\xd8': return None
        while True:
            b=f.read(1)
            if not b: return None
            if b != b'\xff': continue
            marker=f.read(1)
            while marker == b'\xff': marker=f.read(1)
            code=marker[0]
            if code in [0xd8,0xd9]: continue
            size=struct.unpack('>H',f.read(2))[0]
            if code in [0xc0,0xc1,0xc2]:
                _,h,w=struct.unpack('>BHH',f.read(5)); return [w,h]
            f.seek(size-2,1)

splits={s:json.loads((meta/(s+'.json')).read_text()) for s in ['train','val','test']}
report={'frame_root':str(root),'metadata_root':str(meta),'class_order':classes,'splits':{},'cross_split':{},'image_size_sample':{},'errors':[]}
all_counts=[]; dimensions=collections.Counter(); suffixes=collections.Counter(); frame_columns=set(); seen=0
for split, data in splits.items():
    counts=[]; entropy=[]; dom=[]; dominant=collections.Counter(); positive=collections.Counter(); prob=collections.Counter(); sums=[]; ties=0
    movie_ids=set(); missing_fields=collections.Counter(); mismatch=0; unknown=set(); nonmonotonic=0
    for video, row in data.items():
        p=row['reaction_outcome']['reaction_distribution']; unknown.update(set(p)-set(classes)); sums.append(sum(p.values()))
        if any(v<0 or not math.isfinite(v) for v in p.values()): report['errors'].append([video,'invalid probabilities'])
        entropy.append(-sum(v*math.log(v) for v in p.values() if v>0)); dom.append(max(p.values())); dominant[row['reaction_outcome']['dominant_reaction']]+=1
        winners=[c for c in classes if p.get(c,0)==max(p.values())]; ties+=len(winners)>1
        mismatch+=row['reaction_outcome']['dominant_reaction'] not in winners
        prob.update(p); positive.update(c for c in classes if p.get(c,0)>0)
        if row.get('imdbid'): movie_ids.add(row['imdbid'])
        for key in ['clip_description','imdbid','movie_name','genre']:
            if not row.get(key): missing_fields[key]+=1
        folder=root/video
        if not (folder/'index.csv').exists(): report['errors'].append([video,'missing index']); continue
        with (folder/'index.csv').open() as f:
            reader=csv.DictReader(f); frame_columns.update(reader.fieldnames); frames=list(reader)
        expected=[f"{int(r['scene_number'])+1:03d}.jpg" for r in frames]
        files={p.name for p in folder.iterdir()}
        missing=set(expected)-files
        if missing: report['errors'].append([video,'missing images',sorted(missing)])
        starts=[int(r['start_frame']) for r in frames]
        nonmonotonic+=any(a>=b for a,b in zip(starts,starts[1:]))
        counts.append(len(frames)); all_counts.append(len(frames))
        suffixes.update(Path(n).suffix for n in files)
        if seen%50==0 and expected and not missing:
            sz=jpeg_size(folder/expected[0]); dimensions[str(sz)]+=1
        seen+=1
    mean={c:prob[c]/len(data) for c in classes}
    report['splits'][split]={'size':len(data),'sha256':hashlib.sha256((meta/(split+'.json')).read_bytes()).hexdigest(), 'keyframes':stats(counts),'entropy_nats':stats(entropy),'dominant_probability':stats(dom),'dominant_label_frequency':dict(dominant),'per_label_mean_probability':mean,'label_presence':dict(positive),'mean_probability_imbalance':max(mean.values())/min(mean.values()),'dominant_ties':ties,'metadata_dominant_not_argmax':mismatch,'target_sum_min':min(sums),'target_sum_max':max(sums),'unknown_labels':sorted(unknown),'distinct_movies':len(movie_ids),'missing_fields':dict(missing_fields),'nonmonotonic_indices':nonmonotonic}
    print('audited',split,len(data),'clips',flush=True)
for a,b in [('train','val'),('train','test'),('val','test')]:
    ma={r['imdbid'] for r in splits[a].values() if r.get('imdbid')}; mb={r['imdbid'] for r in splits[b].values() if r.get('imdbid')}
    da={r['clip_description'].strip().lower() for r in splits[a].values()}; db={r['clip_description'].strip().lower() for r in splits[b].values()}
    shared=ma&mb
    report['cross_split'][a+'_'+b]={'duplicate_video_ids':sorted(set(splits[a])&set(splits[b])), 'shared_movie_count':len(shared),'clips_in_a_with_shared_movie':sum(r.get('imdbid') in shared for r in splits[a].values()),'clips_in_b_with_shared_movie':sum(r.get('imdbid') in shared for r in splits[b].values()),'duplicate_descriptions':len(da&db)}
report['keyframes_overall']=stats(all_counts); report['frame_index_columns']=sorted(frame_columns)
report['image_size_sample']={'sampling':'first image of every 50th clip in official split order; JPEG header inspected', 'count':sum(dimensions.values()),'width_height_counts':dict(dimensions)}
report['file_suffix_counts']=dict(suffixes)
out=args.output; out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
