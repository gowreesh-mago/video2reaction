"""Build private VAD assets directly from the author's NRC v1 archive.

Generated coordinates are excluded from Git because the source disallows redistribution.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import urllib.request
import zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.experiments.taxonomy import REACTION_CLASSES

URL='https://saifmohammad.com/WebDocs/Lexicons/NRC-VAD-Lexicon.zip'
MEMBER='NRC-VAD-Lexicon/NRC-VAD-Lexicon.txt'

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--archive',type=Path,help='Previously downloaded original NRC archive; supports clusters blocked by the author website')
    args=parser.parse_args()
    if args.archive:
        raw=args.archive.read_bytes()
    else:
        request=urllib.request.Request(URL,headers={'User-Agent':'Mozilla/5.0'})
        raw=urllib.request.urlopen(request,timeout=120).read()
    archive=zipfile.ZipFile(io.BytesIO(raw))
    data=archive.read(MEMBER)
    lexicon={}
    for line in data.decode().splitlines():
        word,v,a,d=line.split('\t')
        if word in REACTION_CLASSES:
            lexicon[word]=dict(valence=float(v),arousal=float(a),dominance=float(d))
    if set(lexicon)!=set(REACTION_CLASSES):
        raise ValueError('Missing exact reaction words; no synonym approximations allowed')
    order=sorted(lexicon,key=lambda c:(lexicon[c]['valence'],lexicon[c]['arousal']))
    if order!=REACTION_CLASSES:
        raise ValueError('NRC v1 order differs from the published benchmark order')
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'nrc_vad.json').write_text(json.dumps(lexicon,indent=2)+'\n')
    source=dict(url=URL,version='1 (2018; README updated August 2022)',member=MEMBER,
                member_sha256=hashlib.sha256(data).hexdigest(),archive_sha256=hashlib.sha256(raw).hexdigest(),
                copyright='2018 National Research Council Canada (NRC)',
                author='Saif M. Mohammad',citation='Obtaining Reliable Human Ratings of Valence, Arousal, and Dominance for 20,000 English Words. ACL 2018.',
                terms='Non-commercial research/education; do not redistribute source data.')
    (args.output/'emotion_vad.json').write_text(json.dumps(dict(source=source,labels={
        c:dict(**lexicon[c],source_word=c,approximation=False,provenance='Exact English NRC VAD v1 entry; all three coordinates from the same row.')
        for c in REACTION_CLASSES}),indent=2)+'\n')
    (args.output/'NRC_README.txt').write_bytes(archive.read('NRC-VAD-Lexicon/README.txt'))
    print(json.dumps(dict(label_count=len(lexicon),order_verified=True,source=source),indent=2))
