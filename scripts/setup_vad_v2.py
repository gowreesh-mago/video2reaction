"""Provision exact private NRC v2.1 coordinates without changing the v1 assets."""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.experiments.taxonomy import REACTION_CLASSES

URL = 'https://www.saifmohammad.com/WebDocs/Lexicons/NRC-VAD-Lexicon-v2.1.zip'
ARCHIVE_SHA256 = '8bcd04831ffda149f683f8d3091c76a3aee138ee3aa1924b67b50b9416d37df0'
MEMBER = 'NRC-VAD-Lexicon-v2.1/NRC-VAD-Lexicon-v2.1.txt'


def provision(archive_path, output):
    raw = Path(archive_path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != ARCHIVE_SHA256:
        raise ValueError('NRC v2.1 archive checksum mismatch')
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        data = archive.read(MEMBER)
        readme = archive.read('NRC-VAD-Lexicon-v2.1/README.txt')
    labels = {}
    for row in csv.DictReader(io.StringIO(data.decode('utf-8-sig')), delimiter='\t'):
        word = row['term']
        if word not in REACTION_CLASSES:
            continue
        if word in labels:
            raise ValueError('Duplicate NRC reaction entry')
        values = {axis: float(row[axis]) for axis in ('valence', 'arousal', 'dominance')}
        if any(not math.isfinite(v) or not -1 <= v <= 1 for v in values.values()):
            raise ValueError('Invalid NRC v2.1 coordinates')
        labels[word] = dict(**values, source_word=word, approximation=False)
    if set(labels) != set(REACTION_CLASSES):
        raise ValueError(f'Missing exact NRC terms: {set(REACTION_CLASSES) - set(labels)}')
    asset = {'class_order': REACTION_CLASSES, 'scale': [-1, 1], 'source': {
        'url': URL, 'version': '2.1', 'member': MEMBER,
        'archive_sha256': ARCHIVE_SHA256, 'member_sha256': hashlib.sha256(data).hexdigest(),
        'author': 'Saif M. Mohammad', 'copyright': '2025 National Research Council Canada (NRC)',
        'citation': 'NRC VAD Lexicon v2: Norms for Valence, Arousal, and Dominance for over 55k English Terms. 2025.',
        'terms': 'Non-commercial research/education; do not redistribute source data.'},
        'labels': {c: labels[c] for c in REACTION_CLASSES}}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(asset, indent=2) + '\n').encode()
    path = output / 'reaction_vad.json'
    if path.exists() and path.read_bytes() != payload:
        raise ValueError('Refusing to replace a different existing VAD asset')
    path.write_bytes(payload)
    (output / 'NRC_README.txt').write_bytes(readme)
    return {'asset_sha256': hashlib.sha256(payload).hexdigest(), 'label_count': len(labels),
            'class_order_preserved': True, 'scale': [-1, 1], 'source': asset['source']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(provision(args.archive, args.output), indent=2))
