"""Provision pinned public DSNet weights and the four reviewed inference modules."""
import hashlib
import json
import os
from pathlib import Path
import urllib.request
import zipfile


def verify(path, expected):
    return path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == expected


def download(path, url, expected):
    if path.exists():
        if not verify(path, expected):
            raise ValueError(f'Existing model asset checksum differs: {path}')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.partial')
    if not verify(temp, expected):
        urllib.request.urlretrieve(url, temp)
    if not verify(temp, expected):
        raise ValueError(f'Download checksum mismatch: {path}')
    temp.rename(path)


def extract(archive, member, path, expected):
    if path.exists():
        if not verify(path, expected):
            raise ValueError(f'Existing source or checkpoint changed: {path}')
        return
    with zipfile.ZipFile(archive) as z:
        raw = z.read(member)
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError(f'Archive member checksum mismatch: {member}')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)


if __name__ == '__main__':
    spec = json.loads((Path(__file__).resolve().parents[1] / 'assets/highlight_model.json').read_text())
    root = Path(os.environ['V2R_HIGHLIGHT_MODEL_DIR'])
    download(root/'source.zip', spec['source_url'], spec['source_archive_sha256'])
    download(root/'pretrain_af_basic.zip', spec['checkpoint_archive_url'], spec['checkpoint_archive_sha256'])
    download(root/'googlenet.pth', spec['backbone_url'], spec['backbone_sha256'])
    extract(root/'pretrain_af_basic.zip', spec['checkpoint_member'], root/'tvsum0.pt', spec['checkpoint_sha256'])
    for name, sha in spec['source_files'].items():
        extract(root/'source.zip', f"DSNet-{spec['revision']}/{name}", root/'source'/name, sha)
    print(json.dumps({'path': str(root), 'verified': True, 'model': spec['name'], 'checkpoint_sha256': spec['checkpoint_sha256']}))
