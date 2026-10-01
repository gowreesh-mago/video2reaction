"""Provision only the published emotion checkpoint, with a pinned checksum."""
import hashlib
import json
import os
from pathlib import Path
import urllib.request


def matches(path, descriptor):
    return (path.exists() and path.stat().st_size == descriptor['checkpoint_bytes']
            and hashlib.sha256(path.read_bytes()).hexdigest() == descriptor['checkpoint_sha256'])


if __name__ == '__main__':
    descriptor = json.loads((Path(__file__).resolve().parents[1] / 'assets/frame_emotion_model.json').read_text())
    path = Path(os.environ['V2R_EMOTION_CHECKPOINT'])
    if path.exists() and not matches(path, descriptor):
        raise ValueError('Existing checkpoint differs from the pinned model; refusing to overwrite')
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + '.download')
        if not matches(temporary, descriptor):
            urllib.request.urlretrieve(descriptor['checkpoint_url'], temporary)
        if not matches(temporary, descriptor):
            raise ValueError('Downloaded emotion checkpoint checksum mismatch')
        temporary.rename(path)
    print(json.dumps({'path': str(path), 'checkpoint_sha256': descriptor['checkpoint_sha256'], 'verified': True}))
