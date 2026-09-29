"""Read original metadata and chronological keyframes without changing splits."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
from .taxonomy import REACTION_CLASSES


def load_metadata(directory, split):
    path = Path(directory) / f'{split}.json'
    raw = path.read_bytes()
    rows = json.loads(raw)
    if not isinstance(rows, dict) or not rows:
        raise ValueError(f'Expected nonempty video-ID mapping: {path}')
    return rows, hashlib.sha256(raw).hexdigest()


def target_distribution(row):
    values = row['reaction_outcome']['reaction_distribution']
    if set(values) - set(REACTION_CLASSES):
        raise ValueError('Unknown reaction labels')
    target = np.array([values.get(c, 0.) for c in REACTION_CLASSES], dtype=np.float32)
    if not np.isfinite(target).all() or (target < 0).any() or not np.isclose(target.sum(), 1., atol=1e-5):
        raise ValueError('Target must be a finite, normalized distribution')
    return target


def frame_index(root, video_id, max_frames=None):
    folder = Path(root) / video_id
    with (folder / 'index.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f'Empty frame index: {video_id}')
    starts = [int(r['start_frame']) for r in rows]
    if any(a >= b for a, b in zip(starts, starts[1:])):
        raise ValueError(f'Frame index is not chronological: {video_id}')
    chosen = np.arange(len(rows))
    if max_frames and len(rows) > max_frames:
        chosen = np.linspace(0, len(rows)-1, max_frames, dtype=int)
    selected = []
    for i in chosen:
        row = dict(rows[i])
        row['index'] = int(i)
        row['path'] = str(folder / f"{int(row['scene_number'])+1:03d}.jpg")
        if not Path(row['path']).is_file():
            raise FileNotFoundError(row['path'])
        selected.append(row)
    return selected, len(rows)


def read_images(rows):
    images = []
    for row in rows:
        with Image.open(row['path']) as image:
            images.append(image.convert('RGB'))
    return images
