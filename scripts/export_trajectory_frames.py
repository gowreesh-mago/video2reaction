"""Copy compact source keyframes for the already-exported trajectories only."""
import argparse
import base64
import csv
import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image


def sha(data):
    return hashlib.sha256(data).hexdigest()


def seconds(value):
    h, m, s = value.split(':')
    return int(h)*3600+int(m)*60+float(s)


def export(source, keyframes, output):
    raw = source.read_bytes()
    payload = json.loads(raw)
    n = len(payload['examples'])
    if n not in (10, 20) or len({e['sample_id'] for e in payload['examples']}) != n:
        raise ValueError('Expected ten or twenty unique selected videos')
    videos = {}
    for example in payload['examples']:
        sid = example['sample_id']
        folder = keyframes/sid
        index = (folder/'index.csv').read_bytes()
        rows = {row['scene_number']: row for row in csv.DictReader(io.StringIO(index.decode()))}
        frames = []
        for number, interval in zip(example['scene_numbers'], example['intervals_seconds']):
            row = rows[number]
            recorded = [seconds(row['start_time']), seconds(row['end_time'])]
            if any(abs(a-b) > 1e-6 for a, b in zip(recorded, interval)):
                raise ValueError(f'Source timestamp mismatch: {sid} / {number}')
            name = f'{int(number)+1:03d}.jpg'
            data = (folder/name).read_bytes()
            with Image.open(io.BytesIO(data)) as image:
                image = image.convert('RGB')
                original_size = list(image.size)
                image.thumbnail((640, 360), Image.Resampling.LANCZOS)
                thumb = io.BytesIO()
                image.save(thumb, format='JPEG', quality=78, optimize=True)
                size = list(image.size)
            encoded = thumb.getvalue()
            frames.append({'scene_number': number, 'interval_seconds': recorded,
                           'source_file': name, 'source_sha256': sha(data),
                           'source_size': original_size, 'thumbnail_size': size,
                           'thumbnail_sha256': sha(encoded),
                           'jpeg_base64': base64.b64encode(encoded).decode()})
        if len(frames) != len(example['vad']):
            raise ValueError('Missing scene keyframes')
        videos[sid] = {'index_sha256': sha(index), 'frames': frames}
    result = {'schema': 1, 'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'trajectory_json_sha256': sha(raw), 'source_root': str(keyframes),
              'interpretation': 'Original dataset scene keyframes, resized without cropping. '
                                'Timestamps are source scene intervals; exact keyframe capture times are not recorded.',
              'thumbnail_max_size': [640, 360], 'jpeg_quality': 78, 'videos': videos}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, separators=(',', ':'), allow_nan=False)+'\n')
    print(json.dumps({'output': str(output), 'videos': len(videos),
                      'frames': sum(len(v['frames']) for v in videos.values()),
                      'bytes': output.stat().st_size, 'sha256': sha(output.read_bytes())}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trajectory', type=Path, required=True)
    parser.add_argument('--keyframes', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    export(args.trajectory, args.keyframes, args.out)
