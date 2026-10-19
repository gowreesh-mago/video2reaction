"""Build an offline Plotly explorer from the compact ten-video trajectory export."""
import argparse
import base64
import hashlib
import json
from pathlib import Path

import numpy as np
from plotly.offline import get_plotlyjs

from export_trajectory_examples import validate_example


def validate_frames(frames, payload, source_hash):
    if frames['trajectory_json_sha256'] != source_hash:
        raise ValueError('Frames belong to a different trajectory export')
    if set(frames['videos']) != {e['sample_id'] for e in payload['examples']}:
        raise ValueError('Frame/video inventory differs')
    for example in payload['examples']:
        rows = frames['videos'][example['sample_id']]['frames']
        if len(rows) != len(example['vad']):
            raise ValueError('Frame/scene counts differ')
        for i, row in enumerate(rows):
            if row['scene_number'] != example['scene_numbers'][i]:
                raise ValueError('Frame scene order differs')
            np.testing.assert_allclose(row['interval_seconds'], example['intervals_seconds'][i], atol=1e-6, rtol=0)
            image = base64.b64decode(row['jpeg_base64'], validate=True)
            if not image.startswith(b'\xff\xd8') or hashlib.sha256(image).hexdigest() != row['thumbnail_sha256']:
                raise ValueError('Invalid or modified thumbnail')


def build(source, output, frames_path=None):
    payload = json.loads(source.read_text())
    if len(payload['examples']) != 10 or len({e['sample_id'] for e in payload['examples']}) != 10:
        raise ValueError('Expected exactly ten unique examples')
    prototypes = np.asarray(payload['prototypes'], dtype=float)
    if prototypes.shape != (21, 3) or not np.isfinite(prototypes).all():
        raise ValueError('Expected 21 finite VAD prototypes')
    for example in payload['examples']:
        distance = validate_example(example, prototypes, payload['temperature'])
        np.testing.assert_allclose(distance, example['distance_squared'], atol=1e-8)
    for key in ['training_mean', 'test_mean', 'example_mean']:
        values = np.asarray(payload[key])
        if values.shape != (21,) or not np.isfinite(values).all() or (values < 0).any():
            raise ValueError('Invalid reference distribution')
        np.testing.assert_allclose(values.sum(), 1, atol=2e-6)
    template = Path(__file__).with_name('trajectory_explorer.html').read_text()
    # Metadata can contain arbitrary text. Do not let it terminate a script element.
    encoded = json.dumps(payload, separators=(',', ':'), allow_nan=False).replace('<', '\\u003c')
    frames = json.loads(frames_path.read_text()) if frames_path else {}
    if frames:
        validate_frames(frames, payload, hashlib.sha256(source.read_bytes()).hexdigest())
    encoded_frames = json.dumps(frames, separators=(',', ':'), allow_nan=False).replace('<', '\\u003c')
    document = (template.replace('__TRAJECTORY_PAYLOAD__', encoded)
                .replace('__FRAME_PAYLOAD__', encoded_frames).replace('__PLOTLY_BUNDLE__', get_plotlyjs()))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document)
    print(json.dumps({'html': str(output), 'bytes': output.stat().st_size,
                      'sha256': hashlib.sha256(output.read_bytes()).hexdigest()}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--frames', type=Path, help='Optional compact keyframe export')
    args = parser.parse_args()
    build(args.input, args.output, args.frames)
