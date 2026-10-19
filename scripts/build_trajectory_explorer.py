"""Build an offline Plotly explorer from the compact ten-video trajectory export."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from plotly.offline import get_plotlyjs

from export_trajectory_examples import validate_example


def build(source, output):
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
    document = template.replace('__TRAJECTORY_PAYLOAD__', encoded).replace('__PLOTLY_BUNDLE__', get_plotlyjs())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document)
    print(json.dumps({'html': str(output), 'bytes': output.stat().st_size,
                      'sha256': hashlib.sha256(output.read_bytes()).hexdigest()}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    build(args.input, args.output)
