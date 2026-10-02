"""Submit the two predeclared VAD objectives and their semantic-permutation controls."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.submit_tier0 import submit_batch


if __name__ == '__main__':
    submit_batch(['b_vad_aux', 'b_vad_aux_permuted', 'b_vad_geometry', 'b_vad_geometry_permuted'],
                 'b_vad', max_parallel=4)
