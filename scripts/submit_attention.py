"""Submit only the two reviewed attention comparisons, using the completed visual cache."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.submit_tier0 import submit_batch


if __name__ == '__main__':
    submit_batch(['e_reaction_query', 'e_shared_query_control'], 'e_attention')
