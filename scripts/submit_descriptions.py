"""Submit the description diagnostic and capacity controls after text feature extraction."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.submit_tier0 import submit_batch


PREDICTORS = ['description_only', 'visual_description', 'description_visual_control', 'description_text_control']


if __name__ == '__main__':
    submit_batch(['description_cache'] + PREDICTORS, 'descriptions',
                 {name: 'description_cache' for name in PREDICTORS}, max_parallel=4)
