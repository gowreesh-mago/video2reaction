"""Write the predeclared visual-only batch; does not submit or run experiments."""
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (42, 43, 44)
VARIANTS = {
    'traj_vad_duration': ('duration', 'vad', 'Time spent near each emotion predicts reaction probability.'),
    'traj_vad_peak': ('peak', 'vad', 'The single closest approach to any emotion explains the reaction distribution.'),
    'traj_vad_class_peak': ('class_peak', 'vad', 'Each emotion receives evidence from its own closest moment.'),
    'traj_vad_soft': ('soft', 'vad', 'Learned visual relevance improves on duration-only integration.'),
    'traj_vad_sparse': ('sparse', 'vad', 'Suppressing irrelevant moments improves over soft visual relevance.'),
    'traj_vad_power': ('power', 'vad', 'Adaptive probability power pooling improves on mean or peak pooling.'),
    'traj_vad_sparse_proximity': ('sparse_proximity', 'vad', 'Absolute proximity adds information beyond local normalized probabilities.'),
    'traj_vad_sparse_permuted': ('sparse', 'vad', 'Semantic VAD assignments help beyond a fixed three-dimensional bottleneck.'),
    'traj_vad_duration_rare': ('duration', 'vad', 'Rare-reaction sampling improves rare-class coverage with duration pooling.'),
    'traj_vad_sparse_rare': ('sparse', 'vad', 'Rare-reaction sampling and sparse relevance have complementary effects.'),
    'traj_vad_duration_importance': ('duration', 'vad', 'Importance correction separates rare sampling from changing the objective.'),
    'traj_vad_sparse_importance': ('sparse', 'vad', 'Importance correction separates rare sampling from changing the objective.'),
    'traj_free_soft': ('soft', 'free', 'An unrestricted local classifier tests the VAD bottleneck under soft pooling.'),
    'traj_free_sparse': ('sparse', 'free', 'An unrestricted local classifier tests the VAD bottleneck under sparse pooling.'),
    'traj_vad_sparse_no_time': ('sparse', 'vad', 'Timestamp-free contextualization tests the value of temporal position.'),
    'dino_meanpool': (None, None, 'A matched purely visual backbone baseline separates encoder and pooling changes.'),
}


def write_config(name, data):
    (ROOT / f'configs/experiments/{name}.yaml').write_text(yaml.safe_dump(data, sort_keys=False))


def sbatch(name, cache=False):
    return f'''#!/usr/bin/env bash
#SBATCH --job-name=v2r-{name}
#SBATCH --account=gusr133332
#SBATCH --partition={'gpu_a100' if cache else 'gpu_mig'}
#SBATCH --gpus=1
#SBATCH --cpus-per-task={8 if cache else 4}
#SBATCH --mem={48 if cache else 24}G
#SBATCH --time={'08:00:00' if cache else '00:45:00'}
#SBATCH --signal=B:TERM@30
set -euo pipefail
bash scripts/run_experiment.sh {name}
'''


def generate():
    write_config('dino_common', {'base': 'tier0_common.yaml', 'encoder': {
        'kind': 'dinov2', 'model': 'facebook/dinov2-base',
        'revision': 'f9e44c814b77203eaa57a6bdbbd535f21ede1415', 'local_files_only': True,
        'use_fast': False, 'batch_size': 64, 'workers': 6, 'feature_dim': 768},
        'model': {'input_dim': 768}, 'training': {'learning_rate': .0003},
        'evaluation': {'save_attention': False}})
    write_config('trajectory_common', {'base': 'dino_common.yaml',
        'model': {'aggregation': 'trajectory'}, 'trajectory': {
            'pooling': 'sparse', 'decoder': 'vad', 'relevance_temperature': .2, 'temperature_init': .25,
            'prototypes': {'asset_sha256': 'fb2ce00a38d37e6e8b41418a09cb374ad96b51bc9c24c32f0a51bb790a351906',
                           'mapping': 'sourced', 'permutation_seed': 271828}},
        'evaluation': {'save_attention': True}})
    write_config('dino_cache', {'base': 'dino_common.yaml', 'experiment': {
        'name': 'dino_cache', 'hypothesis': 'Purely visual self-supervised features for all official keyframes.',
        'changed_component': 'Frozen DINOv2 CLS features; no language model or text features.',
        'expected_outcome': 'Complete verified cache, not a benchmark result.'}})
    (ROOT / 'slurm/dino_cache.sbatch').write_text(sbatch('dino_cache', cache=True))
    for name, (pooling, decoder, hypothesis) in VARIANTS.items():
        cfg = {'base': 'trajectory_common.yaml' if pooling else 'dino_common.yaml', 'experiment': {
            'name': name, 'hypothesis': hypothesis, 'changed_component': name,
            'expected_outcome': 'Compare validation-selected official KL and rare-class diagnostics across paired seeds.'}}
        if pooling:
            cfg['trajectory'] = {'pooling': pooling, 'decoder': decoder}
        if name.endswith('_permuted'):
            cfg['trajectory']['prototypes'] = {'mapping': 'permuted'}
        if name.endswith('_no_time'):
            cfg['model'] = {'positional_encoding': False}
        if name.endswith(('_rare', '_importance')):
            cfg['training'] = {'sampling': {'exponent': .5, 'cap': 3., 'importance_corrected': name.endswith('_importance')}}
        write_config(name, cfg)
        for seed in SEEDS:
            run = f'{name}_s{seed}'
            write_config(run, {'base': name + '.yaml', 'seed': seed, 'experiment': {'name': run}})
            (ROOT / f'slurm/{run}.sbatch').write_text(sbatch(run))
    write_config('smoke_trajectory_3videos', {'base': 'trajectory_common.yaml', 'experiment': {
        'name': 'smoke_trajectory_3videos', 'hypothesis': 'Verify every production variant on three training videos.',
        'changed_component': 'Infrastructure smoke only', 'expected_outcome': 'Finite outputs, lower fit loss and exact reload/resume.'},
        'data': {'max_frames': 8, 'smoke_video_ids': ['4nSkJZ3i2-g', 'eiqBbLVXbQg', 'iKp5ARBBpyc']},
        'encoder': {'workers': 0}, 'training': {'epochs': 40, 'patience': 40, 'batch_size': 3, 'learning_rate': .0003}})


if __name__ == '__main__':
    generate()
    print(f'Configured {len(VARIANTS)} variants x {len(SEEDS)} paired seeds, plus cache and smoke.')
