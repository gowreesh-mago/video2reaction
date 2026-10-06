"""Predeclare additional visual-only controls without changing the first batch."""
import json
from pathlib import Path

from scripts.configure_trajectories import sbatch, write_config

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (42, 43, 44)
SHARDS = 4
MAX_PARALLEL = 8


def variants():
    cases = {}

    def add(base, suffix, group, hypothesis, **overrides):
        name = f'on_{base.removeprefix("traj_")}_{suffix}'
        cases[name] = {'base': base + '.yaml', 'experiment': {
            'name': name, 'hypothesis': hypothesis, 'changed_component': group,
            'expected_outcome': 'Compare validation-selected KL and training-defined rare-class diagnostics across three paired seeds.'},
            **overrides}

    for pooling in ('duration', 'peak', 'class_peak', 'soft', 'sparse', 'power', 'sparse_proximity'):
        for value, tag in ((.1, '010'), (.6, '060'), (1.2, '120')):
            add('traj_vad_' + pooling, 'tau' + tag, 'temperature_initialization',
                'Does the fitted VAD decoder depend strongly on its initial distance bandwidth?',
                trajectory={'temperature_init': value})
    for pooling in ('soft', 'sparse', 'sparse_proximity'):
        for value, tag in ((.05, '005'), (.1, '010'), (.5, '050'), (1., '100')):
            add('traj_vad_' + pooling, 'gate' + tag, 'relevance_temperature',
                'Does the strength of moment selection change predictive performance and effective temporal support?',
                trajectory={'relevance_temperature': value})
    for base in ('traj_vad_duration', 'traj_vad_peak', 'traj_vad_class_peak', 'traj_vad_soft',
                 'traj_vad_power', 'traj_vad_sparse_proximity', 'traj_free_soft', 'traj_free_sparse'):
        add(base, 'notime', 'temporal_position',
            'Does temporal position help with the same contextual model, duration measure and pooling rule?',
            model={'positional_encoding': False})
    for base in ('traj_vad_duration', 'traj_vad_soft', 'traj_vad_sparse', 'traj_free_soft',
                 'traj_free_sparse', 'dino_meanpool'):
        add(base, 'wide', 'hidden_width',
            'Is performance limited by the 128-dimensional visual hidden representation?', model={'hidden_dim': 256})
    for pooling in ('duration', 'sparse'):
        for exponent, tag in ((.25, '025'), (1., '100')):
            for corrected in (False, True):
                suffix = f'rare{tag}' + ('_ipw' if corrected else '')
                add('traj_vad_' + pooling, suffix, 'rare_sampling_strength',
                    'Does stronger training-only rare sampling help, and does importance correction remove the gain?',
                    training={'sampling': {'exponent': exponent, 'cap': 3., 'importance_corrected': corrected}})
    for pooling in ('soft', 'sparse'):
        for corrected in (False, True):
            add('traj_free_' + pooling, 'rare' + ('_ipw' if corrected else ''), 'free_decoder_rare_sampling',
                'Is rare-emotion sampling useful without the fixed VAD bottleneck?',
                training={'sampling': {'exponent': .5, 'cap': 3., 'importance_corrected': corrected}})
    for pooling in ('duration', 'power'):
        add('traj_vad_' + pooling, 'free', 'unrestricted_local_decoder',
            'Does an unrestricted local classifier improve the same duration or power probability mixture?',
            trajectory={'decoder': 'free'})
    for pooling in ('duration', 'soft', 'sparse'):
        for permutation in (314159, 161803):
            add('traj_vad_' + pooling, f'perm{permutation}', 'semantic_assignment',
                'Do sourced VAD assignments outperform multiple arbitrary label-to-coordinate assignments?',
                trajectory={'prototypes': {'mapping': 'permuted', 'permutation_seed': permutation}})
    for base in ('traj_vad_duration', 'traj_vad_soft', 'traj_vad_sparse', 'traj_free_soft', 'traj_free_sparse'):
        add(base, 'shallow', 'context_depth',
            'Does a single contextual layer generalize better than two with the same pooling and decoder?',
            model={'layers': 1})
    if len(cases) != 72:
        raise AssertionError('Expected 72 additional distinct conditions')
    return cases


def generate():
    cases = variants()
    names = list(cases)
    for name, cfg in cases.items():
        write_config(name, cfg)
        for seed in SEEDS:
            run = f'{name}_s{seed}'
            write_config(run, {'base': name + '.yaml', 'seed': seed, 'experiment': {'name': run}})
            script = sbatch(run).replace('00:45:00', '00:20:00')
            (ROOT / f'slurm/{run}.sbatch').write_text(script)
    manifest = {'schema': 1, 'batch': 'trajectory_overnight', 'variants': names, 'seeds': list(SEEDS),
                'smoke_shards': SHARDS, 'smoke_epochs': 20, 'max_parallel': MAX_PARALLEL,
                'time_limit_minutes': 20, 'predictor_count': len(names) * len(SEEDS),
                'groups': {n: c['experiment']['changed_component'] for n, c in cases.items()}}
    (ROOT / 'configs/trajectory_overnight.json').write_text(json.dumps(manifest, indent=2) + '\n')
    template = (ROOT / 'slurm/smoke_trajectory_3videos.sbatch').read_text()
    for shard in range(SHARDS):
        script = template.replace('v2r-smoke-trajectory', f'v2r-smoke-overnight-{shard}')
        script = script.replace('smoke_trajectory_3videos_', 'smoke_trajectory_overnight_3videos_')
        script = script.replace('--run-name "$EXP"',
            f'--run-name "$EXP" --batch overnight --shard-index {shard} --shard-count {SHARDS} --epochs 20')
        (ROOT / f'slurm/smoke_trajectory_overnight_{shard}.sbatch').write_text(script)
    print(f'Configured {len(names)} conditions x {len(SEEDS)} seeds = {manifest["predictor_count"]} additional runs.')


if __name__ == '__main__':
    generate()
