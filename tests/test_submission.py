"""Scheduler mocks verify bounded submissions and dependency handling without SLURM."""
import json

import pytest

from scripts import submit_tier0 as submission


def test_all_preflights_precede_submission_and_retry_does_not_resubmit(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('V2R_ROOT', str(tmp_path))
    monkeypatch.setenv('V2R_REGISTRY', str(tmp_path / 'registry.json'))
    monkeypatch.setenv('USER', 'mock')
    (tmp_path / 'code_version.json').write_text(json.dumps({
        'dirty': False, 'source_sha256': 'a' * 64, 'git_commit': 'b' * 40}))
    events = []
    monkeypatch.setattr(submission.subprocess, 'run', lambda args, check: events.append(('preflight', args)))
    def command(*args):
        events.append(('command', args))
        if args[0] == 'sbatch':
            return str(100 + sum(e[0] == 'command' and e[1][0] == 'sbatch' for e in events))
        return ''
    monkeypatch.setattr(submission, 'command', command)
    submission.submit_batch(['cache', 'model'], 'mock', {'model': 'cache'})
    assert [kind for kind, _ in events[:2]] == ['preflight', 'preflight']
    submits = [args for kind, args in events if kind == 'command' and args[0] == 'sbatch']
    assert len(submits) == 2
    assert not any(arg.startswith('--dependency') for arg in submits[0])
    assert '--dependency=afterok:101' in submits[1]
    submission.submit_batch(['cache', 'model'], 'mock', {'model': 'cache'})
    assert sum(kind == 'command' and args[0] == 'sbatch' for kind, args in events) == 2
    registry = json.loads((tmp_path / 'registry.json').read_text())['experiments']
    assert registry['model_102']['dependency'] == '101'


def test_bad_or_forward_dependencies_fail_before_any_scheduler_command():
    with pytest.raises(ValueError, match='earlier'):
        submission.submit_batch(['model', 'cache'], 'mock', {'model': 'cache'})
    with pytest.raises(ValueError, match='distinct'):
        submission.submit_batch(['model', 'model'], 'mock')
