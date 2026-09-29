"""Separate stable lock inode + atomic replacement for concurrent SLURM jobs."""
import fcntl
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

STATUSES = {'planned', 'submitted', 'running', 'completed', 'failed'}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f'.{path.name}.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def update_registry(path, experiment, **fields):
    if fields.get('status') not in STATUSES:
        raise ValueError('A valid status is required')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(path.suffix + '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        # Corrupt JSON is an error; never silently discard other runs.
        registry = json.loads(path.read_text()) if path.exists() else {'experiments': {}}
        run = registry['experiments'].setdefault(experiment, {})
        # A fast job may start/finish before the submitter records its job ID.
        if fields['status'] == 'submitted' and run.get('status') in {'running','completed','failed'}:
            fields = {k: v for k, v in fields.items() if k != 'status'}
        run.update(fields)
        run['updated_at'] = utc_now()
        atomic_json(path, registry)
        return dict(run)
