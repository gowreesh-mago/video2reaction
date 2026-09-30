"""Freeze tracked, verified source files before SLURM can queue a job."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile


def freeze(source, releases):
    source, releases = Path(source), Path(releases)
    manifest = json.loads((source / 'code_version.json').read_text())
    if manifest['dirty']:
        raise ValueError('Commit and synchronize changes before freezing a release')
    releases.mkdir(parents=True, exist_ok=True)
    destination = releases / f"{manifest['git_commit'][:12]}-{manifest['source_sha256'][:12]}"
    with (releases / 'release.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if destination.exists():
            expected_manifest = json.loads((destination / 'code_version.json').read_text())
            if expected_manifest != manifest:
                raise ValueError('Existing release identity mismatch')
        else:
            temporary = Path(tempfile.mkdtemp(prefix='.release-', dir=releases))
            try:
                for name, expected in manifest['files'].items():
                    raw = (source / name).read_bytes()
                    if hashlib.sha256(raw).hexdigest() != expected:
                        raise ValueError(f'Changed file during release freeze: {name}')
                    target = temporary / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(raw)
                    target.chmod((source / name).stat().st_mode & 0o777)
                (temporary / 'code_version.json').write_text(json.dumps(manifest, indent=2) + '\n')
                (temporary / '.venv').symlink_to(source / '.venv', target_is_directory=True)
                temporary.rename(destination)
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary)
        for name, expected in manifest['files'].items():
            if hashlib.sha256((destination / name).read_bytes()).hexdigest() != expected:
                raise ValueError(f'Frozen release was modified: {name}')
    return destination


if __name__ == '__main__':
    source = Path(__file__).resolve().parents[1]
    print(freeze(source, Path(os.environ['V2R_ROOT']) / 'releases'))
