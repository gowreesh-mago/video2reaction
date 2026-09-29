"""Capture synchronized code identity without copying the Git database."""
import hashlib
import json
from pathlib import Path
import subprocess

root=Path(__file__).resolve().parents[1]
files=subprocess.check_output(['git','ls-files','-z'],cwd=root).decode().split('\0')
hashes={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in files if p and (root/p).is_file()}
manifest=dict(git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
              git_branch=subprocess.check_output(['git','branch','--show-current'],cwd=root,text=True).strip(),
              dirty=bool(subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True).strip()),
              files=hashes,source_sha256=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest())
(root/'code_version.json').write_text(json.dumps(manifest,indent=2)+'\n')
