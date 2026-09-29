"""Download only the official metadata, pinned to a dataset revision."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

REPO = "infofusionlab/Video2Reaction"
REVISION = "578d1423f89f1a7b52471b01e78770e08f0a226c"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {"repository": REPO, "revision": REVISION, "files": {}}
    for split in ("train", "val", "test"):
        url = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{split}.json"
        path = args.output / f"{split}.json"
        data = urllib.request.urlopen(url, timeout=120).read()
        parsed = json.loads(data)
        if path.exists() and path.read_bytes() != data:
            raise RuntimeError(f"Refusing to overwrite different metadata: {path}")
        path.write_bytes(data)
        manifest["files"][split] = {"url": url, "sha256": hashlib.sha256(data).hexdigest(), "size": len(parsed)}
        print(split, len(parsed), "fields:", list(next(iter(parsed.values()))), flush=True)
    (args.output / "download_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
