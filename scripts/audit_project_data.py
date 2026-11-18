"""Create a private SHA-256 inventory for complete local/server data checks."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


DEFAULTS = [
    "data", "outputs", "courses/industrial-vision-lab/data",
    "courses/industrial-vision-lab/outputs",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    files = []
    for relative in DEFAULTS:
        base = root / relative
        if not base.exists():
            continue
        for path in sorted(p for p in base.rglob("*") if p.is_file() and not p.is_symlink()):
            files.append({
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            })
    payload = {"files": files, "file_count": len(files),
               "total_bytes": sum(item["bytes"] for item in files)}
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("file_count", "total_bytes")}, indent=2))


if __name__ == "__main__":
    main()
