"""Remove only byte-identical duplicates misplaced by a multi-dir SCP copy."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--expected", type=Path, required=True)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    expected = {row["path"]: row["sha256"] for row in json.loads(args.expected.read_text(encoding="utf-8"))["files"]}
    current = json.loads(args.current.read_text(encoding="utf-8"))["files"]
    extras = [row for row in current if row["path"] not in expected]
    removable = []
    for row in extras:
        path = row["path"]
        if path.startswith("data/"):
            canonical = "courses/industrial-vision-lab/data/" + path.removeprefix("data/")
        elif path.startswith("outputs/"):
            canonical = "courses/industrial-vision-lab/outputs/" + path.removeprefix("outputs/")
        else:
            raise RuntimeError(f"Unexpected extra outside transfer targets: {path}")
        target = root / canonical
        duplicate = root / path
        if expected.get(canonical) != row["sha256"] or not target.is_file() or sha256(target) != row["sha256"]:
            raise RuntimeError(f"Refusing to remove unverified duplicate: {path}")
        if sha256(duplicate) != row["sha256"]:
            raise RuntimeError(f"Refusing changed extra file: {path}")
        removable.append(duplicate)
    print(f"Verified duplicate files: {len(removable)}; apply={args.apply}")
    if args.apply:
        for path in removable:
            path.unlink()


if __name__ == "__main__":
    main()
