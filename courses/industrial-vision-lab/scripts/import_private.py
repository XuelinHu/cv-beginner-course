"""Explicit local-only import. Never adds private pixels to the public curriculum."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--expected-count", type=int, default=None, help="Optional human-confirmed count, never inferred from the filename")
    args = parser.parse_args()
    if not args.image.is_file():
        parser.error("Image does not exist")
    if args.expected_count is not None and args.expected_count < 0:
        parser.error("Expected count must be non-negative")
    content = args.image.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    image = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        parser.error("Image is not decodable")
    directory = ROOT / "data/private" / digest[:16]
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / "image.png"
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("PNG encoding failed")
    png = encoded.tobytes()
    if output.exists() and output.read_bytes() != png:
        raise ValueError("An existing local case was changed; preserve it for review")
    output.write_bytes(png)
    metadata = {"private": True, "source_sha256": digest,
                "image_sha256": hashlib.sha256(png).hexdigest(),
                "width": image.shape[1], "height": image.shape[0],
                "expected_count": args.expected_count,
                "count_basis": "user supplied" if args.expected_count is not None else "unknown",
                "slots_truth": "unknown", "public_export_allowed": False,
                "processing": "decoded and re-encoded PNG, original metadata omitted"}
    meta_path = directory / "metadata.json"
    if meta_path.exists():
        current = json.loads(meta_path.read_text(encoding="utf-8"))
        if current != metadata:
            raise ValueError("Existing private metadata differs; not overwriting")
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Imported locally: data/private/{digest[:16]}/image.png (Git-ignored; no public export)")


if __name__ == "__main__":
    main()
