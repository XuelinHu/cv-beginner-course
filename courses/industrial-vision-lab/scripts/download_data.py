"""Download a small, auditable public teaching subset; never use private images."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
import urllib.request
import zipfile

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REVISION = "8b80d1afe88c0b946500a498cbdb9d7a1eda5533"
HF = f"https://huggingface.co/datasets/rikkarth/welding-defect-object-detection/resolve/{REVISION}"
WELDS = [
    ("weld_good", "Good-Welding-images_11_jpeg.rf.4807e1282a383cba87b1e2e3c77ca9ce.jpg",
     "a57d4e8615c3c87e0cd6b522fd84a21e4e39e55265907b8cd49589a7a4160f1b"),
    ("weld_crack", "1aa4b44c-d01b8a11-Hot-crack_jpg.rf.57393ca7936f061ca5991481b2b5c485.jpg",
     "4ae59bd790ea6ae4bea66fc6268c512b2510128e638c648890220765a10ecf09"),
    ("weld_other", "02fd0af7-51ab46bb-c13_jpg.rf.7de08cd3b264d9098d09f5fe6d21c959.jpg",
     "dca734f3287f823ab85fbfb3c85a4a04b4acb43046b4dbc39af53cafc8e85f47"),
]
SPECS = [{
    "id": "kolektor_archive", "url": "https://data.vicos.si/datasets/KSDD/KolektorSDD.zip",
    "file": "data/cache/KolektorSDD.zip", "kind": "archive", "dataset": "kolektor",
    "homepage": "https://www.vicos.si/resources/kolektorsdd/", "license": "CC-BY-NC-SA-4.0",
    "license_url": "https://creativecommons.org/licenses/by-nc-sa/4.0/",
    "author": "Kolektor Group d.o.o.; ViCoS Laboratory", "revision": "official unversioned archive; pinned by SHA256",
    "citation": "Tabernik et al. 2019, doi:10.1007/s10845-019-01476-x",
}, {
    "id": "parts_separated", "url": "https://upload.wikimedia.org/wikipedia/commons/c/ce/Screws.png",
    "file": "data/public/parts/separated.png", "kind": "image", "dataset": "parts",
    "homepage": "https://commons.wikimedia.org/w/index.php?title=File:Screws.png&oldid=504053334",
    "license": "CC-BY-SA-3.0", "license_url": "https://creativecommons.org/licenses/by-sa/3.0/",
    "author": "Tung1213", "revision": "2012-11-12T22:50:00Z original file",
    "description": "Real photograph: six screws and two metal balls, not an industrial benchmark.",
}, {
    "id": "parts_occluded", "url": "https://upload.wikimedia.org/wikipedia/commons/8/8f/Schrauben.jpg",
    "file": "data/public/parts/occluded.jpg", "kind": "image", "dataset": "parts",
    "homepage": "https://commons.wikimedia.org/w/index.php?title=File:Schrauben.jpg&oldid=927807040",
    "license": "CC-BY-SA-4.0", "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
    "author": "Technologyfelix", "revision": "2023-04-15T12:40:00Z original file",
    "description": "Real photograph: overlapping hex screws in a box; complete count is unknown.",
}]
for case, name, digest in WELDS:
    for kind, folder, ext in [("image", "images", ".jpg"), ("annotation", "labels", ".txt")]:
        filename = Path(name).with_suffix(ext).name
        SPECS.append({
            "id": case + ("_label" if kind == "annotation" else ""),
            "url": f"{HF}/test/{folder}/{filename}",
            "file": f"data/public/weld/{case}{ext}", "kind": kind, "dataset": "weld",
            "homepage": "https://huggingface.co/datasets/rikkarth/welding-defect-object-detection",
            "original_homepage": "https://www.kaggle.com/datasets/sukmaadhiwijaya/welding-defect-object-detection",
            "license": "CC0-1.0", "license_url": "https://creativecommons.org/publicdomain/zero/1.0/",
            "author": "sukmaadhiwijaya (Kaggle); rikkarth (pinned distribution / COCO conversion)",
            "revision": REVISION, "declared_sha256": digest if kind == "image" else None,
            "source_split": "test", "use": "teaching only; never a newly claimed test set",
        })


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def download(spec: dict, expected: str | None) -> dict:
    target = ROOT / spec["file"]
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        actual = sha256(target)
        if expected and actual != expected:
            raise ValueError(f"Existing file checksum mismatch: {spec['file']}; preserve and inspect it")
        if expected or target.stat().st_size:
            return {**spec, "sha256": actual, "bytes": target.stat().st_size}
    partial = target.with_name(target.name + ".part")
    for attempt in range(3):
        try:
            request = urllib.request.Request(spec["url"], headers={"User-Agent": "industrial-vision-lab/0.1 educational-reproducibility"})
            with urllib.request.urlopen(request, timeout=45) as response, partial.open("wb") as stream:
                downloaded = 0
                for chunk in iter(lambda: response.read(1024 * 1024), b""):
                    downloaded += len(chunk)
                    if downloaded > 180 * 1024 * 1024:
                        raise ValueError("Download exceeded the teaching subset archive size limit")
                    stream.write(chunk)
            actual = sha256(partial)
            if expected and actual != expected:
                raise ValueError(f"Downloaded file checksum mismatch: {spec['id']}")
            partial.replace(target)
            print(f"Downloaded {spec['id']} ({target.stat().st_size} bytes)", flush=True)
            return {**spec, "sha256": actual, "bytes": target.stat().st_size}
        except (OSError, TimeoutError) as error:
            if attempt == 2:
                raise RuntimeError(f"Download failed for {spec['id']}: {type(error).__name__}") from error
            time.sleep(1 + attempt)
    raise RuntimeError("Unreachable")


def select_kolektor(archive: Path, source: dict) -> list[dict]:
    """Preserve exact bytes of three normal and three defective image/mask pairs."""
    rows = []
    counts = {False: 0, True: 0}
    with zipfile.ZipFile(archive) as bundle:
        names = set(bundle.namelist())
        for name in sorted(names):
            if not name.lower().endswith("_label.bmp"):
                continue
            image_name = name[:-10] + ".jpg"
            if image_name not in names:
                image_name = name[:-10] + ".bmp"
            if image_name not in names:
                continue
            mask_bytes = bundle.read(name)
            mask = cv2.imdecode(np.frombuffer(mask_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise ValueError(f"Unreadable mask: {name}")
            defective = bool(np.any(mask > 0))
            if counts[defective] >= 3:
                continue
            idx = counts[defective]
            label = "defect" if defective else "normal"
            for kind, member in [("image", image_name), ("mask", name)]:
                output = ROOT / f"data/public/kolektor/{label}_{idx}{'_mask' if kind == 'mask' else ''}{Path(member).suffix.lower()}"
                output.parent.mkdir(parents=True, exist_ok=True)
                content = bundle.read(member)
                if output.exists() and output.read_bytes() != content:
                    raise ValueError(f"Refusing to overwrite changed sample: {output.name}")
                output.write_bytes(content)
                rows.append({
                    **{key: source[key] for key in ("homepage", "license", "license_url", "author", "citation")},
                    "id": f"kolektor_{label}_{idx}" + ("_mask" if kind == "mask" else ""),
                    "dataset": "kolektor", "kind": kind, "file": output.relative_to(ROOT).as_posix(),
                    "archive_member": member, "archive_sha256": source["sha256"],
                    "sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content),
                    "defective": defective, "source_group": str(Path(member).parent),
                    "split": "teaching only; no train/val/test accuracy claim",
                })
            counts[defective] += 1
            if min(counts.values()) == 3:
                break
    if min(counts.values()) != 3:
        raise ValueError(f"Expected three positive and negative pairs, found {counts}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initialize-lock", action="store_true", help="Only for initial curator capture; subsequent runs require the committed SHA256 lock")
    args = parser.parse_args()
    lock_path = ROOT / "sources/source_manifest.json"
    existing = json.loads(lock_path.read_text(encoding="utf-8")) if lock_path.exists() else {}
    locked = {row["id"]: row for row in existing.get("files", [])}
    if not locked and not args.initialize_lock:
        parser.error("Missing source lock. Obtain the repository source_manifest.json; do not silently trust new data.")
    public_locked = [row for row in locked.values() if row["file"].startswith("data/public/")]
    if public_locked and all((ROOT / row["file"]).is_file() for row in public_locked):
        for row in public_locked:
            if sha256(ROOT / row["file"]) != row["sha256"]:
                raise ValueError(f"Existing public sample checksum mismatch: {row['id']}")
        print(f"Verified {len(public_locked)} cached public files; no network needed.")
        return
    results = []
    for spec in SPECS:
        expected = locked.get(spec["id"], {}).get("sha256") or spec.get("declared_sha256")
        if not expected and not args.initialize_lock:
            raise ValueError(f"Missing checksum for {spec['id']}")
        results.append(download(spec, expected))
    results += select_kolektor(ROOT / SPECS[0]["file"], results[0])
    payload = {"schema_version": 1, "curated_on": "2026-11-17", "private_images_included": False,
               "purpose": "Chinese non-commercial teaching subset; not validation of a battery weld product",
               "files": results}
    if locked:
        for row in results:
            if row["id"] in locked and row["sha256"] != locked[row["id"]]["sha256"]:
                raise ValueError(f"Sample changed: {row['id']}")
    if args.initialize_lock:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Verified {len(results)} public files. Sources: weld / Kolektor surface / real parts.")


if __name__ == "__main__":
    main()
