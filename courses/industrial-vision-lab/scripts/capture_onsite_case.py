"""Export one explicitly authorized laptop-camera frame as a teaching case.

Uses the existing edge API; never opens or takes over the USB camera.
Full-resolution evidence remains in ignored runtime; only a workpiece crop and
an allowlisted result enter the authorized teaching directory.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from http.cookiejar import CookieJar
import json
from pathlib import Path
import time
from urllib.parse import urljoin, urlsplit
from urllib.request import HTTPCookieProcessor, build_opener, Request

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def checksum(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8019")
    parser.add_argument("--main-root", type=Path, default=ROOT.parents[2] / "battery-weld")
    parser.add_argument("--publish-authorized", action="store_true", required=True,
                        help="Use only after the owner authorizes this current frame for the tutorial")
    args = parser.parse_args()
    manifest = ROOT / "sources/onsite_capture_manifest.json"
    if manifest.exists():
        raise FileExistsError("Preserving the curated capture; review a new capture separately")
    if urlsplit(args.base_url).hostname not in {"127.0.0.1", "localhost"}:
        raise ValueError("Capture must use the current laptop's loopback edge service")
    credentials = {}
    for line in (args.main_root / "frontend/.env.local").read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if separator and key.strip() in {"VITE_LOGIN_USERNAME", "VITE_LOGIN_PASSWORD"}:
            credentials[key.strip()] = value.strip().strip("\"'")
    client = build_opener(HTTPCookieProcessor(CookieJar()))

    def request(path, data=None, *, raw=False):
        # Do not forward the authenticated session to another origin.
        if urlsplit(path).netloc or not path.startswith("/"):
            raise ValueError("Evidence URL must be relative to the authenticated local service")
        body = None if data is None else json.dumps(data).encode("utf-8")
        req = Request(urljoin(args.base_url, path), data=body,
                      headers={"Content-Type": "application/json"} if body else {})
        with client.open(req, timeout=20) as response:
            value = response.read()
        return value if raw else json.loads(value)

    request("/api/auth/login", {"username": credentials["VITE_LOGIN_USERNAME"],
                                "password": credentials["VITE_LOGIN_PASSWORD"]})
    latest = request("/api/edge/latest")
    if not latest.get("stream_valid") or latest.get("frame_age_ms", 99999) > 3000:
        raise RuntimeError("No fresh camera frame: keep the case unpublished")
    queued = request("/api/edge/save", {"frame_id": latest["frame_id"]})
    event_id = queued["event_id"]
    record = None
    for _ in range(40):
        state = request("/api/edge/latest")
        inspection_id = state.get("sync", {}).get("last_inspection_id")
        if inspection_id:
            candidate = request(f"/api/history/{inspection_id}")["record"]
            if candidate.get("result", {}).get("edge_event_id") == event_id:
                record = candidate
                break
        time.sleep(0.5)
    if record is None:
        raise RuntimeError("Saved event has not been acknowledged by the inspection database")
    result = record["result"]
    encoded = request(record["image_url"], raw=True)
    frame = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR)
    if frame is None or not encoded.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("Expected the lossless PNG evidence, not a compressed live preview")
    if checksum(encoded) != result["stored_image_sha256"]:
        raise ValueError("Stored evidence checksum mismatch")
    if checksum(frame.tobytes()) != result["source_frame_sha256"]:
        raise ValueError("Decoded PNG differs from the inspected BGR8 pixels")
    targets = result.get("targets", [])
    if not targets:
        raise ValueError("No workpiece targets available for the authorized crop")
    height, width = frame.shape[:2]
    boxes = np.asarray([t["bbox"] for t in targets], dtype=float)
    x0 = max(0, int(np.floor(boxes[:, 0].min() * width)) - 28)
    y0 = max(0, int(np.floor(boxes[:, 1].min() * height)) - 28)
    x1 = min(width, int(np.ceil((boxes[:, 0] + boxes[:, 2]).max() * width)) + 28)
    y1 = min(height, int(np.ceil((boxes[:, 1] + boxes[:, 3]).max() * height)) + 28)
    crop = frame[y0:y1, x0:x1].copy()
    ok, crop_png = cv2.imencode(".png", crop)
    if not ok:
        raise ValueError("Could not encode the authorized crop")
    case_id = "osmo_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    directory = ROOT / "data/authorized_onsite" / case_id
    directory.mkdir(parents=True, exist_ok=False)
    private = ROOT / "runtime/onsite_evidence" / case_id
    private.mkdir(parents=True, exist_ok=False)
    (private / "source.png").write_bytes(encoded)
    # Credentials, session cookies, complete DB records and private configuration
    # must never be copied into either the manifest or notebook output.
    image_path = directory / "workpieces.png"
    image_path.write_bytes(crop_png.tobytes())
    permitted_targets = []
    for target in targets:
        row = {key: target[key] for key in ("index", "bbox", "status", "quality", "reason") if key in target}
        nickel = target.get("nickel", {})
        row["nickel"] = {key: nickel[key] for key in ("status", "bbox", "reason") if key in nickel}
        weld = target.get("weld", {})
        row["weld"] = {key: weld[key] for key in ("status", "good_count", "bad_count", "unknown_count", "reason") if key in weld}
        row["weld"]["slots7"] = [
            {key: slot[key] for key in ("index", "status", "bbox", "features", "quality", "reason") if key in slot}
            for slot in weld.get("slots7", [])]
        permitted_targets.append(row)
    case = {
        "case_id": case_id, "publication_scope": "user_authorized_onsite",
        "authorization": "Owner explicitly requested the current laptop-camera scene in the existing online tutorial on 2026-11-18",
        "license": "Owner retains rights; permission limited to this project tutorial; no general open-data license",
        "camera": "OsmoAction4 attached to the current Windows laptop",
        "captured_at": result.get("captured_at", latest["captured_at"]),
        "lighting_context": "Owner reports room lights off and one fill light; no measured spectrum or illuminance",
        "image": image_path.relative_to(ROOT).as_posix(), "image_sha256": checksum(crop_png.tobytes()),
        "source_frame_sha256": result["source_frame_sha256"],
        "stored_image_sha256": result["stored_image_sha256"],
        "source_shape_bgr": list(frame.shape), "crop_xywh": [x0, y0, x1-x0, y1-y0],
        "transform": "Lossless BGR8 PNG; workpiece union crop with 28px margin; EXIF not retained",
        "database_evidence": {"acknowledged": True, "png_file_hash_verified": True,
                              "decoded_bgr_hash_verified": True},
        "lighting_requested": {key: result.get("lighting_requested", {}).get(key)
                               for key in ("scene", "lamp_on", "exposure", "autofocus", "auto_exposure")},
        "camera_reported": {key: result.get("camera_reported", {}).get(key)
                            for key in ("width", "height", "fps", "backend", "exposure", "gain", "focus", "white_balance", "controls_supported", "requested_controls_accepted")},
        "live_result": {key: result[key] for key in ("engine", "observed", "counts", "status", "quality") if key in result},
        "targets": permitted_targets,
        "engine_source_sha256": checksum((args.main_root / "backend/classical.py").read_bytes()),
        "ground_truth": {"busbar_count": None, "nickel": "unverified", "seven_slots": "unverified"},
        "limitations": ["Single frozen current-scene frame, not a live public stream or an independent test set",
                        "Detected count is an algorithm output, not manual count truth",
                        "Unknown nickel/slots must remain REVIEW; no weld-strength conclusion"],
    }
    manifest.write_text(json.dumps(case, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"case_id": case_id, "captured_at": case["captured_at"],
                      "crop_xywh": case["crop_xywh"], "live_result": case["live_result"],
                      "database_evidence": case["database_evidence"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
