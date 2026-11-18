"""Checksum-locked, explicitly authorized on-site lesson helpers."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import lab_utils as lab

ROOT = Path(__file__).resolve().parent


def load_case():
    case = json.loads((ROOT / "sources/onsite_capture_manifest.json").read_text(encoding="utf-8"))
    if case.get("publication_scope") != "user_authorized_onsite" or not case.get("authorization"):
        raise ValueError("On-site sample lacks explicit publication authorization")
    relative = Path(case["image"])
    path = (ROOT / relative).resolve()
    if relative.is_absolute() or not path.is_relative_to(ROOT / "data/authorized_onsite"):
        raise ValueError("Authorized sample must stay within its separate teaching directory")
    if path.is_symlink() or lab.file_hash(path) != case["image_sha256"]:
        raise ValueError("Authorized sample checksum mismatch")
    return case, lab.read_image(path)


def replay(image, case):
    source = ROOT.parent / "battery-weld/backend/classical.py"
    if not source.is_file():
        raise FileNotFoundError("第10课须把 https://github.com/XuelinHu/battery-weld 克隆到本项目旁边；前十课不需要")
    if lab.file_hash(source) != case.get("replay_engine_source_sha256", case["engine_source_sha256"]):
        raise ValueError("Main inspection engine changed; review and revalidate this lesson before updating its source lock")
    spec = importlib.util.spec_from_file_location("onsite_classical_engine", source)
    engine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(engine)
    # No private reference images are opened. Unknown nickel and weld slots are
    # intentionally retained, while the actual production contour code runs.
    config = {"assets_dir": str(ROOT / "data/authorized_onsite/no_reference_assets"),
              "working_max_side": 1440, "contour_fallback": True,
              "slot_calibration": {"mode": "strict", "verified": False}}
    if (Path(config["assets_dir"]) / "manifest.json").exists():
        raise ValueError("This lesson must run without private reference assets")
    result = engine.inspect_frame(image, config)
    if result.get("engine") != "classical_v1":
        raise RuntimeError("The main inspection engine could not process this frame")
    return result, {key: value for key, value in config.items() if key != "assets_dir"}


def align_target(image, target):
    polygon = np.float32(target["pose"]["polygon"]) * [image.shape[1], image.shape[0]]
    sides = np.linalg.norm(polygon - np.roll(polygon, -1, axis=0), axis=1)
    width, height = max(2, round(float(max(sides[0], sides[2])))), max(2, round(float(max(sides[1], sides[3]))))
    matrix = cv2.getPerspectiveTransform(polygon.astype(np.float32),
        np.float32([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]]))
    patch = cv2.warpPerspective(image, matrix, (width, height), borderMode=cv2.BORDER_CONSTANT)
    return cv2.rotate(patch, cv2.ROTATE_90_CLOCKWISE) if width > height else patch


def draw_pose(image, result):
    output = image.copy()
    for target in result["targets"]:
        polygon = np.round(np.float32(target["pose"]["polygon"]) * [image.shape[1], image.shape[0]]).astype(np.int32)
        color = {"OK": (55, 175, 68), "NG": (40, 60, 220)}.get(target["status"], (0, 190, 230))
        cv2.polylines(output, [polygon], True, color, 2)
        x, y = polygon.min(axis=0)
        cv2.putText(output, f"#{target['index']}", (max(0,int(x)), max(18,int(y))), cv2.FONT_HERSHEY_SIMPLEX, .65, color, 2)
    return output


def derived_challenges(image, result):
    """Explicit correlated transformations; no additional physical samples."""
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w/2, h/2), 23, 1)
    cosine, sine = abs(matrix[0, 0]), abs(matrix[0, 1])
    nw, nh = int(np.ceil(h*sine+w*cosine)), int(np.ceil(h*cosine+w*sine))
    matrix[:, 2] += [(nw-w)/2, (nh-h)/2]
    rotated = cv2.warpAffine(image, matrix, (nw, nh), borderValue=(20, 35, 60))
    occluded = image.copy()
    x, y, bw, bh = pixel_boxes(result["targets"], image)[0]["bbox"]
    cv2.rectangle(occluded, (max(0,x-5), max(0,y+round(bh*.20))),
                  (min(w-1,x+round(bw*.65)), min(h-1,y+round(bh*.65))), (24, 24, 24), -1)
    return [("原始实拍裁切", image), ("派生：旋转23°，扩画布防裁边", rotated),
            ("派生：黑色遮挡块，未修补像素", occluded),
            ("派生：右侧裁边，仅保留80%视野", image[:, :round(w*.8)].copy()),
            ("派生：完全遮住视野，隐藏数量未知", np.full_like(image, (24, 24, 24)))]


def pixel_boxes(targets, image):
    height, width = image.shape[:2]
    return [{"bbox": [round(t["bbox"][0] * width), round(t["bbox"][1] * height),
                       round(t["bbox"][2] * width), round(t["bbox"][3] * height)],
             "status": t["status"]} for t in targets]


def captured_boxes(case):
    height, width = case["source_shape_bgr"][:2]
    x0, y0, _, _ = case["crop_xywh"]
    return [{"bbox": [round(t["bbox"][0] * width - x0), round(t["bbox"][1] * height - y0),
                       round(t["bbox"][2] * width), round(t["bbox"][3] * height)],
             "status": t["status"]} for t in case["targets"]]
