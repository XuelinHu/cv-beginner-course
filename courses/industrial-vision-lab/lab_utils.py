"""Small transparent helpers shared by the executable Chinese lessons."""
from __future__ import annotations

import base64
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import sqlite3
from typing import Iterable

import cv2
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "outputs/public"
OUTPUT.mkdir(parents=True, exist_ok=True)
matplotlib.rcParams["axes.unicode_minus"] = False
available_fonts = {font.name for font in font_manager.fontManager.ttflist}
for preferred in ("Microsoft YaHei", "Noto Sans CJK SC", "SimHei", "DejaVu Sans"):
    if preferred in available_fonts:
        matplotlib.rcParams["font.family"] = preferred
        break


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def manifest() -> dict:
    path = ROOT / "sources/source_manifest.json"
    if not path.is_file():
        raise FileNotFoundError("缺少公开数据来源清单，请先运行 python scripts/download_data.py")
    return json.loads(path.read_text(encoding="utf-8"))


def case_row(case_id: str) -> dict:
    for row in manifest()["files"]:
        if row["id"] == case_id:
            return row
    raise KeyError(f"找不到公开案例：{case_id}")


def read_image(path: Path, grayscale: bool = False) -> np.ndarray:
    data = np.fromfile(path, np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"图像无法读取：{path.name}")
    return image


def load_case(case_id: str, grayscale: bool = False) -> np.ndarray:
    row = case_row(case_id)
    path = ROOT / row["file"]
    if not path.is_file():
        raise FileNotFoundError(f"缺少公开案例 {case_id}，请运行 python scripts/download_data.py")
    if file_hash(path) != row["sha256"]:
        raise ValueError(f"公开案例 {case_id} 的 SHA256 不符，请检查是否被编辑")
    return read_image(path, grayscale)


def gray(image: np.ndarray) -> np.ndarray:
    return image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def show_images(images: Iterable[np.ndarray], titles: Iterable[str], *, cols: int = 3,
                figsize: tuple[float, float] | None = None) -> None:
    images, titles = list(images), list(titles)
    if not images or len(images) != len(titles):
        raise ValueError("图片与标题须非空且一一对应")
    cols = min(cols, len(images))
    rows = (len(images) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=figsize or (5 * cols, 3.9 * rows), squeeze=False)
    for axis, image, title in zip(axes.flat, images, titles):
        axis.imshow(image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                    cmap="gray" if image.ndim == 2 else None, vmin=0, vmax=255)
        axis.set_title(title, fontsize=13)
        axis.axis("off")
    for axis in list(axes.flat)[len(images):]:
        axis.axis("off")
    fig.tight_layout()
    plt.show()
    plt.close(fig)


def image_quality(image: np.ndarray) -> dict:
    g = gray(image)
    return {
        "brightness": round(float(g.mean()), 3),
        "sharpness": round(float(cv2.Laplacian(g, cv2.CV_64F).var()), 3),
        "dark_ratio": round(float(np.mean(g < 20)), 5),
        "overexposed_ratio": round(float(np.mean(g > 245)), 5),
        "width": g.shape[1], "height": g.shape[0],
    }


def degrade(image: np.ndarray, factor: float = 0.3, *, noise: float = 0,
            blur: int = 0, seed: int = 17) -> np.ndarray:
    """Controlled derived challenge, never a measured night-light experiment."""
    result = image.astype(np.float32) * factor
    if noise:
        result += np.random.default_rng(seed).normal(0, noise, image.shape)
    result = np.clip(result, 0, 255).astype(np.uint8)
    if blur:
        if blur < 1 or blur % 2 == 0:
            raise ValueError("高斯模糊核须为正奇数")
        result = cv2.GaussianBlur(result, (blur, blur), 0)
    return result


def component_boxes(binary: np.ndarray, min_area: int = 30, max_area: int | None = None) -> list[dict]:
    """Count image-connected regions, which are not automatically physical objects."""
    if binary.ndim != 2 or binary.dtype != np.uint8:
        raise ValueError("连通域输入须为 uint8 单通道二值图")
    count, _, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    boxes = []
    for idx in range(1, count):
        x, y, width, height, area = map(int, stats[idx])
        if area < min_area or (max_area is not None and area > max_area):
            continue
        boxes.append({"bbox": [x, y, width, height], "area": area,
                      "center": centroids[idx].tolist(), "status": "REVIEW"})
    return sorted(boxes, key=lambda row: (round(row["center"][1] / 40), row["center"][0]))


def draw_boxes(image: np.ndarray, boxes: list[dict], *, thickness: int = 2) -> np.ndarray:
    result = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()
    colors = {"OK": (68, 175, 55), "NG": (40, 60, 220), "REVIEW": (0, 190, 230)}
    for idx, row in enumerate(boxes, 1):
        x, y, width, height = map(int, row["bbox"])
        color = colors.get(row.get("status"), colors["REVIEW"])
        cv2.rectangle(result, (x, y), (x + width, y + height), color, thickness)
        cv2.putText(result, f"#{idx}", (x, max(15, y - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return result


def pixel_yolo_labels(case_id: str, image: np.ndarray) -> list[dict]:
    row = case_row(case_id + "_label")
    path = ROOT / row["file"]
    if file_hash(path) != row["sha256"]:
        raise ValueError("标注文件校验失败")
    height, width = image.shape[:2]
    names = {0: "Bad Weld", 1: "Good Weld", 2: "Defect"}
    boxes = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        category, cx, cy, w, h = map(float, line.split())
        boxes.append({"category_id": int(category), "label": names[int(category)],
                      "bbox": [round((cx - w / 2) * width), round((cy - h / 2) * height),
                               round(w * width), round(h * height)],
                      "status": "REVIEW", "source": "upstream_annotation"})
    return boxes


def local_std(gray_image: np.ndarray, window: int = 11) -> np.ndarray:
    source = gray(gray_image).astype(np.float32)
    mean = cv2.blur(source, (window, window))
    variance = cv2.blur(source * source, (window, window)) - mean * mean
    return np.sqrt(np.maximum(variance, 0))


def suppress_matches(scores: np.ndarray, template_shape: tuple[int, int], *, threshold: float = 0.85,
                     max_matches: int = 40) -> list[dict]:
    """Greedy spatial suppression for a fixed-size template; not detector confidence."""
    heatmap = scores.copy()
    height, width = template_shape
    matches = []
    for _ in range(max_matches):
        _, value, _, point = cv2.minMaxLoc(heatmap)
        if not np.isfinite(value) or value < threshold:
            break
        x, y = point
        matches.append({"bbox": [x, y, width, height], "score": float(value), "status": "REVIEW"})
        left, top = max(0, x - width // 2), max(0, y - height // 2)
        right, bottom = min(heatmap.shape[1], x + width // 2 + 1), min(heatmap.shape[0], y + height // 2 + 1)
        heatmap[top:bottom, left:right] = -1
    return matches


def seven_slot_verdict(slots: list[str | None], *, busbar_visible: bool = True,
                       nickel_present: bool | None = True, quality_clear: bool = True) -> dict:
    """Conservative visible-appearance rule; cannot establish fusion or strength."""
    good = sum(status == "OK" for status in slots)
    base = {"good_count": good, "required_good": 5, "total_slots": 7, "slots": slots}
    if not busbar_visible or not quality_clear:
        return {**base, "status": "REVIEW", "reason": "工件不完整或画质不满足判读条件"}
    if nickel_present is None:
        return {**base, "status": "REVIEW", "reason": "镍片存在性未知"}
    if nickel_present is False:
        return {**base, "status": "NG", "reason": "已确认缺少镍片"}
    if len(slots) != 7 or any(value not in ("OK", "NG") for value in slots):
        return {**base, "status": "REVIEW", "reason": "七格未全部清晰判读，未知不能自动填为合格"}
    return {**base, "status": "OK" if good >= 5 else "NG",
            "reason": f"七格外观中{good}格符合已确认标准"}


def png_data(image: np.ndarray) -> str:
    ok, buffer = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("PNG 编码失败")
    return "data:image/png;base64," + base64.b64encode(buffer).decode("ascii")


def parameter_gallery(images: list[np.ndarray], labels: list[str], element_id: str) -> str:
    """Offline HTML selector for already-computed alternatives, no external JS/CDN."""
    if len(images) != len(labels) or not images:
        raise ValueError("参数图与说明必须一一对应")
    safe_id = "gallery-" + "".join(ch for ch in element_id if ch.isalnum() or ch == "-")
    encoded = [png_data(image) for image in images]
    options = "".join(f'<option value="{idx}">{html.escape(label)}</option>' for idx, label in enumerate(labels))
    payload = html.escape(json.dumps(encoded), quote=True)
    return (f'<div style="border:1px solid #9aafc3;padding:16px;border-radius:8px;font-size:18px">'
            f'<label>切换预计算参数效果：<select style="font-size:18px" onchange="document.getElementById(\'{safe_id}\').src=JSON.parse(this.dataset.images)[this.value]" data-images="{payload}">{options}</select></label>'
            f'<br><img id="{safe_id}" src="{encoded[0]}" style="max-width:100%;margin-top:12px" alt="公开照片的参数对照，派生图并非新增独立样本"></div>')


def create_database(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA foreign_keys=ON")
    connection.executescript("""
        CREATE TABLE IF NOT EXISTS inspections (
          id INTEGER PRIMARY KEY, event_key TEXT NOT NULL UNIQUE,
          source_id TEXT NOT NULL, source_sha256 TEXT NOT NULL,
          captured_at TEXT NOT NULL, status TEXT NOT NULL CHECK(status IN ('OK','NG','REVIEW')),
          observed INTEGER NOT NULL CHECK(observed >= 0), evidence_path TEXT NOT NULL,
          recipe_json TEXT NOT NULL, quality_json TEXT NOT NULL, result_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS sync_outbox (
          id INTEGER PRIMARY KEY, inspection_id INTEGER NOT NULL UNIQUE REFERENCES inspections(id),
          state TEXT NOT NULL DEFAULT 'pending' CHECK(state IN ('pending','synced','failed')),
          attempts INTEGER NOT NULL DEFAULT 0, payload_json TEXT NOT NULL, last_error TEXT
        );
    """)
    return connection


def save_inspection(connection: sqlite3.Connection, *, event_key: str, source_id: str,
                    source_sha256: str, evidence_path: str, result: dict,
                    recipe: dict, quality: dict, observed: int = 1) -> tuple[int, bool]:
    if not event_key or len(source_sha256) != 64 or result.get("status") not in ("OK", "NG", "REVIEW"):
        raise ValueError("事件标识、来源摘要或检测状态无效")
    existing = connection.execute("SELECT id FROM inspections WHERE event_key=?", (event_key,)).fetchone()
    if existing:
        return int(existing[0]), False
    created_at = datetime.now(timezone.utc).isoformat()
    with connection:
        cursor = connection.execute("""INSERT INTO inspections
            (event_key,source_id,source_sha256,captured_at,status,observed,evidence_path,recipe_json,quality_json,result_json)
            VALUES(?,?,?,?,?,?,?,?,?,?)""", (event_key, source_id, source_sha256, created_at,
            result["status"], observed, evidence_path, json.dumps(recipe, ensure_ascii=False),
            json.dumps(quality, ensure_ascii=False), json.dumps(result, ensure_ascii=False)))
        record_id = int(cursor.lastrowid)
        connection.execute("INSERT INTO sync_outbox(inspection_id,payload_json) VALUES(?,?)",
                           (record_id, json.dumps({"event_key": event_key, "result": result}, ensure_ascii=False)))
    return record_id, True
