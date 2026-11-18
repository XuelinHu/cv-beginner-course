"""Execute every lesson in the current Python environment and export offline HTML."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import html
from importlib.metadata import version
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import nbformat
from nbclient import NotebookClient
from nbconvert.filters.markdown import markdown2html
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import lab_utils as lab
import onsite_utils as onsite

STYLE = """
:root{color-scheme:light}*{box-sizing:border-box}body{margin:0;background:#edf2f7;color:#172b42;font-family:system-ui,'Microsoft YaHei',sans-serif;font-size:18px;line-height:1.7}main{max-width:1120px;margin:auto;background:white;padding:32px 44px;min-height:100vh}h1{font-size:34px;line-height:1.3;color:#123b68}h2{font-size:26px;margin-top:34px}h3{font-size:22px}a{color:#064f9e;text-underline-offset:3px}pre{padding:18px;background:#eff4f9;border:1px solid #b9c8d8;border-radius:8px;overflow:auto;font-size:16px;line-height:1.55;white-space:pre-wrap;overflow-wrap:anywhere}code{font-family:Consolas,monospace}p code,li code{background:#edf2f8;padding:2px 5px;border-radius:3px}img{max-width:100%;height:auto}table{border-collapse:collapse;display:block;overflow:auto;width:100%}td,th{padding:10px 12px;border:1px solid #adbed0;text-align:left}th{background:#e9f1fa}nav{display:flex;gap:16px;flex-wrap:wrap;border-bottom:1px solid #b4c4d8;padding-bottom:16px;margin-bottom:24px}.lesson-note{padding:14px 18px;background:#fff5d3;border:1px solid #c6a344;border-radius:8px}.cell{margin:22px 0}.output{margin:12px 0}.stream{background:#f3f7f3;border-color:#b9ccb9}.error{background:#fff0f0;border:1px solid #c44}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:15px}.card{border:1px solid #adbed0;border-radius:9px;padding:17px;display:block;text-decoration:none;background:#f6f9fd}.card strong{display:block;font-size:21px}footer{margin-top:40px;border-top:1px solid #adbed0;padding-top:18px;font-size:16px;color:#3e566e}select{max-width:100%;padding:6px}blockquote{border-left:4px solid #b68a25;padding:4px 18px;background:#fff7df}@media(max-width:700px){main{padding:22px 17px}h1{font-size:29px}pre{font-size:14px}body{font-size:18px}}
"""


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def page(title: str, body: str, *, previous: str | None = None, following: str | None = None) -> str:
    nav = '<a href="index.html">课程目录</a><a href="data-sources.html">素材与许可</a><a href="execution_report.json">执行记录</a>'
    if previous:
        nav += f'<a href="{html.escape(previous)}">上一课</a>'
    if following:
        nav += f'<a href="{html.escape(following)}">下一课</a>'
    return ('<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>{html.escape(title)} · 工业视觉入门实验室</title><style>{STYLE}</style></head>'
            f'<body><main><nav>{nav}</nav>{body}'
            '<footer>非商业教学 · 公开素材保留原许可；现场实拍仅按用户授权用于本项目教程，权利归原权利人。'
            '教材执行成功不等于产品通过工业验收；未知应待复核。所有图表与交互选图可离线查看。</footer></main></body></html>')


def export_notebook(notebook) -> str:
    chunks = ['<p class="lesson-note">本页由已执行的 Jupyter 笔记本导出。代码可以复制回笔记本修改后重跑。黄色表示需复核，不代表检测合格。</p>']
    for cell in notebook.cells:
        if cell.cell_type == "markdown":
            text = markdown2html(cell.source)
            text = text.replace('../DATA_SOURCES.md', 'data-sources.html').replace('../sources/source_manifest.json', 'source_manifest.json')
            chunks.append(f'<section class="cell">{text}</section>')
        elif cell.cell_type == "code":
            chunks.append(f'<section class="cell"><pre class="input"><code>{html.escape(cell.source)}</code></pre>')
            for output in cell.get("outputs", []):
                if output.output_type == "error":
                    raise RuntimeError("Refusing to export a notebook with an error output")
                if output.output_type == "stream":
                    chunks.append(f'<pre class="output stream">{html.escape(output.get("text", ""))}</pre>')
                elif output.output_type in ("display_data", "execute_result"):
                    data = output.get("data", {})
                    if "text/html" in data:
                        chunks.append(f'<div class="output">{data["text/html"]}</div>')
                    elif "image/png" in data:
                        chunks.append(f'<div class="output"><img alt="当前案例的真实运行图表" src="data:image/png;base64,{data["image/png"]}"></div>')
                    elif "text/plain" in data:
                        chunks.append(f'<pre class="output">{html.escape(data["text/plain"])}</pre>')
            chunks.append('</section>')
    return "\n".join(chunks)


def execute_one(path: Path, directory: Path, *, timeout: int) -> tuple[dict, dict]:
    notebook = nbformat.read(path, as_version=4)
    settings = notebook.metadata.get("industrial_vision_lab", {})
    if settings.get("private_data") is not False:
        raise ValueError(f"Notebook lacks explicit public-only metadata: {path.name}")
    scope = settings.get("publication_scope", "licensed_public")
    if scope not in {"licensed_public", "user_authorized_onsite"}:
        raise ValueError(f"Unknown publication scope: {path.name}")
    if scope == "user_authorized_onsite":
        onsite.load_case()
    with tempfile.TemporaryDirectory(prefix="industrial-vision-kernel-") as temp:
        kernel_root = Path(temp)
        kernel = kernel_root / "industrial-vision-current"
        kernel.mkdir()
        (kernel / "kernel.json").write_text(json.dumps({
            "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            "display_name": "Industrial Vision (current Python)", "language": "python",
            "env": {"MPLBACKEND": "module://matplotlib_inline.backend_inline", "PYTHONIOENCODING": "utf-8"},
        }), encoding="utf-8")
        spec_manager = KernelSpecManager(kernel_dirs=[str(kernel_root)], ensure_native_kernel=False)
        manager = KernelManager(kernel_name="industrial-vision-current", kernel_spec_manager=spec_manager)
        client = NotebookClient(notebook, km=manager, timeout=timeout, allow_errors=False,
                                resources={"metadata": {"path": str(ROOT)}})
        start = time.perf_counter()
        try:
            client.execute()
        finally:
            if manager.has_kernel:
                manager.shutdown_kernel(now=True)
            if client.kc is not None:
                client.kc.stop_channels()
        elapsed = time.perf_counter() - start
    output = directory / path.name
    nbformat.write(notebook, output)
    return notebook, {"notebook": path.name, "title": settings.get("title", path.stem),
                      "status": "passed", "seconds": round(elapsed, 2),
                      "publication_scope": scope,
                      "code_cells": sum(cell.cell_type == "code" for cell in notebook.cells),
                      "source_sha256": digest(path), "executed_sha256": digest(output), "errors": 0}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="Fetch missing public samples using the committed checksums")
    parser.add_argument("--timeout", type=int, default=120, help="Per-code-cell timeout")
    args = parser.parse_args()
    if args.download:
        subprocess.run([sys.executable, str(ROOT / "scripts/download_data.py")], cwd=ROOT, check=True)
    image_rows = [row for row in lab.manifest()["files"] if row["kind"] == "image"]
    for row in image_rows:
        lab.load_case(row["id"])
    onsite.load_case()
    paths = sorted((ROOT / "notebooks").glob("*.ipynb"))
    if len(paths) != 11:
        raise ValueError(f"Expected 11 source notebooks, found {len(paths)}")
    executed_dir, site = ROOT / "executed", ROOT / "site"
    executed_dir.mkdir(exist_ok=True); site.mkdir(exist_ok=True)
    reports = []
    for index, path in enumerate(paths):
        print(f"[{index + 1}/{len(paths)}] Executing {path.name}", flush=True)
        notebook, report = execute_one(path, executed_dir, timeout=args.timeout)
        body = export_notebook(notebook)
        output = site / f"{path.stem}.html"
        output.write_text(page(report["title"], body,
                              previous=paths[index - 1].stem + ".html" if index else None,
                              following=paths[index + 1].stem + ".html" if index + 1 < len(paths) else None), encoding="utf-8")
        report["html"] = output.relative_to(ROOT).as_posix()
        report["html_sha256"] = digest(output)
        reports.append(report)
        print(f"  Passed ({report['seconds']} s)", flush=True)
    payload = {"executed_at": datetime.now(timezone.utc).isoformat(), "python": sys.version.split()[0],
               "environment": {name: version(name) for name in ["numpy", "opencv-python", "matplotlib", "nbformat", "nbclient", "nbconvert", "ipykernel"]},
               "source_manifest_sha256": digest(ROOT / "sources/source_manifest.json"),
               "public_case_families": sorted({row["dataset"] for row in image_rows}),
               "authorized_onsite_data_included": True,
               "onsite_manifest_sha256": digest(ROOT / "sources/onsite_capture_manifest.json"),
               "private_data_read": False, "notebooks_passed": len(reports), "notebooks": reports,
               "limitations": ["Teaching examples are not independent industrial validation", "No trained battery seven-slot recognizer", "Derived challenges are correlated with source photos"]}
    report_path = site / "execution_report.json"
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (site / "source_manifest.json").write_text((ROOT / "sources/source_manifest.json").read_text(encoding="utf-8"), encoding="utf-8")
    (site / "onsite_capture_manifest.json").write_bytes((ROOT / "sources/onsite_capture_manifest.json").read_bytes())
    sources = markdown2html((ROOT / "DATA_SOURCES.md").read_text(encoding="utf-8"))
    sources = sources.replace('sources/source_manifest.json', 'source_manifest.json')
    sources = sources.replace('sources/onsite_capture_manifest.json', 'onsite_capture_manifest.json')
    (site / "data-sources.html").write_text(page("素材与许可", sources), encoding="utf-8")
    descriptions = ["环境、来源与原标注", "暗环境与暗场、局部像素、试光记录", "灰度分布、全局与局部增强", "固定/自适应/Otsu、开闭运算", "边缘、局部标准差、方向纹理", "真实零件、当前编号、粘连与碎裂", "定位模板、旋转挑战、来源分组", "真实遮挡失败、拒判与验收", "五格规则、缺片与未知、坐标", "证据、幂等、待同步与私有导入", "当前实拍、方向校正、遮挡出画拒判与入库"]
    cards = ''.join(f'<a class="card" href="{path.stem}.html"><strong>{html.escape(report["title"])}</strong>{html.escape(description)}</a>'
                    for path, report, description in zip(paths, reports, descriptions))
    intro = ('<h1>工业视觉入门实验室</h1><p>从真实照片出发，先拍清楚，再定位、处理、计数、判规则、存证据。十一本中文可执行教程，适合第一次使用 OpenCV 的读者。</p>'
             '<p class="lesson-note">00–09 使用许可已记录的公开素材；新增第10课使用用户授权的当前 OsmoAction4 工件实拍，含取帧时间、实际计数与数据库证据。网页是实拍记录，不是实时视频。七格仍待复核，尚无独立工业验收结论。</p>'
             f'<p>本次完整执行通过 <strong>{len(reports)}/{len(paths)}</strong> 本；<a href="execution_report.json">查看版本、耗时和摘要</a>。无需联网即可阅读全部图表和参数选图。</p><div class="cards">{cards}</div>'
             '<h2>如何学习</h2><p>按顺序阅读 00–10；需要修改参数时，在 JupyterLab 打开 notebooks 里的同名文件并逐格执行。第10课调用课程工作区同级 battery-weld 项目的实际传统引擎，前十课可独立执行。</p>')
    (site / "index.html").write_text(page("课程目录", intro), encoding="utf-8")
    print(f"All {len(reports)} notebooks passed; offline HTML: site/index.html", flush=True)


if __name__ == "__main__":
    main()
