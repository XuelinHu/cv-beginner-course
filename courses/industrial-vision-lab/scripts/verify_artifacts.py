"""Verify actual outputs, public provenance, offline links and image integrity."""
from __future__ import annotations

import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import sys
from urllib.parse import unquote, urlsplit

import nbformat
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import lab_utils as lab
import onsite_utils as onsite


class Resources(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
        self.sources = []

    def handle_starttag(self, tag, attributes):
        attrs = dict(attributes)
        if attrs.get("href"):
            self.links.append(attrs["href"])
        if attrs.get("src"):
            self.sources.append((tag, attrs["src"]))


def main() -> None:
    manifest_path = ROOT / "sources/source_manifest.json"
    manifest = lab.manifest()
    assert manifest["private_images_included"] is False
    public = [row for row in manifest["files"] if row["file"].startswith("data/public/")]
    assert len(public) == 20
    assert {row["dataset"] for row in public} == {"weld", "kolektor", "parts"}
    for row in public:
        path = ROOT / row["file"]
        assert path.is_file() and lab.file_hash(path) == row["sha256"], row["id"]
        assert row["license"] and row["license_url"] and row["homepage"]
        if row["kind"] == "image":
            image = lab.load_case(row["id"])
            assert image.size > 0
            if row["dataset"] == "kolektor":
                mask = lab.load_case(row["id"] + "_mask", grayscale=True)
                assert mask.shape == image.shape[:2]
                assert bool(np.any(mask > 0)) == row["defective"]
    report_path = ROOT / "site/execution_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["notebooks_passed"] == 11 and report["private_data_read"] is False
    case, image = onsite.load_case()
    assert report["authorized_onsite_data_included"] is True
    assert report["onsite_manifest_sha256"] == lab.file_hash(ROOT / "sources/onsite_capture_manifest.json")
    assert (ROOT / "site/onsite_capture_manifest.json").read_bytes() == (ROOT / "sources/onsite_capture_manifest.json").read_bytes()
    assert case["database_evidence"] == {"acknowledged": True, "png_file_hash_verified": True, "decoded_bgr_hash_verified": True}
    assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == report["source_manifest_sha256"]
    image_outputs = 0
    for row in report["notebooks"]:
        assert row["status"] == "passed" and row["errors"] == 0
        for directory, field in [("notebooks", "source_sha256"), ("executed", "executed_sha256")]:
            path = ROOT / directory / row["notebook"]
            assert lab.file_hash(path) == row[field], path.name
        assert lab.file_hash(ROOT / row["html"]) == row["html_sha256"]
        notebook = nbformat.read(ROOT / "executed" / row["notebook"], as_version=4)
        assert notebook.metadata.industrial_vision_lab.private_data is False
        assert all(cell.get("execution_count") is not None for cell in notebook.cells if cell.cell_type == "code")
        for cell in notebook.cells:
            for output in cell.get("outputs", []):
                assert output.output_type != "error"
                image_outputs += int("image/png" in output.get("data", {}))
    assert image_outputs >= 20
    pages = list((ROOT / "site").glob("*.html"))
    assert len(pages) == 13
    for page in pages:
        parser = Resources()
        text = page.read_text(encoding="utf-8")
        parser.feed(text)
        assert 'lang="zh-CN"' in text
        for tag, value in parser.sources:
            assert value.startswith("data:image/"), f"Unexpected external resource: {tag} in {page.name}"
        for value in parser.links:
            link = urlsplit(value)
            assert not link.username and not link.password
            if link.scheme or not link.path:
                continue
            target = (page.parent / unquote(link.path)).resolve()
            assert target.is_relative_to(ROOT / "site"), f"Link leaves site: {page.name} {value}"
            assert target.exists(), f"Broken link: {page.name} {value}"
    print(f"Verified 20 public files / 3 public case families / 1 authorized on-site case / 11 executed notebooks / {image_outputs} plots / 13 offline HTML pages.")


if __name__ == "__main__":
    main()
