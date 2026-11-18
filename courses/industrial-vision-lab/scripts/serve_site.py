"""Serve only the published public tutorial pages. No notebook execution or uploads."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
import os
from pathlib import Path
import subprocess
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
APP = "industrial-vision-lab-static"


def public_files(directory: Path) -> set[str]:
    report = json.loads((directory / "execution_report.json").read_text(encoding="utf-8"))
    sources = json.loads((directory / "source_manifest.json").read_text(encoding="utf-8"))
    if report.get("private_data_read") is not False or sources.get("private_images_included") is not False:
        raise ValueError("Static publication requires an explicitly public-only report and source manifest")
    count = report.get("notebooks_passed")
    if count not in {10, 11} or len(report.get("notebooks", [])) != count:
        raise ValueError("Static publication requires all completed curated lessons")
    paths = {"index.html", "data-sources.html", "execution_report.json", "source_manifest.json"}
    if count == 11:
        onsite_path = directory / "onsite_capture_manifest.json"
        if onsite_path.is_symlink():
            raise ValueError("Authorized manifest cannot be a symlink")
        onsite = json.loads(onsite_path.read_text(encoding="utf-8"))
        if (report.get("authorized_onsite_data_included") is not True
                or onsite.get("publication_scope") != "user_authorized_onsite"
                or not onsite.get("authorization") or not onsite.get("license")
                or hashlib.sha256(onsite_path.read_bytes()).hexdigest() != report.get("onsite_manifest_sha256")):
            raise ValueError("On-site publication requires explicit authorization and matching manifest checksum")
        if sum(row.get("publication_scope") == "user_authorized_onsite" for row in report["notebooks"]) != 1:
            raise ValueError("Exactly one curated on-site lesson is authorized")
        paths.add("onsite_capture_manifest.json")
    for row in report["notebooks"]:
        name = row.get("html", "")
        if not name.startswith("site/") or Path(name).name != name[5:] or not name.endswith(".html"):
            raise ValueError("Invalid published lesson path")
        paths.add(name[5:])
    if len(paths) != count + 4 + (count == 11):
        raise ValueError("Publication requires distinct lesson pages and curated metadata/index files")
    for name in paths:
        candidate = directory / name
        if candidate.is_symlink() or not candidate.is_file() or not candidate.resolve().is_relative_to(directory.resolve()):
            raise ValueError(f"Published file is missing or is a symlink: {name}")
    return paths


def git_revision() -> str | None:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=3)
        return result.stdout.strip() if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def make_handler(directory: Path, *, instance_id: str, revision: str | None = None):
    directory = directory.resolve()
    allowed = public_files(directory)
    started_at = datetime.now(timezone.utc).isoformat()
    report_hash = hashlib.sha256((directory / "execution_report.json").read_bytes()).hexdigest()

    class PublicHandler(BaseHTTPRequestHandler):
        server_version = "IndustrialVisionStatic/1.0"
        sys_version = ""

        def _headers(self, status: int, content_type: str, length: int):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(length))
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'none'; img-src 'self' data:; style-src 'unsafe-inline'; script-src 'unsafe-inline'; base-uri 'none'; frame-ancestors 'none'; form-action 'none'")
            self.end_headers()

        def _serve(self, head_only: bool = False):
            path = unquote(urlsplit(self.path).path)
            if path == "/healthz":
                data = json.dumps({"app": APP, "status": "ready", "pid": os.getpid(),
                    "instance_id": instance_id, "revision": revision, "started_at": started_at,
                    "execution_report_sha256": report_hash, "published_files": len(allowed)}).encode("utf-8")
                self._headers(HTTPStatus.OK, "application/json; charset=utf-8", len(data))
                if not head_only:
                    self.wfile.write(data)
                return
            name = "index.html" if path == "/" else path.removeprefix("/")
            if name not in allowed or "\\" in name or any(part.startswith(".") for part in Path(name).parts):
                self.send_error(HTTPStatus.NOT_FOUND, "Public page not found")
                return
            candidate = directory / name
            if candidate.is_symlink() or not candidate.resolve().is_relative_to(directory) or not candidate.is_file():
                self.send_error(HTTPStatus.NOT_FOUND, "Public page not found")
                return
            try:
                with candidate.open("rb") as stream:
                    size = os.fstat(stream.fileno()).st_size
                    content_type = mimetypes.guess_type(name)[0] or "application/octet-stream"
                    self._headers(HTTPStatus.OK, content_type + "; charset=utf-8", size)
                    if not head_only:
                        for chunk in iter(lambda: stream.read(128 * 1024), b""):
                            self.wfile.write(chunk)
            except (BrokenPipeError, ConnectionResetError):
                return

        def do_GET(self):
            self._serve()

        def do_HEAD(self):
            self._serve(head_only=True)

        def _read_only(self):
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                length = 0
            if 0 < length <= 65536:
                self.connection.settimeout(5)
                self.rfile.read(length)
            data = b"This service provides read-only public tutorial pages.\n"
            self.send_response(HTTPStatus.METHOD_NOT_ALLOWED)
            self.send_header("Allow", "GET, HEAD")
            self.send_header("Connection", "close")
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            self.wfile.flush()
            self.close_connection = True

        do_POST = do_PUT = do_PATCH = do_DELETE = do_OPTIONS = _read_only

        def log_request(self, code="-", size="-"):
            self.log_message('%s %s %s %s', self.command, urlsplit(self.path).path, code, size)

    return PublicHandler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bind", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=4034)
    parser.add_argument("--instance-id", default="foreground")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("Port must be between 1 and 65535")
    directory = ROOT / "site"
    if directory.is_symlink() or not directory.resolve().is_relative_to(ROOT):
        raise ValueError("The public site directory must be inside this project")
    handler = make_handler(directory, instance_id=args.instance_id, revision=git_revision())
    server = ThreadingHTTPServer((args.bind, args.port), handler)
    print(json.dumps({"app": APP, "bind": args.bind, "port": args.port, "pid": os.getpid()}, ensure_ascii=False), flush=True)
    try:
        server.serve_forever(poll_interval=0.3)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
