from __future__ import annotations

from contextlib import contextmanager
from http.server import ThreadingHTTPServer
import importlib.util
import hashlib
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


serve = load_module("lab_serve_site", "serve_site.py")
manage = load_module("lab_manage_site", "manage_site.py")


def fixture_site(directory: Path):
    directory.mkdir()
    lessons = [{"html": f"site/lesson_{index}.html"} for index in range(10)]
    (directory / "execution_report.json").write_text(json.dumps({"private_data_read": False,
        "notebooks_passed": 10, "notebooks": lessons}), encoding="utf-8")
    (directory / "source_manifest.json").write_text(json.dumps({"private_images_included": False}), encoding="utf-8")
    for name in ["index.html", "data-sources.html"] + [f"lesson_{index}.html" for index in range(10)]:
        (directory / name).write_text(f"<html><body>PUBLIC {name}</body></html>", encoding="utf-8")


@contextmanager
def running_server(directory: Path):
    handler = serve.make_handler(directory, instance_id="test-instance", revision="test-revision")
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


class StaticSiteTests(unittest.TestCase):
    def test_onsite_lesson_requires_explicit_authorization_and_checksum(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp) / "site"
            fixture_site(directory)
            report_path = directory / "execution_report.json"
            report = json.loads(report_path.read_text())
            report["notebooks_passed"] = 11
            report["notebooks"].append({"html": "site/onsite.html", "publication_scope": "user_authorized_onsite"})
            report["authorized_onsite_data_included"] = True
            (directory / "onsite.html").write_text("<html>AUTHORIZED CASE</html>")
            manifest_path = directory / "onsite_capture_manifest.json"
            manifest_path.write_text(json.dumps({"publication_scope": "user_authorized_onsite",
                "authorization": "Owner authorized this teaching case", "license": "Owner retains rights"}))
            report["onsite_manifest_sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            report_path.write_text(json.dumps(report))
            self.assertEqual(len(serve.public_files(directory)), 16)
            with running_server(directory) as base:
                with urllib.request.urlopen(base + "/onsite.html") as response:
                    self.assertIn(b"AUTHORIZED CASE", response.read())
            expected_hash = report["onsite_manifest_sha256"]
            report["onsite_manifest_sha256"] = "0" * 64
            report_path.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, "checksum"):
                serve.public_files(directory)
            report["onsite_manifest_sha256"] = expected_hash
            manifest_path.write_text(json.dumps({"publication_scope": "user_authorized_onsite"}))
            report["onsite_manifest_sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            report_path.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, "authorization"):
                serve.public_files(directory)

    def test_only_published_files_and_health_are_served(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp) / "site"
            fixture_site(directory)
            (directory / "not-published.json").write_text("LOCAL-ONLY-TEST-CANARY")
            with running_server(directory) as base:
                with urllib.request.urlopen(base + "/") as response:
                    self.assertEqual(response.status, 200)
                    self.assertIn(b"PUBLIC index.html", response.read())
                    self.assertIn("form-action 'none'", response.headers["Content-Security-Policy"])
                with urllib.request.urlopen(base + "/healthz") as response:
                    health = json.load(response)
                    self.assertEqual(health["app"], serve.APP)
                    self.assertEqual(health["published_files"], 14)
                for path in ["/README.md", "/.env", "/data/private/image.png", "/../lab_utils.py",
                             "/%2e%2e/README.md", "/not-published.json", "/api/execute", "/notebooks/"]:
                    with self.assertRaises(urllib.error.HTTPError) as caught:
                        urllib.request.urlopen(base + path)
                    self.assertEqual(caught.exception.code, 404, path)

    def test_mutating_methods_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp) / "site"
            fixture_site(directory)
            with running_server(directory) as base:
                for method in ["POST", "PUT", "PATCH", "DELETE", "OPTIONS"]:
                    request = urllib.request.Request(base + "/", data=b"test", method=method)
                    with self.assertRaises(urllib.error.HTTPError) as caught:
                        urllib.request.urlopen(request)
                    self.assertEqual(caught.exception.code, 405)
                    self.assertEqual(caught.exception.headers["Allow"], "GET, HEAD")

    def test_head_has_headers_without_body(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp) / "site"
            fixture_site(directory)
            with running_server(directory) as base:
                with urllib.request.urlopen(urllib.request.Request(base + "/", method="HEAD")) as response:
                    self.assertEqual(response.status, 200)
                    self.assertGreater(int(response.headers["Content-Length"]), 0)
                    self.assertEqual(response.read(), b"")

    def test_publication_rejects_private_report_or_symlink(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp) / "site"
            fixture_site(directory)
            report = directory / "execution_report.json"
            data = json.loads(report.read_text())
            data["private_data_read"] = True
            report.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "public-only"):
                serve.public_files(directory)
            data["private_data_read"] = False
            report.write_text(json.dumps(data))
            external = Path(temp) / "external.html"
            external.write_text("PRIVATE-TEST-CANARY")
            page = directory / "lesson_0.html"
            page.unlink()
            try:
                page.symlink_to(external)
            except OSError:
                self.skipTest("Symlink creation unavailable on this host; verified on Linux deployment")
            with self.assertRaisesRegex(ValueError, "symlink"):
                serve.public_files(directory)


class ProcessSafetyTests(unittest.TestCase):
    def test_stop_does_not_signal_reused_unrelated_pid(self):
        state = {"pid": 12345, "port": 4034, "instance_id": "old-instance"}
        with patch.object(manage, "alive", return_value=True), patch.object(manage, "commandline", return_value=["other-server"]), patch.object(manage.os, "kill") as kill:
            with self.assertRaisesRegex(RuntimeError, "refusing"):
                manage.stop(state)
            kill.assert_not_called()

    def test_legacy_adoption_requires_exact_site_and_port(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            pid_path = directory / "site.pid"
            pid_path.write_text("12345")
            with patch.object(manage, "ROOT", directory), patch.object(manage, "PID_FILE", pid_path), patch.object(manage, "alive", return_value=True), patch.object(manage, "commandline", return_value=["python", "-m", "http.server", "4034", "--directory", "/another/project/site"]), patch.object(manage, "terminate") as terminate:
                with self.assertRaisesRegex(RuntimeError, "exact static server"):
                    manage.adopt_legacy(4034)
                terminate.assert_not_called()

    def test_start_is_idempotent_when_same_managed_server_is_running(self):
        state = {"pid": 12345, "port": 4034, "bind": "0.0.0.0", "instance_id": "test"}
        expected = {"status": "running", "pid": 12345}
        with patch.object(manage, "status", return_value=expected), patch.object(manage.subprocess, "Popen") as spawn:
            self.assertEqual(manage.start(state, bind="0.0.0.0", port=4034), expected)
            spawn.assert_not_called()


if __name__ == "__main__":
    unittest.main()
