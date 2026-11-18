from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import lab_utils as lab


class DataSafetyTests(unittest.TestCase):
    def test_corrupted_public_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            (directory / "sources").mkdir()
            path = directory / "sample.jpg"
            path.write_bytes(b"changed bytes, not a verified picture")
            manifest = {"files": [{"id": "example", "file": "sample.jpg", "sha256": "0" * 64}]}
            (directory / "sources/source_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with patch.object(lab, "ROOT", directory):
                with self.assertRaisesRegex(ValueError, "SHA256"):
                    lab.load_case("example")

    def test_private_import_stays_under_private_directory(self):
        spec = importlib.util.spec_from_file_location("lab_private_import", ROOT / "scripts/import_private.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        source = ROOT / "data/public/parts/separated.png"
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            with patch.object(module, "ROOT", directory), patch.object(sys, "argv", ["import_private.py", str(source), "--expected-count", "8"]):
                with contextlib.redirect_stdout(io.StringIO()):
                    module.main()
                    module.main()  # identical replay is safe
            output = directory / "data/private" / digest[:16]
            metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["source_sha256"], digest)
            self.assertEqual(metadata["expected_count"], 8)
            self.assertFalse(metadata["public_export_allowed"])
            self.assertEqual(metadata["slots_truth"], "unknown")
            self.assertNotIn(str(source), json.dumps(metadata))
            self.assertFalse((directory / "data/public").exists())
            self.assertFalse((directory / "site").exists())

    def test_evidence_and_pending_sync_are_atomic_and_idempotent(self):
        with tempfile.TemporaryDirectory() as temp:
            connection = lab.create_database(Path(temp) / "test.sqlite3")
            params = dict(event_key="session1:frame7:'test'", source_id="public-example", source_sha256="a" * 64,
                          evidence_path="data/public/example.png", result={"status": "REVIEW", "reason": "unknown"},
                          recipe={"version": 1}, quality={"exposure": None})
            first, created = lab.save_inspection(connection, **params)
            second, duplicate = lab.save_inspection(connection, **params)
            self.assertTrue(created)
            self.assertFalse(duplicate)
            self.assertEqual(first, second)
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM inspections").fetchone()[0], 1)
            self.assertEqual(connection.execute("SELECT state,attempts FROM sync_outbox").fetchall(), [("pending", 0)])
            self.assertEqual(json.loads(connection.execute("SELECT quality_json FROM inspections").fetchone()[0])["exposure"], None)
            connection.close()


if __name__ == "__main__":
    unittest.main()
