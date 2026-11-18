"""Report only SQLite table names, row counts, and schema checksums."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3


def main() -> None:
    root = Path(__file__).resolve().parents[1] / "courses/industrial-vision-lab/data/legacy_import/database"
    report = {}
    for path in sorted(root.glob("*.sqlite3")):
        connection = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
        try:
            tables = {}
            for name, sql in connection.execute(
                "SELECT name,sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            ):
                quoted = '"' + name.replace('"', '""') + '"'
                tables[name] = {"rows": connection.execute(f"SELECT count(*) FROM {quoted}").fetchone()[0],
                                "schema_sha256": hashlib.sha256((sql or "").encode()).hexdigest()}
            report[path.name] = tables
        finally:
            connection.close()
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
