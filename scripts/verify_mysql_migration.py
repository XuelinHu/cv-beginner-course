"""Verify every imported SQLite schema and row against the MySQL snapshot."""
from __future__ import annotations

import json
from pathlib import Path
import sqlite3

from mysql_course_store import connect


def canonical(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def main() -> None:
    root = Path(__file__).resolve().parents[1] / "courses/industrial-vision-lab/data/legacy_import/database"
    connection = connect()
    tables_checked = rows_checked = 0
    try:
        for path in sorted(root.glob("*.sqlite3")):
            source = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
            source.row_factory = sqlite3.Row
            try:
                tables = source.execute(
                    "SELECT name,sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
                ).fetchall()
                for table in tables:
                    ddl = connection.execute(
                        "SELECT create_sql FROM legacy_sqlite_tables WHERE source_database=? AND table_name=?",
                        (path.name, table["name"]),
                    ).fetchone()
                    if not ddl or ddl[0] != (table["sql"] or ""):
                        raise AssertionError(f"SQLite schema mismatch: {path.name}.{table['name']}")
                    source_rows = source.execute(f'SELECT * FROM "{table["name"]}"').fetchall()
                    for row in source_rows:
                        record = dict(row)
                        legacy_id = str(record.get("id", record.get("event_key", "row")))
                        imported = connection.execute("""
                            SELECT row_json FROM legacy_sqlite_rows
                            WHERE source_database=? AND table_name=? AND legacy_id=?
                        """, (path.name, table["name"], legacy_id)).fetchone()
                        if not imported or canonical(json.loads(imported[0])) != canonical(record):
                            raise AssertionError(f"SQLite row mismatch: {path.name}.{table['name']}:{legacy_id}")
                        rows_checked += 1
                    tables_checked += 1
            finally:
                source.close()
        live = connection.execute("SELECT COUNT(*) FROM inspections").fetchone()[0]
        outbox = connection.execute("SELECT COUNT(*) FROM sync_outbox").fetchone()[0]
        legacy = connection.execute("SELECT COUNT(*) FROM legacy_sqlite_rows").fetchone()[0]
        if live < 4 or outbox < 4 or legacy != rows_checked:
            raise AssertionError(f"Unexpected row totals: inspections={live}, outbox={outbox}, legacy={legacy}")
        print(json.dumps({"schemas_verified": tables_checked, "source_rows_verified": rows_checked,
                          "legacy_rows": legacy, "operational_inspections": live,
                          "outbox_rows": outbox, "all_match": True}, indent=2))
    finally:
        connection.close()


if __name__ == "__main__":
    main()
