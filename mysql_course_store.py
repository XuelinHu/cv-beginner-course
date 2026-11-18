"""MySQL backend for course evidence, with a SQLite-compatible small API."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sqlite3
from typing import Any


class MySQLConnection:
    def __init__(self, raw):
        self.raw = raw

    def execute(self, statement: str, parameters=()):
        cursor = self.raw.cursor()
        cursor.execute(statement.replace("?", "%s"), parameters)
        return cursor

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            self.raw.commit()
        else:
            self.raw.rollback()
        return False

    def close(self):
        self.raw.close()


def connect():
    import mysql.connector

    required = ("MYSQL_HOST", "MYSQL_USER", "MYSQL_PASSWORD", "MYSQL_DATABASE")
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        raise RuntimeError("Missing MySQL environment settings: " + ", ".join(missing))
    return MySQLConnection(mysql.connector.connect(
        host=os.environ["MYSQL_HOST"], port=int(os.environ.get("MYSQL_PORT", "3306")),
        user=os.environ["MYSQL_USER"], password=os.environ["MYSQL_PASSWORD"],
        database=os.environ["MYSQL_DATABASE"], charset="utf8mb4", autocommit=False,
    ))


def create_database(_sqlite_path: Path | None = None):
    connection = connect()
    connection.raw.cursor().execute("""
        CREATE TABLE IF NOT EXISTS inspections (
          id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
          event_key VARCHAR(255) NOT NULL UNIQUE,
          source_id VARCHAR(160) NOT NULL, source_sha256 CHAR(64) NOT NULL,
          captured_at VARCHAR(40) NOT NULL,
          status ENUM('OK','NG','REVIEW') NOT NULL,
          observed INT UNSIGNED NOT NULL, evidence_path TEXT NOT NULL,
          recipe_json JSON NOT NULL, quality_json JSON NOT NULL, result_json JSON NOT NULL,
          INDEX idx_inspections_captured (captured_at), INDEX idx_inspections_status (status)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci
    """)
    connection.raw.cursor().execute("""
        CREATE TABLE IF NOT EXISTS sync_outbox (
          id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
          inspection_id BIGINT UNSIGNED NOT NULL UNIQUE,
          state ENUM('pending','synced','failed') NOT NULL DEFAULT 'pending',
          attempts INT UNSIGNED NOT NULL DEFAULT 0, payload_json JSON NOT NULL,
          last_error TEXT NULL,
          CONSTRAINT fk_course_outbox_inspection FOREIGN KEY (inspection_id)
            REFERENCES inspections(id) ON DELETE RESTRICT
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci
    """)
    connection.raw.cursor().execute("""
        CREATE TABLE IF NOT EXISTS legacy_sqlite_rows (
          source_database VARCHAR(255) NOT NULL, table_name VARCHAR(255) NOT NULL,
          legacy_id VARCHAR(255) NOT NULL, row_json JSON NOT NULL,
          PRIMARY KEY (source_database, table_name, legacy_id)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci
    """)
    connection.raw.cursor().execute("""
        CREATE TABLE IF NOT EXISTS legacy_sqlite_tables (
          source_database VARCHAR(255) NOT NULL, table_name VARCHAR(255) NOT NULL,
          create_sql LONGTEXT NOT NULL,
          PRIMARY KEY (source_database, table_name)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci
    """)
    connection.raw.commit()
    return connection


def migrate_sqlite(paths: list[Path]) -> dict[str, int]:
    target = create_database()
    counts: dict[str, int] = {}
    try:
        for path in paths:
            source_name = path.name
            source = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
            source.row_factory = sqlite3.Row
            try:
                tables = source.execute("SELECT name,sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'").fetchall()
                for item in tables:
                    table = item["name"]
                    target.execute("""
                        INSERT IGNORE INTO legacy_sqlite_tables
                        (source_database,table_name,create_sql) VALUES(?,?,?)
                    """, (source_name, table, item["sql"] or ""))
                    rows = source.execute(f'SELECT * FROM "{table}"').fetchall()
                    counts[f"{source_name}.{table}"] = len(rows)
                    for row in rows:
                        record = dict(row)
                        legacy_id = str(record.get("id", record.get("event_key", "row")))
                        target.execute("""
                            INSERT IGNORE INTO legacy_sqlite_rows
                            (source_database,table_name,legacy_id,row_json) VALUES(?,?,?,?)
                        """, (source_name, table, legacy_id,
                              json.dumps(record, ensure_ascii=False, default=str)))
                        if table == "inspections":
                            existing = target.execute("SELECT id FROM inspections WHERE event_key=?",
                                                      (record["event_key"],)).fetchone()
                            if not existing:
                                inserted = target.execute("""
                                    INSERT INTO inspections
                                    (event_key,source_id,source_sha256,captured_at,status,observed,
                                     evidence_path,recipe_json,quality_json,result_json)
                                    VALUES(?,?,?,?,?,?,?,?,?,?)
                                """, tuple(record[key] for key in (
                                    "event_key", "source_id", "source_sha256", "captured_at", "status",
                                    "observed", "evidence_path", "recipe_json", "quality_json", "result_json")))
                                new_id = inserted.lastrowid
                                outbox = source.execute("SELECT * FROM sync_outbox WHERE inspection_id=?",
                                                        (record["id"],)).fetchone() if "sync_outbox" in [t["name"] for t in tables] else None
                                if outbox:
                                    target.execute("""
                                        INSERT IGNORE INTO sync_outbox
                                        (inspection_id,state,attempts,payload_json,last_error) VALUES(?,?,?,?,?)
                                    """, (new_id, outbox["state"], outbox["attempts"], outbox["payload_json"], outbox["last_error"]))
                target.raw.commit()
            finally:
                source.close()
        return counts
    except Exception:
        target.raw.rollback()
        raise
    finally:
        target.close()


if __name__ == "__main__":
    root = (Path(__file__).resolve().parent / "courses" / "industrial-vision-lab"
            / "data" / "legacy_import" / "database")
    print(json.dumps(migrate_sqlite([
        root / "teaching_inspections.sqlite3",
        root / "onsite_teaching.sqlite3",
    ]), ensure_ascii=False, indent=2))
