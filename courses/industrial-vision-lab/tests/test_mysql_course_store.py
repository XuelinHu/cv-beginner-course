from __future__ import annotations

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from mysql_course_store import MySQLConnection


class FakeCursor:
    def execute(self, statement, parameters):
        self.statement = statement
        self.parameters = parameters
        return self


class FakeRawConnection:
    def __init__(self):
        self.cursor_instance = FakeCursor()
        self.commits = 0
        self.rollbacks = 0

    def cursor(self):
        return self.cursor_instance

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1

    def close(self):
        pass


class MySQLStoreTests(unittest.TestCase):
    def test_sqlite_style_placeholders_are_adapted(self):
        raw = FakeRawConnection()
        connection = MySQLConnection(raw)
        connection.execute("SELECT id FROM inspections WHERE event_key=?", ("event-1",))
        self.assertEqual(raw.cursor_instance.statement, "SELECT id FROM inspections WHERE event_key=%s")
        self.assertEqual(raw.cursor_instance.parameters, ("event-1",))

    def test_transaction_commits_or_rolls_back(self):
        raw = FakeRawConnection()
        connection = MySQLConnection(raw)
        with connection:
            pass
        self.assertEqual(raw.commits, 1)
        with self.assertRaises(RuntimeError):
            with connection:
                raise RuntimeError("test rollback")
        self.assertEqual(raw.rollbacks, 1)


if __name__ == "__main__":
    unittest.main()
