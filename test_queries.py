"""Verify Oracle query files exist (no DB connection required)."""
from __future__ import annotations

from biz.data_loader import load_sql, split_query_kinds
from biz.output_writer import _load_output_sql

_QUERY_DIR = "config/queries"


def test_input_query_files():
    for kind in ("source", "range_keys", "latest_key"):
        sql = load_sql(_QUERY_DIR, kind)
        assert sql.strip(), f"{kind} query is empty"


def test_split_query_files():
    for kind in split_query_kinds():
        sql = load_sql(_QUERY_DIR, kind)
        assert sql.strip(), f"{kind} query is empty"
        assert ":rule_timekey" in sql and ":fac_id" in sql


def test_output_query_files():
    for kind in ("delete_output", "insert_output", "insert_history"):
        sql = _load_output_sql(_QUERY_DIR, kind)
        assert sql.strip(), f"{kind} query is empty"


if __name__ == "__main__":
    test_input_query_files()
    test_split_query_files()
    test_output_query_files()
    print("All config/queries/*.sql files present and non-empty.")
