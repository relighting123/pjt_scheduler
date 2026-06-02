"""Oracle DB connection / query / write helpers.

Lazy-imports `oracledb` so the package stays usable in environments without it
(benchmarks, CI). Provides simple helpers: `connect`, `fetch_all`, `execute`,
plus a `replace_table` convenience for the RTD_CONV_INF / HIS upsert pattern.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def connect(user: str, password: str, dsn: str):
    try:
        import oracledb
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("oracledb is required; install pjt_scheduler[oracle].") from exc
    return oracledb.connect(user=user, password=password, dsn=dsn)


@contextmanager
def cursor(conn):
    cur = conn.cursor()
    try:
        yield cur
    finally:
        cur.close()


def fetch_all(conn, sql: str, params: Optional[Dict[str, Any]] = None) -> List[Tuple]:
    with cursor(conn) as cur:
        cur.execute(sql, params or {})
        return cur.fetchall()


def execute(conn, sql: str, params: Optional[Dict[str, Any]] = None) -> None:
    with cursor(conn) as cur:
        cur.execute(sql, params or {})


def executemany(conn, sql: str, rows: Iterable[Sequence[Any]]) -> None:
    rows = list(rows)
    if not rows:
        return
    with cursor(conn) as cur:
        cur.executemany(sql, rows)


def replace_table(
    conn,
    table: str,
    columns: Sequence[str],
    rows: Iterable[Sequence[Any]],
    where_clause: str = "1=1",
    where_params: Optional[Dict[str, Any]] = None,
) -> None:
    """Atomic replace: DELETE matching rows + INSERT new rows, single commit."""
    placeholders = ", ".join(f":{i + 1}" for i in range(len(columns)))
    col_csv = ", ".join(columns)
    delete_sql = f"DELETE FROM {table} WHERE {where_clause}"
    insert_sql = f"INSERT INTO {table} ({col_csv}) VALUES ({placeholders})"
    try:
        with cursor(conn) as cur:
            cur.execute(delete_sql, where_params or {})
            data = list(rows)
            if data:
                cur.executemany(insert_sql, data)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
