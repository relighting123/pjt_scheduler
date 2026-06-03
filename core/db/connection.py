"""Oracle DB connection / query / write helpers.

Lazy-imports `oracledb` so the package stays usable in environments without it
(benchmarks, CI). Provides simple helpers: `connect`, `fetch_all`, `execute`,
plus a `replace_table` convenience for the RTD_CONV_INF / HIS upsert pattern.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def connect(user: str, password: str, dsn: str):
    """Oracle 연결. oracledb가 설치되어 있어야 함.

    Example:
        conn = connect("dispatcher", "dispatcher", "localhost:1521/XEPDB1")
    """
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
    """원자적 replace: 조건 DELETE → INSERT → 단일 commit.

    Example:
        replace_table(conn, "RTD_CONV_INF",
                      columns=OUTPUT_COLUMNS, rows=[(...), ...],
                      where_clause="RULE_TIMEKEY = :rule_timekey",
                      where_params={"rule_timekey": "2026051707000000"})
    """
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
