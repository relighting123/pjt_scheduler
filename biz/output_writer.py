"""Persist conversion outputs.

Output schema (README §4 output):
  RULE_TIMEKEY | FROM_BATCH | FROM_PLAN_PROD_KEY | FROM_OPER_ID | EQP_MODEL_CD |
  TO_BATCH_ID | TO_PLAN_PROD_KEY | TO_OPER_ID | TO_EQP_MODEL_CD | START_CONV_TIME | EQP_QTY

The actual DELETE/INSERT statements live as standalone .sql files alongside
the input queries (`config/queries/{delete,insert}_output.sql`,
`insert_history.sql`). Operators edit those files instead of touching code
or JSON. Bind names below match the column names.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

from core.domain import AllocationSet


OUTPUT_COLUMNS: Tuple[str, ...] = (
    "RULE_TIMEKEY",
    "FROM_BATCH",
    "FROM_PLAN_PROD_KEY",
    "FROM_OPER_ID",
    "FROM_EQP_MODEL_CD",
    "TO_BATCH_ID",
    "TO_PLAN_PROD_KEY",
    "TO_OPER_ID",
    "TO_EQP_MODEL_CD",
    "START_CONV_TIME",
    "EQP_QTY",
)

# Named bind keys for the INSERT statements; matches OUTPUT_COLUMNS order.
_BIND_KEYS: Tuple[str, ...] = tuple(c.lower() for c in OUTPUT_COLUMNS)

_OUTPUT_QUERY_FILES = {
    "delete_output":  "delete_output.sql",
    "insert_output":  "insert_output.sql",
    "insert_history": "insert_history.sql",
}


def _load_output_sql(query_dir: Optional[str], kind: str) -> str:
    """`config/queries/<file>` 에서 SQL을 읽어옴."""
    if not query_dir:
        raise FileNotFoundError(
            f"oracle.query_dir not set; cannot resolve '{kind}' query."
        )
    path = Path(query_dir) / _OUTPUT_QUERY_FILES[kind]
    if not path.exists():
        raise FileNotFoundError(f"Query file not found: {path}")
    return path.read_text()


def build_conversion_rows(
    rule_timekey: str,
    previous: Optional[AllocationSet],
    current: AllocationSet,
) -> List[Tuple]:
    """현재 할당과 이전 할당을 diff해 conversion 행을 생성.

    이전 대비 (batch, model) 장비 수가 증가한 만큼 행 1줄. FROM은 이전
    스냅샷에서 같은 모델을 다른 batch에서 가지고 있던 첫 항목.

    Args:
        rule_timekey: 출력 키 = START_CONV_TIME.
        previous: 이전 AllocationSet (없으면 모두 신규).
        current: 현재 AllocationSet.

    Returns:
        OUTPUT_COLUMNS 순서의 튜플 리스트.

    Example:
        rows = build_conversion_rows("2026051707000000", None, alloc)
        # [("2026051707000000", "", "", "", "T5833",
        #   "9C/92", "P1", "OP10", "T5833", "2026051707000000", 2)]
    """
    prev_by_bm = {}
    if previous is not None:
        for a in previous.allocations:
            key = (a.batch_id, a.eqp_model_cd)
            prev_by_bm[key] = prev_by_bm.get(key, []) + [a]

    rows: List[Tuple] = []
    for a in current.allocations:
        prior = prev_by_bm.get((a.batch_id, a.eqp_model_cd), [])
        prior_qty = sum(p.eqp_qty for p in prior)
        delta = max(0, a.eqp_qty - prior_qty)
        if delta == 0:
            continue
        # find a source allocation with the same model from another batch
        from_batch = ""
        from_pk = ""
        from_op = ""
        if previous:
            for p in previous.allocations:
                if p.eqp_model_cd == a.eqp_model_cd and p.batch_id != a.batch_id:
                    from_batch, from_pk, from_op = p.batch_id, p.plan_prod_key, p.oper_id
                    break
        rows.append((
            rule_timekey,
            from_batch,
            from_pk,
            from_op,
            a.eqp_model_cd,
            a.batch_id,
            a.plan_prod_key,
            a.oper_id,
            a.eqp_model_cd,
            rule_timekey,  # START_CONV_TIME == RULE_TIMEKEY (README)
            int(delta),
        ))
    return rows


def _rows_to_named_binds(rows: Sequence[Tuple]) -> List[dict]:
    """튜플 행을 named-bind dict로 변환 (insert SQL이 :rule_timekey 등을 사용)."""
    return [dict(zip(_BIND_KEYS, r)) for r in rows]


def write_oracle(
    conn,
    query_dir: Optional[str],
    rule_timekey: str,
    rows: Sequence[Tuple],
    write_history: bool = True,
) -> None:
    """결정 결과를 출력 테이블에 기록 (DELETE + INSERT 패턴) + 이력 append.

    실제 SQL은 query_dir의 세 파일을 그대로 실행:
        delete_output.sql   bind :rule_timekey
        insert_output.sql   bind :rule_timekey, :from_batch, ...
        insert_history.sql  bind 동일

    Args:
        conn: oracledb 연결.
        query_dir: SQL 파일 디렉터리 (예: "config/queries").
        rule_timekey: 출력 키.
        rows: build_conversion_rows의 출력 (OUTPUT_COLUMNS 순서).
        write_history: False면 이력 INSERT 생략.

    Example:
        write_oracle(conn, "config/queries", "2026051707000000", rows)
    """
    from core.db import cursor

    delete_sql = _load_output_sql(query_dir, "delete_output")
    insert_sql = _load_output_sql(query_dir, "insert_output")
    history_sql = _load_output_sql(query_dir, "insert_history") if write_history else None

    bind_rows = _rows_to_named_binds(rows)
    try:
        with cursor(conn) as cur:
            cur.execute(delete_sql, {"rule_timekey": rule_timekey})
            if bind_rows:
                cur.executemany(insert_sql, bind_rows)
        if history_sql and bind_rows:
            with cursor(conn) as cur:
                cur.executemany(history_sql, bind_rows)
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def write_csv(path: str | Path, rows: Iterable[Tuple]) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(OUTPUT_COLUMNS)
        for r in rows:
            writer.writerow(r)
    return str(path)
