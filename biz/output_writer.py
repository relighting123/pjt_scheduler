"""Persist conversion outputs to RTD_CONV_INF and RTD_CONV_HIS.

Output schema (README §4 output):
  RULE_TIMEKEY | FROM_BATCH | FROM_PLAN_PROD_KEY | FROM_OPER_ID | EQP_MODEL_CD |
  TO_BATCH_ID | TO_PLAN_PROD_KEY | TO_OPER_ID | TO_EQP_MODEL_CD | START_CONV_TIME | EQP_QTY

RTD_CONV_INF is the latest snapshot (DELETE + INSERT for the current
RULE_TIMEKEY). RTD_CONV_HIS is append-only history.
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


def write_oracle(
    conn,
    output_table: str,
    history_table: str,
    rule_timekey: str,
    rows: Sequence[Tuple],
) -> None:
    from core.db import replace_table, executemany

    placeholders = ", ".join(f":{i + 1}" for i in range(len(OUTPUT_COLUMNS)))
    cols_csv = ", ".join(OUTPUT_COLUMNS)

    replace_table(
        conn,
        table=output_table,
        columns=OUTPUT_COLUMNS,
        rows=rows,
        where_clause="RULE_TIMEKEY = :rule_timekey",
        where_params={"rule_timekey": rule_timekey},
    )
    if history_table and rows:
        executemany(
            conn,
            f"INSERT INTO {history_table} ({cols_csv}) VALUES ({placeholders})",
            rows,
        )
        conn.commit()


def write_csv(path: str | Path, rows: Iterable[Tuple]) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(OUTPUT_COLUMNS)
        for r in rows:
            writer.writerow(r)
    return str(path)
