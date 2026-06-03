"""Adapters that materialize `SchedulingProblem` from CSV benchmarks or the
Oracle RTS_LINEDSDB_INF table.

Schema reference (from README):

  RTS_LINEDSDB_INF: (RULE_TIMEKEY, FAC_ID, BATCH_ID, PLAN_PROD_KEY, OPER_ID,
                     OPER_SEQ, EQP_MODEL_CD, GBN_CD, ATTR_VAL)

GBN_CD values pivot into the seven logical tables described in the README:

  WIP_QTY              -> WipRecord
  UPH                  -> UphRecord
  ASSIGN_EQUIP_CNT     -> EquipmentRecord
  D0_TARGET_QTY,
  D1_TARGET_QTY        -> PlanRecord (D0=until next 07h, D1=07h..07h next day)
  TOOL_QTY             -> ToolQtyRecord
  (availability and tool-group are derived: UPH > 0 => available;
   tool-group = batch_id grouping is given by config + presence in BATCH_ID).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from core.domain import (
    AvailabilityRecord,
    EquipmentRecord,
    PlanRecord,
    SchedulingProblem,
    ToolGroupRecord,
    ToolQtyRecord,
    UphRecord,
    WipRecord,
)


# ---------------------------------------------------------------------------
# CSV benchmark loader
# ---------------------------------------------------------------------------
def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def load_problem_from_csv_dir(directory: Path | str) -> SchedulingProblem:
    """벤치마크 디렉토리(7개 CSV + tool_groups.json)에서 SchedulingProblem 로드.

    Args:
        directory: 벤치마크 폴더 경로 (e.g. "benchmarks/benchmark_01").

    Returns:
        SchedulingProblem (rule_timekey = 폴더명).

    Example:
        problem = load_problem_from_csv_dir("benchmarks/benchmark_01")
        # 폴더 내용:
        #   wip_info.csv, uph.csv, equipment.csv, availability.csv,
        #   tool_group.csv, tool_qty.csv, plan.csv,
        #   tool_groups.json, ground_truth.json
    """
    d = Path(directory)
    rule_timekey = d.name

    wip = [
        WipRecord(
            rule_timekey=r.get("rule_timekey", rule_timekey),
            plan_prod_key=r["plan_prod_key"],
            oper_id=r["oper_id"],
            oper_seq=int(r["oper_seq"]),
            wip_qty=float(r["wip_qty"]),
        )
        for r in _read_csv(d / "wip_info.csv")
    ]
    uph = [
        UphRecord(
            rule_timekey=r.get("rule_timekey", rule_timekey),
            plan_prod_key=r["plan_prod_key"],
            oper_id=r["oper_id"],
            eqp_model_cd=r["eqp_model_cd"],
            uph=float(r["uph"]),
        )
        for r in _read_csv(d / "uph.csv")
    ]
    equipment = [
        EquipmentRecord(
            rule_timekey=r.get("rule_timekey", rule_timekey),
            batch_id=r["batch_id"],
            eqp_model_cd=r["eqp_model_cd"],
            eqp_qty=int(r["eqp_qty"]),
        )
        for r in _read_csv(d / "equipment.csv")
    ]
    availability = [
        AvailabilityRecord(
            rule_timekey=r.get("rule_timekey", rule_timekey),
            plan_prod_key=r["plan_prod_key"],
            oper_id=r["oper_id"],
            eqp_model_cd=r["eqp_model_cd"],
            avail_yn=str(r["avail_yn"]).upper() in ("Y", "TRUE", "1"),
        )
        for r in _read_csv(d / "availability.csv")
    ]
    tool_groups = [
        ToolGroupRecord(
            rule_timekey=r.get("rule_timekey", rule_timekey),
            batch_id=r["batch_id"],
            plan_prod_key=r["plan_prod_key"],
            oper_id=r["oper_id"],
        )
        for r in _read_csv(d / "tool_group.csv")
    ]
    tool_qty = [
        ToolQtyRecord(
            rule_timekey=r.get("rule_timekey", rule_timekey),
            batch_id=r["batch_id"],
            eqp_model_cd=r["eqp_model_cd"],
            tool_qty=int(r["tool_qty"]),
        )
        for r in _read_csv(d / "tool_qty.csv")
    ]
    plans = [
        PlanRecord(
            rule_timekey=r.get("rule_timekey", rule_timekey),
            plan_prod_key=r["plan_prod_key"],
            oper_id=r["oper_id"],
            start_time=r["start_time"],
            end_time=r["end_time"],
            plan_qty=float(r["plan_qty"]),
        )
        for r in _read_csv(d / "plan.csv")
    ]

    eqp_groups = _load_groups_from_meta(d / "tool_groups.json")
    problem = SchedulingProblem(
        rule_timekey=rule_timekey,
        wip=wip,
        uph=uph,
        equipment=equipment,
        availability=availability,
        tool_groups=tool_groups,
        tool_qty=tool_qty,
        plans=plans,
        eqp_model_groups=eqp_groups,
    )
    return problem


def _load_groups_from_meta(path: Path) -> Dict[str, List[str]]:
    if path.exists():
        return json.loads(path.read_text())
    return {}


# ---------------------------------------------------------------------------
# Oracle loader
# ---------------------------------------------------------------------------
# Default queries — used when settings don't override them. `{table}` is a
# Python format placeholder for the configured table name; `:name` are Oracle
# bind variables (kept distinct so users overriding the query can use the
# bind names directly).
_SELECT_SNAPSHOT_SQL = """
SELECT RULE_TIMEKEY, BATCH_ID, PLAN_PROD_KEY, OPER_ID, OPER_SEQ,
       EQP_MODEL_CD, GBN_CD, ATTR_VAL
  FROM {table}
 WHERE RULE_TIMEKEY = :rule_timekey
"""

_SELECT_RANGE_KEYS_SQL = """
SELECT DISTINCT RULE_TIMEKEY
  FROM {table}
 WHERE RULE_TIMEKEY BETWEEN :from_key AND :to_key
 ORDER BY RULE_TIMEKEY
"""

_SELECT_MAX_KEY_SQL = """
SELECT MAX(RULE_TIMEKEY) FROM {table}
 WHERE GBN_CD = 'WIP_QTY'
"""


def list_rule_timekeys(
    conn,
    table: str,
    from_key: str,
    to_key: str,
    custom_sql: Optional[str] = None,
) -> List[str]:
    """구간 안의 distinct RULE_TIMEKEY 목록.

    Args:
        conn: oracledb 연결.
        table: 기본 쿼리에 채워지는 테이블명 (custom_sql 지정 시 무시).
        from_key, to_key: 조회 구간 (binds :from_key, :to_key).
        custom_sql: 사용자가 직접 작성한 쿼리. RULE_TIMEKEY 컬럼 1개를
            반환해야 하고, `:from_key`와 `:to_key` bind를 그대로 사용해야 함.

    Example custom_sql:
        SELECT DISTINCT RULE_TIMEKEY
          FROM MY_VIEW
         WHERE RULE_TIMEKEY BETWEEN :from_key AND :to_key
           AND FAC_ID = 'ICPRB'
    """
    from core.db import fetch_all
    sql = custom_sql or _SELECT_RANGE_KEYS_SQL.format(table=table)
    rows = fetch_all(conn, sql, {"from_key": from_key, "to_key": to_key})
    return [r[0] for r in rows]


def latest_rule_timekey(
    conn,
    table: str,
    custom_sql: Optional[str] = None,
) -> Optional[str]:
    """가장 최근 RULE_TIMEKEY (MAX).

    Args:
        conn: oracledb 연결.
        table: 기본 쿼리에 채워지는 테이블명 (custom_sql 지정 시 무시).
        custom_sql: 사용자가 직접 작성한 쿼리. 단일 컬럼(=MAX 값)을 반환해야 함.

    Example custom_sql:
        SELECT MAX(RULE_TIMEKEY) FROM MY_VIEW WHERE FAC_ID = 'ICPRB'
    """
    from core.db import fetch_all
    sql = custom_sql or _SELECT_MAX_KEY_SQL.format(table=table)
    rows = fetch_all(conn, sql)
    if not rows or rows[0][0] is None:
        return None
    return rows[0][0]


def load_problem_from_oracle(
    conn,
    table: str,
    rule_timekey: str,
    tool_groups: Optional[Dict[str, List[str]]] = None,
    custom_sql: Optional[str] = None,
) -> SchedulingProblem:
    """Oracle RTS_LINEDSDB_INF의 한 RULE_TIMEKEY를 SchedulingProblem으로 피벗.

    GBN_CD 값에 따라 7개 record 종류로 분류:
        WIP_QTY → WipRecord
        UPH     → UphRecord
        ASSIGN_EQUIP_CNT → EquipmentRecord
        D0_TARGET_QTY, D1_TARGET_QTY → PlanRecord (합산)
        TOOL_QTY → ToolQtyRecord
        availability/tool_group은 UPH 존재 + batch_id 매핑으로 유도.

    Args:
        conn: oracledb 연결 객체.
        table: 기본 쿼리에 채워지는 테이블명 ("RTS_LINEDSDB_INF" 등).
            custom_sql 지정 시 무시됨.
        rule_timekey: 조회 시각 키 (bind: :rule_timekey).
        tool_groups: {그룹명: [batch1, batch2, ...]} (config에서 주입).
        custom_sql: 사용자가 직접 작성한 쿼리. 반드시 다음 8개 컬럼을
            **이 순서**로 반환해야 한다:
                RULE_TIMEKEY, BATCH_ID, PLAN_PROD_KEY, OPER_ID, OPER_SEQ,
                EQP_MODEL_CD, GBN_CD, ATTR_VAL
            그리고 `:rule_timekey` bind를 사용해야 한다.

    Returns:
        SchedulingProblem.

    Example (기본):
        conn = connect("dispatcher", "dispatcher", "localhost:1521/XEPDB1")
        problem = load_problem_from_oracle(
            conn, "RTS_LINEDSDB_INF", "2026051707000000",
            tool_groups={"G001": ["9C/92", "9C/102"]},
        )

    Example (사용자 쿼리 — 뷰/조인/필터링 가능):
        my_sql = '''
        SELECT RULE_TIMEKEY, BATCH_ID, PLAN_PROD_KEY, OPER_ID, OPER_SEQ,
               EQP_MODEL_CD, GBN_CD, ATTR_VAL
          FROM RTS_LINEDSDB_INF
         WHERE RULE_TIMEKEY = :rule_timekey
           AND FAC_ID = 'ICPRB'
        '''
        problem = load_problem_from_oracle(
            conn, "", "2026051707000000",
            tool_groups={"G001": ["9C/92", "9C/102"]},
            custom_sql=my_sql,
        )
    """
    from core.db import fetch_all
    sql = custom_sql or _SELECT_SNAPSHOT_SQL.format(table=table)
    rows = fetch_all(conn, sql, {"rule_timekey": rule_timekey})
    return _rows_to_problem(rule_timekey, rows, tool_groups or {})


def _rows_to_problem(
    rule_timekey: str,
    rows: Iterable[Tuple],
    tool_groups: Dict[str, List[str]],
) -> SchedulingProblem:
    wip: List[WipRecord] = []
    uph: List[UphRecord] = []
    equipment_map: Dict[Tuple[str, str], int] = {}
    plan_map: Dict[Tuple[str, str], float] = {}
    tool_qty: List[ToolQtyRecord] = []
    tool_groups_recs: List[ToolGroupRecord] = []
    seen_pko_batch: Dict[Tuple[str, str], str] = {}

    for row in rows:
        (rk, batch_id, plan_prod_key, oper_id, oper_seq,
         eqp_model_cd, gbn_cd, attr_val) = row
        gbn = (gbn_cd or "").upper()
        if gbn == "WIP_QTY":
            wip.append(WipRecord(
                rule_timekey=rk, plan_prod_key=plan_prod_key, oper_id=oper_id,
                oper_seq=int(oper_seq or 0), wip_qty=float(attr_val or 0.0),
            ))
            seen_pko_batch[(plan_prod_key, oper_id)] = batch_id
        elif gbn == "UPH":
            uph.append(UphRecord(
                rule_timekey=rk, plan_prod_key=plan_prod_key, oper_id=oper_id,
                eqp_model_cd=eqp_model_cd, uph=float(attr_val or 0.0),
            ))
        elif gbn == "ASSIGN_EQUIP_CNT":
            key = (batch_id, eqp_model_cd)
            equipment_map[key] = equipment_map.get(key, 0) + int(float(attr_val or 0))
        elif gbn in ("D0_TARGET_QTY", "D1_TARGET_QTY"):
            key = (plan_prod_key, oper_id)
            plan_map[key] = plan_map.get(key, 0.0) + float(attr_val or 0.0)
        elif gbn == "TOOL_QTY":
            tool_qty.append(ToolQtyRecord(
                rule_timekey=rk, batch_id=batch_id, eqp_model_cd=eqp_model_cd,
                tool_qty=int(float(attr_val or 0)),
            ))

    for (pk, op), batch_id in seen_pko_batch.items():
        tool_groups_recs.append(ToolGroupRecord(
            rule_timekey=rule_timekey, batch_id=batch_id,
            plan_prod_key=pk, oper_id=op,
        ))

    equipment = [
        EquipmentRecord(rule_timekey=rule_timekey, batch_id=b, eqp_model_cd=m, eqp_qty=q)
        for (b, m), q in equipment_map.items()
    ]
    plans = [
        PlanRecord(rule_timekey=rule_timekey, plan_prod_key=pk, oper_id=op,
                   start_time=rule_timekey, end_time=rule_timekey, plan_qty=qty)
        for (pk, op), qty in plan_map.items()
    ]
    availability = [
        AvailabilityRecord(rule_timekey=rule_timekey, plan_prod_key=u.plan_prod_key,
                           oper_id=u.oper_id, eqp_model_cd=u.eqp_model_cd, avail_yn=u.uph > 0.0)
        for u in uph
    ]
    return SchedulingProblem(
        rule_timekey=rule_timekey,
        wip=wip,
        uph=uph,
        equipment=equipment,
        availability=availability,
        tool_groups=tool_groups_recs,
        tool_qty=tool_qty,
        plans=plans,
        eqp_model_groups=tool_groups,
    )
