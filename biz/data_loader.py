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
    """Load a benchmark dataset from a directory containing the 7 CSVs."""
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


def list_rule_timekeys(conn, table: str, from_key: str, to_key: str) -> List[str]:
    from core.db import fetch_all
    rows = fetch_all(conn, _SELECT_RANGE_KEYS_SQL.format(table=table),
                     {"from_key": from_key, "to_key": to_key})
    return [r[0] for r in rows]


def latest_rule_timekey(conn, table: str) -> Optional[str]:
    from core.db import fetch_all
    rows = fetch_all(conn, _SELECT_MAX_KEY_SQL.format(table=table))
    if not rows or rows[0][0] is None:
        return None
    return rows[0][0]


def load_problem_from_oracle(
    conn,
    table: str,
    rule_timekey: str,
    tool_groups: Optional[Dict[str, List[str]]] = None,
) -> SchedulingProblem:
    """Pivot RTS_LINEDSDB_INF rows for a single RULE_TIMEKEY into the domain."""
    from core.db import fetch_all
    rows = fetch_all(conn, _SELECT_SNAPSHOT_SQL.format(table=table), {"rule_timekey": rule_timekey})
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
