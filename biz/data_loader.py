"""Adapters that materialize `SchedulingProblem` from CSV benchmarks or Oracle.

Oracle input uses one SQL file per category (`config/queries/wip.sql`, …).
Each file returns typed columns; Python maps rows to records and assembles
the problem. availability / tool_group are derived from UPH and WIP batch map.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

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
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


# ---------------------------------------------------------------------------
# Oracle loader — one .sql file per input category
# ---------------------------------------------------------------------------
_QUERY_FILES = {
    "range_keys": "range_keys.sql",
    "latest_key": "latest_key.sql",
    "wip":        "wip.sql",
    "uph":        "uph.sql",
    "equipment":  "equipment.sql",
    "plan":       "plan.sql",
    "tool_qty":   "tool_qty.sql",
}

_INPUT_QUERY_KINDS = ("wip", "uph", "equipment", "plan", "tool_qty")

_DEFAULT_FAC_ID = "CJPRB"


def resolve_fac_id(settings: Optional[dict] = None, override: Optional[str] = None) -> str:
    """Oracle 쿼리 FAC_ID bind 값 (settings.oracle.fac_id, 기본 CJPRB)."""
    if override:
        return override
    if settings:
        return str(settings.get("oracle", {}).get("fac_id", _DEFAULT_FAC_ID))
    return _DEFAULT_FAC_ID


def oracle_query_params(
    settings: Optional[dict] = None,
    *,
    fac_id: Optional[str] = None,
    **binds: Any,
) -> Dict[str, Any]:
    params: Dict[str, Any] = {"fac_id": resolve_fac_id(settings, fac_id)}
    params.update(binds)
    return params


def load_sql(query_dir: Optional[str], kind: str) -> str:
    """Read `query_dir/<kind>.sql`. kind ∈ range_keys, latest_key, wip, uph, …"""
    if not query_dir:
        raise FileNotFoundError(
            f"oracle.query_dir not set; cannot resolve '{kind}' query."
        )
    if kind not in _QUERY_FILES:
        raise KeyError(f"Unknown query kind: {kind!r}")
    path = Path(query_dir) / _QUERY_FILES[kind]
    if not path.exists():
        raise FileNotFoundError(f"Query file not found: {path}")
    return path.read_text(encoding="utf-8")


def input_query_kinds() -> Tuple[str, ...]:
    return _INPUT_QUERY_KINDS


def list_rule_timekeys(
    conn,
    query_dir: Optional[str],
    from_key: str,
    to_key: str,
    settings: Optional[dict] = None,
    *,
    fac_id: Optional[str] = None,
) -> List[str]:
    """`range_keys.sql`로 구간 안의 distinct RULE_TIMEKEY 목록을 반환.

    Example:
        keys = list_rule_timekeys(conn, "config/queries",
                                  "20251020000000", "20251027000000")
        # → ["20251020070000", "20251020080000", ...]
    """
    from core.db import fetch_all
    sql = load_sql(query_dir, "range_keys")
    params = oracle_query_params(
        settings, fac_id=fac_id, from_key=from_key, to_key=to_key,
    )
    rows = fetch_all(conn, sql, params)
    return [r[0] for r in rows]


def latest_rule_timekey(
    conn,
    query_dir: Optional[str],
    settings: Optional[dict] = None,
    *,
    fac_id: Optional[str] = None,
) -> Optional[str]:
    """`latest_key.sql`로 MAX(RULE_TIMEKEY)를 반환. 결과 없으면 None.

    Example:
        rk = latest_rule_timekey(conn, "config/queries")
        # → "20251027060000"
    """
    from core.db import fetch_all
    sql = load_sql(query_dir, "latest_key")
    params = oracle_query_params(settings, fac_id=fac_id)
    rows = fetch_all(conn, sql, params)
    if not rows or rows[0][0] is None:
        return None
    return rows[0][0]


def load_problem_from_oracle(
    conn,
    query_dir: Optional[str],
    rule_timekey: str,
    tool_groups: Optional[Dict[str, List[str]]] = None,
    settings: Optional[dict] = None,
    *,
    fac_id: Optional[str] = None,
) -> SchedulingProblem:
    """Oracle에서 항목별 SQL을 실행해 SchedulingProblem을 조립."""
    if not query_dir:
        raise FileNotFoundError("oracle.query_dir not set.")
    groups = tool_groups or {}
    params = oracle_query_params(
        settings, fac_id=fac_id, rule_timekey=rule_timekey,
    )
    return _load_problem_from_queries(
        conn, query_dir, rule_timekey, groups, params,
    )


def _fetch_query(
    conn, query_dir: str, kind: str, params: Dict[str, Any],
) -> List[Tuple]:
    from core.db import fetch_all
    sql = load_sql(query_dir, kind)
    return fetch_all(conn, sql, params)


def _map_wip_rows(rule_timekey: str, rows: Iterable[Tuple]) -> Tuple[List[WipRecord], Dict[Tuple[str, str], str]]:
    wip: List[WipRecord] = []
    seen_pko_batch: Dict[Tuple[str, str], str] = {}
    for row in rows:
        rk, batch_id, plan_prod_key, oper_id, oper_seq, wip_qty = row
        wip.append(WipRecord(
            rule_timekey=rk or rule_timekey,
            plan_prod_key=plan_prod_key,
            oper_id=oper_id,
            oper_seq=int(oper_seq or 0),
            wip_qty=float(wip_qty or 0.0),
        ))
        seen_pko_batch[(plan_prod_key, oper_id)] = batch_id
    return wip, seen_pko_batch


def _map_uph_rows(rule_timekey: str, rows: Iterable[Tuple]) -> List[UphRecord]:
    return [
        UphRecord(
            rule_timekey=rk or rule_timekey,
            plan_prod_key=plan_prod_key,
            oper_id=oper_id,
            eqp_model_cd=eqp_model_cd,
            uph=float(uph or 0.0),
        )
        for rk, plan_prod_key, oper_id, eqp_model_cd, uph in rows
    ]


def _map_equipment_rows(
    rule_timekey: str, rows: Iterable[Tuple],
) -> Dict[Tuple[str, str], int]:
    equipment_map: Dict[Tuple[str, str], int] = {}
    for rk, batch_id, eqp_model_cd, eqp_qty in rows:
        key = (batch_id, eqp_model_cd)
        equipment_map[key] = equipment_map.get(key, 0) + int(float(eqp_qty or 0))
    return equipment_map


def _map_plan_rows(
    rule_timekey: str, rows: Iterable[Tuple],
) -> Dict[Tuple[str, str], float]:
    plan_map: Dict[Tuple[str, str], float] = {}
    for rk, plan_prod_key, oper_id, _start, _end, plan_qty in rows:
        key = (plan_prod_key, oper_id)
        plan_map[key] = plan_map.get(key, 0.0) + float(plan_qty or 0.0)
    return plan_map


def _map_tool_qty_rows(rule_timekey: str, rows: Iterable[Tuple]) -> List[ToolQtyRecord]:
    return [
        ToolQtyRecord(
            rule_timekey=rk or rule_timekey,
            batch_id=batch_id,
            eqp_model_cd=eqp_model_cd,
            tool_qty=int(float(tool_qty or 0)),
        )
        for rk, batch_id, eqp_model_cd, tool_qty in rows
    ]


def _load_problem_from_queries(
    conn,
    query_dir: str,
    rule_timekey: str,
    tool_groups: Dict[str, List[str]],
    params: Dict[str, Any],
) -> SchedulingProblem:
    wip, seen_pko_batch = _map_wip_rows(
        rule_timekey, _fetch_query(conn, query_dir, "wip", params),
    )
    uph = _map_uph_rows(
        rule_timekey, _fetch_query(conn, query_dir, "uph", params),
    )
    equipment_map = _map_equipment_rows(
        rule_timekey, _fetch_query(conn, query_dir, "equipment", params),
    )
    plan_map = _map_plan_rows(
        rule_timekey, _fetch_query(conn, query_dir, "plan", params),
    )
    tool_qty = _map_tool_qty_rows(
        rule_timekey, _fetch_query(conn, query_dir, "tool_qty", params),
    )
    return _assemble_scheduling_problem(
        rule_timekey, wip, uph, equipment_map, plan_map,
        tool_qty, seen_pko_batch, tool_groups,
    )


def _assemble_scheduling_problem(
    rule_timekey: str,
    wip: List[WipRecord],
    uph: List[UphRecord],
    equipment_map: Dict[Tuple[str, str], int],
    plan_map: Dict[Tuple[str, str], float],
    tool_qty: List[ToolQtyRecord],
    seen_pko_batch: Dict[Tuple[str, str], str],
    tool_groups: Dict[str, List[str]],
) -> SchedulingProblem:
    tool_groups_recs = [
        ToolGroupRecord(
            rule_timekey=rule_timekey, batch_id=batch_id,
            plan_prod_key=pk, oper_id=op,
        )
        for (pk, op), batch_id in seen_pko_batch.items()
    ]
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
