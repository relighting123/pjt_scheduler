"""가상 호기(모델×대수 확장) 및 호기별 제약."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from core.domain import Allocation, AllocationSet, SchedulingProblem

PkOp = Tuple[str, str]


@dataclass(frozen=True)
class VirtualEqp:
    """물리 호기 ID 없을 때 모델·배치·순번으로 만든 가상 호기."""

    virtual_eqp_id: str
    batch_id: str
    eqp_model_cd: str
    unit_index: int  # 1-based


@dataclass
class VirtualAssignment:
    virtual_eqp: VirtualEqp
    plan_prod_key: str
    oper_id: str
    allowed: bool
    block_reason: str = ""


@dataclass
class GanttSegment:
    virtual_eqp_id: str
    batch_id: str
    eqp_model_cd: str
    plan_prod_key: str
    oper_id: str
    slot_start: int
    slot_end: int  # exclusive
    allowed: bool
    block_reason: str = ""


def make_virtual_eqp_id(batch_id: str, eqp_model_cd: str, unit_index: int) -> str:
    return f"V-{eqp_model_cd}@{batch_id}#{unit_index:02d}"


def expand_virtual_equipment(
    problem: SchedulingProblem,
    allocation: AllocationSet,
) -> List[VirtualEqp]:
    """Allocation의 eqp_qty를 가상 호기 1대 단위로 펼침."""
    units: List[VirtualEqp] = []
    for a in allocation.allocations:
        for i in range(1, max(0, int(a.eqp_qty)) + 1):
            units.append(VirtualEqp(
                virtual_eqp_id=make_virtual_eqp_id(a.batch_id, a.eqp_model_cd, i),
                batch_id=a.batch_id,
                eqp_model_cd=a.eqp_model_cd,
                unit_index=i,
            ))
    return units


def _parse_unit_rules(settings: dict) -> Dict[str, List[Optional[Set[PkOp]]]]:
    """settings.virtual_eqp.unit_rules: { "T5833|9C/92": [ ["P1","OP10"], null, ... ] }"""
    raw = settings.get("virtual_eqp", {}).get("unit_rules", {})
    out: Dict[str, List[Optional[Set[PkOp]]]] = {}
    for key, spec in raw.items():
        if not isinstance(spec, list):
            continue
        rules: List[Optional[Set[PkOp]]] = []
        for item in spec:
            if item is None:
                rules.append(None)
            elif isinstance(item, list):
                rules.append({(str(p), str(o)) for p, o in item})
            else:
                rules.append(None)
        out[str(key)] = rules
    return out


def _allowed_pk_ops_for_unit(
    ve: VirtualEqp,
    unit_rules: Dict[str, List[Optional[Set[PkOp]]]],
) -> Optional[Set[PkOp]]:
    """None = 제한 없음, set = 허용 (pk,op)만."""
    pool_key = f"{ve.eqp_model_cd}|{ve.batch_id}"
    rules = unit_rules.get(pool_key)
    if not rules:
        return None
    idx = ve.unit_index - 1
    if idx < 0 or idx >= len(rules):
        return None
    return rules[idx]


def model_level_feasible(problem: SchedulingProblem, pk: str, op: str, model: str) -> Tuple[bool, str]:
    if not problem.is_available(pk, op, model):
        return False, "model/pk/op not available (UPH)"
    if problem.wip_of(pk, op) <= 0 and problem.plan_qty_of(pk, op) > 0:
        return False, "WIP=0 for target"
    return True, ""


def assign_virtual_units(
    problem: SchedulingProblem,
    allocation: AllocationSet,
    settings: Optional[dict] = None,
) -> List[VirtualAssignment]:
    """집계 Allocation → 가상 호기별 (pk,op) + 허용 여부."""
    settings = settings or {}
    unit_rules = _parse_unit_rules(settings)
    assignments: List[VirtualAssignment] = []

    for a in allocation.allocations:
        for i in range(1, max(0, int(a.eqp_qty)) + 1):
            ve = VirtualEqp(
                virtual_eqp_id=make_virtual_eqp_id(a.batch_id, a.eqp_model_cd, i),
                batch_id=a.batch_id,
                eqp_model_cd=a.eqp_model_cd,
                unit_index=i,
            )
            pk, op = a.plan_prod_key, a.oper_id
            allowed_set = _allowed_pk_ops_for_unit(ve, unit_rules)
            if allowed_set is not None and (pk, op) not in allowed_set:
                assignments.append(VirtualAssignment(
                    ve, pk, op, False, f"unit rule: only {allowed_set}",
                ))
                continue
            ok, reason = model_level_feasible(problem, pk, op, ve.eqp_model_cd)
            assignments.append(VirtualAssignment(ve, pk, op, ok, reason if not ok else ""))

    return assignments


def build_gantt_segments(
    virtual_assignments: List[VirtualAssignment],
    *,
    num_slots: int = 24,
    slot_hours: float = 1.0,
    rule_timekey: str = "",
) -> List[GanttSegment]:
    """가상 호기별 24슬롯(기본) 동일 작업 막대. 추후 슬롯별 schedule로 확장 가능."""
    segments: List[GanttSegment] = []
    for va in virtual_assignments:
        if not va.plan_prod_key:
            segments.append(GanttSegment(
                va.virtual_eqp.virtual_eqp_id,
                va.virtual_eqp.batch_id,
                va.virtual_eqp.eqp_model_cd,
                "", "", 0, num_slots, False, va.block_reason or "idle",
            ))
            continue
        segments.append(GanttSegment(
            virtual_eqp_id=va.virtual_eqp.virtual_eqp_id,
            batch_id=va.virtual_eqp.batch_id,
            eqp_model_cd=va.virtual_eqp.eqp_model_cd,
            plan_prod_key=va.plan_prod_key,
            oper_id=va.oper_id,
            slot_start=0,
            slot_end=num_slots,
            allowed=va.allowed,
            block_reason=va.block_reason,
        ))
    return segments


def gantt_config(settings: dict) -> Tuple[int, float]:
    ve = settings.get("virtual_eqp", {})
    infer = settings.get("infer", {})
    slots = int(ve.get("gantt_slots", infer.get("hours_per_day", 24)))
    hours = float(ve.get("slot_hours", infer.get("horizon_hours", 1.0)))
    return slots, hours
