"""가상 호기(모델×대수 확장) — 간트 표시용."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from core.domain import AllocationSet, SchedulingProblem


@dataclass(frozen=True)
class VirtualEqp:
    virtual_eqp_id: str
    batch_id: str
    eqp_model_cd: str
    plan_prod_key: str
    oper_id: str
    unit_index: int


@dataclass
class GanttSegment:
    virtual_eqp_id: str
    batch_id: str
    eqp_model_cd: str
    plan_prod_key: str
    oper_id: str
    slot_start: int
    slot_end: int  # exclusive


def make_virtual_eqp_id(batch_id: str, eqp_model_cd: str, unit_index: int) -> str:
    return f"V-{eqp_model_cd}@{batch_id}#{unit_index:02d}"


def expand_allocation_to_virtual(allocation: AllocationSet) -> List[VirtualEqp]:
    """Allocation eqp_qty → 가상 호기 1대씩."""
    units: List[VirtualEqp] = []
    for a in allocation.allocations:
        for i in range(1, max(0, int(a.eqp_qty)) + 1):
            units.append(VirtualEqp(
                virtual_eqp_id=make_virtual_eqp_id(a.batch_id, a.eqp_model_cd, i),
                batch_id=a.batch_id,
                eqp_model_cd=a.eqp_model_cd,
                plan_prod_key=a.plan_prod_key,
                oper_id=a.oper_id,
                unit_index=i,
            ))
    return units


def build_gantt_segments(
    virtual_units: List[VirtualEqp],
    *,
    num_slots: int = 24,
) -> List[GanttSegment]:
    """가상 호기별 24슬롯 동일 작업 막대."""
    return [
        GanttSegment(
            virtual_eqp_id=u.virtual_eqp_id,
            batch_id=u.batch_id,
            eqp_model_cd=u.eqp_model_cd,
            plan_prod_key=u.plan_prod_key,
            oper_id=u.oper_id,
            slot_start=0,
            slot_end=num_slots,
        )
        for u in virtual_units
    ]


def gantt_config(settings: dict) -> Tuple[int, float]:
    ve = settings.get("virtual_eqp", {})
    infer = settings.get("infer", {})
    slots = int(ve.get("gantt_slots", infer.get("hours_per_day", 24)))
    hours = float(ve.get("slot_hours", infer.get("horizon_hours", 1.0)))
    return slots, hours
