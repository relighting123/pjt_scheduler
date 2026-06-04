"""가상 호기(모델×대수 확장) — 간트·SEQ 스케줄."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Tuple

from core.domain import Allocation, AllocationSet, SchedulingProblem


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


@dataclass(frozen=True)
class EqSlotAssignment:
    """한 호기·한 슬롯의 작업."""
    eqp_id: str
    batch_id: str
    eqp_model_cd: str
    plan_prod_key: str
    oper_id: str
    slot_index: int


@dataclass(frozen=True)
class SeqSegment:
    eqp_id: str
    plan_prod_key: str
    oper_id: str
    seq_no: int
    slot_start: int
    slot_end: int  # exclusive


def make_virtual_eqp_id(batch_id: str, eqp_model_cd: str, unit_index: int) -> str:
    return f"V-{eqp_model_cd}@{batch_id}#{unit_index:02d}"


def expand_allocation_to_virtual(allocation: AllocationSet) -> List[VirtualEqp]:
    """Allocation eqp_qty → 가상 호기 1대씩 (배치·모델 내 순번 유일)."""
    used: Dict[Tuple[str, str], int] = {}
    units: List[VirtualEqp] = []
    for a in allocation.allocations:
        key = (a.batch_id, a.eqp_model_cd)
        for _ in range(max(0, int(a.eqp_qty))):
            used[key] = used.get(key, 0) + 1
            idx = used[key]
            units.append(VirtualEqp(
                virtual_eqp_id=make_virtual_eqp_id(a.batch_id, a.eqp_model_cd, idx),
                batch_id=a.batch_id,
                eqp_model_cd=a.eqp_model_cd,
                plan_prod_key=a.plan_prod_key,
                oper_id=a.oper_id,
                unit_index=idx,
            ))
    return units


def expand_allocation_slot_assignments(
    allocation: AllocationSet,
) -> List[EqSlotAssignment]:
    """할당 1스냅샷 → 호기×슬롯용 (동일 스냅샷 전 구간 동일 작업)."""
    units = expand_allocation_to_virtual(allocation)
    return [
        EqSlotAssignment(
            eqp_id=u.virtual_eqp_id,
            batch_id=u.batch_id,
            eqp_model_cd=u.eqp_model_cd,
            plan_prod_key=u.plan_prod_key,
            oper_id=u.oper_id,
            slot_index=0,
        )
        for u in units
    ]


def build_gantt_segments(
    virtual_units: List[VirtualEqp],
    *,
    num_slots: int = 24,
) -> List[GanttSegment]:
    """가상 호기별 전 슬롯 동일 작업 막대."""
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


def build_eqp_slot_timelines(
    problem: SchedulingProblem,
    allocation: AllocationSet,
    settings: dict,
    *,
    mode: str,
) -> Dict[str, List[EqSlotAssignment]]:
    """호기 ID → 슬롯별 (pk, op) 리스트."""
    num_slots, _ = gantt_config(settings)
    if mode == "dynamic":
        schedule = _dynamic_schedule(problem, settings)
        return _timelines_from_slot_schedule(schedule, num_slots)
    return _timelines_static(allocation, num_slots)


def _timelines_static(
    allocation: AllocationSet,
    num_slots: int,
) -> Dict[str, List[EqSlotAssignment]]:
    units = expand_allocation_to_virtual(allocation)
    out: Dict[str, List[EqSlotAssignment]] = {}
    for u in units:
        out[u.virtual_eqp_id] = [
            EqSlotAssignment(
                eqp_id=u.virtual_eqp_id,
                batch_id=u.batch_id,
                eqp_model_cd=u.eqp_model_cd,
                plan_prod_key=u.plan_prod_key,
                oper_id=u.oper_id,
                slot_index=t,
            )
            for t in range(num_slots)
        ]
    return out


def _timelines_from_slot_schedule(
    schedule: List[AllocationSet],
    num_slots: int,
) -> Dict[str, List[EqSlotAssignment]]:
    """슬롯별 AllocationSet → 호기별 타임라인 (순번 고정)."""
    out: Dict[str, List[EqSlotAssignment]] = {}
    for slot_idx, alloc in enumerate(schedule[:num_slots]):
        for u in expand_allocation_to_virtual(alloc):
            if u.virtual_eqp_id not in out:
                out[u.virtual_eqp_id] = []
            # pad missing earlier slots with idle
            while len(out[u.virtual_eqp_id]) < slot_idx:
                prev = out[u.virtual_eqp_id][-1] if out[u.virtual_eqp_id] else None
                out[u.virtual_eqp_id].append(EqSlotAssignment(
                    eqp_id=u.virtual_eqp_id,
                    batch_id=u.batch_id,
                    eqp_model_cd=u.eqp_model_cd,
                    plan_prod_key="",
                    oper_id="",
                    slot_index=len(out[u.virtual_eqp_id]),
                ))
            out[u.virtual_eqp_id].append(EqSlotAssignment(
                eqp_id=u.virtual_eqp_id,
                batch_id=u.batch_id,
                eqp_model_cd=u.eqp_model_cd,
                plan_prod_key=u.plan_prod_key,
                oper_id=u.oper_id,
                slot_index=slot_idx,
            ))
    # pad to num_slots for all known eqps
    for eqp_id, timeline in list(out.items()):
        while len(timeline) < num_slots:
            last = timeline[-1] if timeline else None
            timeline.append(EqSlotAssignment(
                eqp_id=eqp_id,
                batch_id=last.batch_id if last else "",
                eqp_model_cd=last.eqp_model_cd if last else "",
                plan_prod_key="",
                oper_id="",
                slot_index=len(timeline),
            ))
    return out


def _dynamic_schedule(
    problem: SchedulingProblem,
    settings: dict,
) -> List[AllocationSet]:
    from core.sim.flow import MultiPeriodSimulator, dynamic_greedy_policy

    dyn = settings.get("dynamic", {})
    infer = settings.get("infer", {})
    num_slots = int(dyn.get("num_slots", infer.get("hours_per_day", 4)))
    slot_hours = float(dyn.get("slot_hours", infer.get("horizon_hours", 1.0)))
    switch = float(dyn.get("switch_time_hours", 0.0))
    sim = MultiPeriodSimulator(problem, num_slots, slot_hours, switch)
    result = sim.run(dynamic_greedy_policy)
    return result.schedule


def merge_timeline_to_seq_segments(
    slots: List[EqSlotAssignment],
    *,
    merge_key: Literal["plan_prod_key"] = "plan_prod_key",
) -> List[SeqSegment]:
    """연속 동일 제품(빈 슬롯 제외) → SEQ_NO 증가 구간."""
    segments: List[SeqSegment] = []
    seq_no = 0
    cur_pk = ""
    cur_op = ""
    cur_start = 0

    def flush(end_slot: int) -> None:
        nonlocal seq_no, cur_pk, cur_op, cur_start
        if not cur_pk:
            return
        segments.append(SeqSegment(
            eqp_id=slots[0].eqp_id if slots else "",
            plan_prod_key=cur_pk,
            oper_id=cur_op,
            seq_no=seq_no,
            slot_start=cur_start,
            slot_end=end_slot,
        ))

    for i, s in enumerate(slots):
        pk = s.plan_prod_key
        if not pk:
            if cur_pk:
                flush(i)
                cur_pk = ""
                cur_op = ""
            continue
        if pk != cur_pk:
            if cur_pk:
                flush(i)
            seq_no += 1
            cur_pk = pk
            cur_op = s.oper_id
            cur_start = i
        else:
            cur_op = s.oper_id or cur_op
    if cur_pk:
        flush(len(slots))
    return segments


def gantt_config(settings: dict) -> Tuple[int, float]:
    ve = settings.get("virtual_eqp", {})
    infer = settings.get("infer", {})
    slots = int(ve.get("gantt_slots", infer.get("hours_per_day", 24)))
    hours = float(ve.get("slot_hours", infer.get("horizon_hours", 1.0)))
    return slots, hours
