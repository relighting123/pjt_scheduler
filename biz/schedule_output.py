"""호기별 SEQ 스케줄 행 (START_TIME/END_TIME) 생성·CSV 출력."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from core.domain import AllocationSet, SchedulingProblem

from .timekey_util import add_hours_to_timekey, resolve_horizon_start
from .virtual_eqp import (
    build_eqp_slot_timelines,
    gantt_config,
    merge_timeline_to_seq_segments,
)

SCHEDULE_COLUMNS: Tuple[str, ...] = (
    "RULE_TIMEKEY",
    "EQP_ID",
    "PLAN_PROD_KEY",
    "SEQ_NO",
    "START_TIME",
    "END_TIME",
    "PRODUCE_QTY",
    "CRT_USER_ID",
    "CRT_TM",
)


@dataclass(frozen=True)
class ScheduleSeqRow:
    rule_timekey: str
    eqp_id: str
    plan_prod_key: str
    seq_no: int
    start_time: str
    end_time: str
    produce_qty: float
    oper_id: str = ""  # UPH 계산용 (CSV에는 미출력)

    def as_tuple(self, crt_user_id: str, crt_tm: str) -> Tuple:
        return (
            self.rule_timekey,
            self.eqp_id,
            self.plan_prod_key,
            self.seq_no,
            self.start_time,
            self.end_time,
            round(self.produce_qty, 4),
            crt_user_id,
            crt_tm,
        )


def _produce_qty(
    problem: SchedulingProblem,
    *,
    plan_prod_key: str,
    oper_id: str,
    eqp_model_cd: str,
    duration_hours: float,
) -> float:
    uph = problem.uph_of(plan_prod_key, oper_id, eqp_model_cd)
    return max(0.0, uph * duration_hours)


def build_schedule_rows(
    problem: SchedulingProblem,
    allocation: AllocationSet,
    settings: dict,
    *,
    mode: str = "wip-static",
    crt_user_id: Optional[str] = None,
    crt_tm: Optional[str] = None,
) -> List[ScheduleSeqRow]:
    """가상 호기·슬롯 타임라인 → SEQ 병합 행.

    동일 EQP_ID에서 PLAN_PROD_KEY가 연속이면 SEQ_NO 유지, 바뀌면 +1.
    """
    infer_cfg = settings.get("infer", {})
    num_slots, slot_hours = gantt_config(settings)
    horizon_start = resolve_horizon_start(
        problem.rule_timekey,
        fallback=_first_plan_start(problem),
    )
    timelines = build_eqp_slot_timelines(
        problem, allocation, settings, mode=mode,
    )
    rows: List[ScheduleSeqRow] = []
    for eqp_id, slots in timelines.items():
        if not slots:
            continue
        model_cd = slots[0].eqp_model_cd
        for seg in merge_timeline_to_seq_segments(slots, merge_key="plan_prod_key"):
            if not seg.plan_prod_key:
                continue
            start_t = add_hours_to_timekey(horizon_start, seg.slot_start * slot_hours)
            end_t = add_hours_to_timekey(horizon_start, seg.slot_end * slot_hours)
            dur = (seg.slot_end - seg.slot_start) * slot_hours
            qty = _produce_qty(
                problem,
                plan_prod_key=seg.plan_prod_key,
                oper_id=seg.oper_id,
                eqp_model_cd=model_cd,
                duration_hours=dur,
            )
            rows.append(ScheduleSeqRow(
                rule_timekey=problem.rule_timekey,
                eqp_id=eqp_id,
                plan_prod_key=seg.plan_prod_key,
                seq_no=seg.seq_no,
                start_time=start_t,
                end_time=end_t,
                produce_qty=qty,
                oper_id=seg.oper_id,
            ))
    rows.sort(key=lambda r: (r.eqp_id, r.seq_no))
    return rows


def format_crt_tm(dt: Optional[datetime] = None) -> str:
    from .timekey_util import format_timekey
    return format_timekey(dt or datetime.now())


def _first_plan_start(problem: SchedulingProblem) -> Optional[str]:
    for p in problem.plans:
        if p.start_time:
            return p.start_time
    return None


def write_schedule_csv(
    path: str | Path,
    rows: Sequence[ScheduleSeqRow],
    *,
    crt_user_id: str = "SCHEDULER",
    crt_tm: Optional[str] = None,
) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tm = crt_tm or format_crt_tm()
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(SCHEDULE_COLUMNS)
        for r in rows:
            w.writerow(r.as_tuple(crt_user_id, tm))
    return str(path)


def resolve_schedule_path(settings: dict, rule_timekey: str, mode: str) -> Path:
    infer = settings.get("infer", {})
    out_dir = Path(infer.get("schedule_dir", infer.get("report_dir", "artifacts/reports")))
    return out_dir / f"schedule_{rule_timekey}_{mode}.csv"
