"""JSON snapshot helpers for SchedulingProblem (infer debugging).

Oracle/CSV에서 로드한 문제를 JSON 파일로 저장한 뒤 다시 읽어 동일한
SchedulingProblem으로 복원할 수 있다. infer 시 로그·원인 분석용.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from core.domain import (
    Allocation,
    AllocationSet,
    AvailabilityRecord,
    EquipmentRecord,
    PlanRecord,
    SchedulingProblem,
    ToolGroupRecord,
    ToolQtyRecord,
    UphRecord,
    WipRecord,
)


def problem_to_dict(problem: SchedulingProblem) -> Dict[str, Any]:
    return {
        "rule_timekey": problem.rule_timekey,
        "wip": [asdict(w) for w in problem.wip],
        "uph": [asdict(u) for u in problem.uph],
        "equipment": [asdict(e) for e in problem.equipment],
        "availability": [asdict(a) for a in problem.availability],
        "tool_groups": [asdict(t) for t in problem.tool_groups],
        "tool_qty": [asdict(t) for t in problem.tool_qty],
        "plans": [asdict(p) for p in problem.plans],
        "eqp_model_groups": dict(problem.eqp_model_groups),
    }


def problem_from_dict(data: Dict[str, Any]) -> SchedulingProblem:
    return SchedulingProblem(
        rule_timekey=data["rule_timekey"],
        wip=[WipRecord(**w) for w in data.get("wip", [])],
        uph=[UphRecord(**u) for u in data.get("uph", [])],
        equipment=[EquipmentRecord(**e) for e in data.get("equipment", [])],
        availability=[AvailabilityRecord(**a) for a in data.get("availability", [])],
        tool_groups=[ToolGroupRecord(**t) for t in data.get("tool_groups", [])],
        tool_qty=[ToolQtyRecord(**t) for t in data.get("tool_qty", [])],
        plans=[PlanRecord(**p) for p in data.get("plans", [])],
        eqp_model_groups=dict(data.get("eqp_model_groups", {})),
    )


def allocation_set_to_dict(allocation: AllocationSet) -> Dict[str, Any]:
    return {
        "rule_timekey": allocation.rule_timekey,
        "allocations": [asdict(a) for a in allocation.allocations],
    }


def allocation_set_from_dict(data: Dict[str, Any]) -> AllocationSet:
    return AllocationSet(
        rule_timekey=data["rule_timekey"],
        allocations=[Allocation(**a) for a in data.get("allocations", [])],
    )


def problem_input_summary(problem: SchedulingProblem) -> Dict[str, Any]:
    """infer 결과 JSON에 넣을 입력 요약."""
    return {
        "wip_rows": len(problem.wip),
        "uph_rows": len(problem.uph),
        "plan_rows": len(problem.plans),
        "equipment_rows": len(problem.equipment),
        "availability_rows": len(problem.availability),
        "tool_group_rows": len(problem.tool_groups),
        "tool_qty_rows": len(problem.tool_qty),
        "plan_targets": len(problem.plan_targets()),
        "equipment_pool_size": len(problem.equipment_pool()),
    }


def dump_infer_snapshot(
    path: str | Path,
    problem: SchedulingProblem,
    *,
    mode: str,
    source: str,
) -> Path:
    """SchedulingProblem을 JSON 파일로 저장."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "rule_timekey": problem.rule_timekey,
        "mode": mode,
        "source": source,
        "input_summary": problem_input_summary(problem),
        "problem": problem_to_dict(problem),
    }
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def load_infer_snapshot(path: str | Path) -> Tuple[SchedulingProblem, Dict[str, Any]]:
    """JSON 스냅샷에서 SchedulingProblem과 메타데이터를 복원."""
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    problem = problem_from_dict(payload["problem"])
    meta = {k: v for k, v in payload.items() if k != "problem"}
    return problem, meta


def resolve_snapshot_path(
    settings: dict,
    rule_timekey: str,
    mode: str,
    override: Optional[str] = None,
) -> Path:
    if override:
        return Path(override)
    infer_cfg = settings.get("infer", {})
    snap_dir = Path(infer_cfg.get("snapshot_dir", "artifacts/inference/snapshots"))
    return snap_dir / f"{rule_timekey}_{mode}.json"
