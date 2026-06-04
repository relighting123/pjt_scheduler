"""Post-infer summary: per-target achievement, eqp counts, daily capacity."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from core.domain import AllocationSet, SchedulingProblem
from core.sim.simulator import Simulator


def _infer_horizon_hours(settings: dict, mode: str) -> float:
    if mode == "dynamic":
        dyn = settings.get("dynamic", {})
        return float(dyn.get("slot_hours", 1.0))
    return float(settings.get("infer", {}).get("horizon_hours", 1.0))


def _hours_per_day(settings: dict) -> float:
    return float(settings.get("infer", {}).get("hours_per_day", 24.0))


def build_infer_report(
    problem: SchedulingProblem,
    allocation: AllocationSet,
    settings: dict,
    mode: str,
) -> Dict[str, Any]:
    """(batch, pk, op)별 달성률·모델별 댓수·일 capa 합산."""
    horizon = _infer_horizon_hours(settings, mode)
    hours_day = _hours_per_day(settings)
    ignore_wip = mode == "plan-only"
    sim = Simulator(problem, horizon_hours=horizon, ignore_wip=ignore_wip)
    sim_result = sim.simulate(allocation)

    targets: List[Dict[str, Any]] = []
    total_daily_capacity = 0.0

    for pk, op, plan_qty in sorted(problem.plan_targets()):
        batch_id = problem.batch_of(pk, op)
        eqp_by_model: Dict[str, int] = {}
        daily_capacity = 0.0
        for a in allocation.allocations:
            if a.plan_prod_key != pk or a.oper_id != op:
                continue
            qty = max(0, int(a.eqp_qty))
            if qty == 0:
                continue
            eqp_by_model[a.eqp_model_cd] = eqp_by_model.get(a.eqp_model_cd, 0) + qty
            uph = problem.uph_of(pk, op, a.eqp_model_cd)
            daily_capacity += uph * qty * hours_day

        pk_op = (pk, op)
        targets.append({
            "batch_id": batch_id,
            "plan_prod_key": pk,
            "oper_id": op,
            "plan_qty": float(plan_qty),
            "produced_qty": float(sim_result.produced_by_pko.get(pk_op, 0.0)),
            "achievement_rate": float(sim_result.achievement_by_pko.get(pk_op, 0.0)),
            "eqp_qty_by_model": eqp_by_model,
            "daily_capacity": daily_capacity,
        })
        total_daily_capacity += daily_capacity

    return {
        "horizon_hours": horizon,
        "hours_per_day": hours_day,
        "avg_achievement": float(sim_result.avg_achievement),
        "total_daily_capacity": total_daily_capacity,
        "targets": targets,
    }


def format_infer_report_log(report: Dict[str, Any]) -> str:
    """운영 로그용 텍스트."""
    lines = [
        "--- infer report ---",
        f"avg_achievement: {report['avg_achievement']:.4f}",
        f"total_daily_capacity: {report['total_daily_capacity']:.1f}",
        "batch_id | plan_prod_key | oper_id | achv | eqp_by_model | daily_capa",
    ]
    for t in report.get("targets", []):
        models = ", ".join(
            f"{m}:{q}" for m, q in sorted(t.get("eqp_qty_by_model", {}).items())
        ) or "-"
        lines.append(
            f"{t.get('batch_id', '')} | {t['plan_prod_key']} | {t['oper_id']} | "
            f"{t['achievement_rate']:.4f} | [{models}] | {t['daily_capacity']:.1f}"
        )
    return "\n".join(lines)
