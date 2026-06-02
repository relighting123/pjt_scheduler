"""Scheduling simulator.

Given a `SchedulingProblem` and an `AllocationSet`, compute production per
(plan_prod_key, oper_id) and the achievement rate vs. plan. The output is the
canonical reward signal for RL and the evaluation metric for benchmarks.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from .domain import AllocationSet, SchedulingProblem


@dataclass
class SimulationResult:
    produced_by_pko: Dict[Tuple[str, str], float] = field(default_factory=dict)
    plan_by_pko: Dict[Tuple[str, str], float] = field(default_factory=dict)
    achievement_by_pko: Dict[Tuple[str, str], float] = field(default_factory=dict)
    avg_achievement: float = 0.0
    over_allocations: List[str] = field(default_factory=list)

    def __str__(self) -> str:  # pragma: no cover
        return f"avg_achievement={self.avg_achievement:.4f}, items={len(self.achievement_by_pko)}"


class Simulator:
    """Horizon-agnostic single-step production simulator.

    The plan covers a fixed horizon (per RULE_TIMEKEY). UPH is per hour and the
    horizon is normalized to 1.0 — benchmarks express PLAN_QTY in the same unit
    so the comparison is consistent. Sub-hour horizons can be reflected by
    scaling PLAN_QTY in biz layer.
    """

    def __init__(self, problem: SchedulingProblem, horizon_hours: float = 1.0) -> None:
        self.problem = problem
        self.horizon_hours = float(horizon_hours)

    def simulate(self, allocations: AllocationSet) -> SimulationResult:
        problem = self.problem
        result = SimulationResult()
        result.plan_by_pko = {(pk, op): qty for pk, op, qty in problem.plan_targets()}

        # validate equipment pool isn't overcommitted per (batch, model)
        pool = problem.equipment_pool()
        used: Dict[Tuple[str, str], int] = {}
        for alloc in allocations.allocations:
            key = (alloc.batch_id, alloc.eqp_model_cd)
            used[key] = used.get(key, 0) + max(0, int(alloc.eqp_qty))
        for key, qty in used.items():
            if qty > pool.get(key, 0):
                result.over_allocations.append(f"over-alloc {key}: {qty} > {pool.get(key, 0)}")

        # production = sum(eqp_qty * uph * horizon) limited by remaining plan
        produced: Dict[Tuple[str, str], float] = {}
        for alloc in allocations.allocations:
            if not problem.is_available(alloc.plan_prod_key, alloc.oper_id, alloc.eqp_model_cd):
                continue
            uph = problem.uph_of(alloc.plan_prod_key, alloc.oper_id, alloc.eqp_model_cd)
            qty = uph * max(0, int(alloc.eqp_qty)) * self.horizon_hours
            pk_op = (alloc.plan_prod_key, alloc.oper_id)
            produced[pk_op] = produced.get(pk_op, 0.0) + qty

        # achievement capped at 1.0; missing keys count as 0 achievement
        total = 0.0
        counted = 0
        for pk_op, plan_qty in result.plan_by_pko.items():
            actual = produced.get(pk_op, 0.0)
            if plan_qty <= 0:
                rate = 1.0 if actual >= 0 else 0.0
            else:
                rate = min(1.0, actual / plan_qty)
            result.produced_by_pko[pk_op] = actual
            result.achievement_by_pko[pk_op] = rate
            total += rate
            counted += 1
        result.avg_achievement = (total / counted) if counted else 0.0
        return result


def count_switches(previous: AllocationSet | None, current: AllocationSet) -> int:
    """Count tool conversions vs. previous allocation.

    A switch is a (batch, model) pair whose equipment count increased compared
    to the previous snapshot — i.e. equipment moved into this batch.
    """
    if previous is None:
        return 0
    prev_by_bm: Dict[Tuple[str, str], int] = {}
    for a in previous.allocations:
        prev_by_bm[(a.batch_id, a.eqp_model_cd)] = prev_by_bm.get((a.batch_id, a.eqp_model_cd), 0) + a.eqp_qty
    cur_by_bm: Dict[Tuple[str, str], int] = {}
    for a in current.allocations:
        cur_by_bm[(a.batch_id, a.eqp_model_cd)] = cur_by_bm.get((a.batch_id, a.eqp_model_cd), 0) + a.eqp_qty
    switches = 0
    for key, qty in cur_by_bm.items():
        diff = qty - prev_by_bm.get(key, 0)
        if diff > 0:
            switches += diff
    return switches
