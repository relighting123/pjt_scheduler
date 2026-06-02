"""Greedy heuristic baseline.

Allocates equipment per (batch_id, eqp_model_cd) to the (plan_prod_key, oper_id)
with the highest remaining shortfall * UPH. Respects tool-group restrictions —
units stay within the same batch unless an explicit tool-group conversion is
beneficial. Used as the imitation-learning teacher and a baseline.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from .domain import Allocation, AllocationSet, SchedulingProblem


def greedy_allocate(problem: SchedulingProblem) -> AllocationSet:
    pool: Dict[Tuple[str, str], int] = dict(problem.equipment_pool())
    remaining: Dict[Tuple[str, str], float] = {
        (pk, op): qty for pk, op, qty in problem.plan_targets()
    }
    # WIP cap: each target can produce at most the WIP currently in queue.
    # WIP <= 0 in the records means "unconstrained" (treated as infinite).
    wip_remaining: Dict[Tuple[str, str], float] = {}
    for (pk, op) in remaining:
        w = problem.wip_of(pk, op)
        wip_remaining[(pk, op)] = w if w > 0 else float("inf")
    allocations: List[Allocation] = []

    candidates: List[Tuple[str, str]] = list(remaining.keys())
    # iterate until no positive marginal allocation remains
    while True:
        best = None
        best_score = 0.0
        for plan_prod_key, oper_id in candidates:
            need = remaining.get((plan_prod_key, oper_id), 0.0)
            wip_left = wip_remaining.get((plan_prod_key, oper_id), 0.0)
            if need <= 0 or wip_left <= 0:
                continue
            batch_id = problem.batch_of(plan_prod_key, oper_id)
            for (b_id, model), free in pool.items():
                if free <= 0:
                    continue
                if not problem.is_available(plan_prod_key, oper_id, model):
                    continue
                # restrict conversions: same batch always OK; cross-batch only
                # if the model belongs to a defined tool-group with the target.
                if b_id != batch_id:
                    group = problem.model_group_of(model)
                    if not group:
                        continue
                uph = problem.uph_of(plan_prod_key, oper_id, model)
                # marginal contribution of one unit, bounded by plan need AND
                # remaining WIP at this op.
                score = min(need, uph, wip_left)
                if score > best_score:
                    best_score = score
                    best = (plan_prod_key, oper_id, b_id, model, uph)
        if best is None:
            break
        plan_prod_key, oper_id, b_id, model, uph = best
        pool[(b_id, model)] -= 1
        remaining[(plan_prod_key, oper_id)] = max(0.0, remaining[(plan_prod_key, oper_id)] - uph)
        wip_remaining[(plan_prod_key, oper_id)] = max(0.0, wip_remaining[(plan_prod_key, oper_id)] - uph)
        # consolidate per (target_batch, plan_prod_key, oper_id, model)
        target_batch = problem.batch_of(plan_prod_key, oper_id) or b_id
        merged = False
        for a in allocations:
            if (
                a.plan_prod_key == plan_prod_key
                and a.oper_id == oper_id
                and a.batch_id == target_batch
                and a.eqp_model_cd == model
            ):
                a.eqp_qty += 1
                merged = True
                break
        if not merged:
            allocations.append(
                Allocation(
                    batch_id=target_batch,
                    plan_prod_key=plan_prod_key,
                    oper_id=oper_id,
                    eqp_model_cd=model,
                    eqp_qty=1,
                )
            )
    return AllocationSet(rule_timekey=problem.rule_timekey, allocations=allocations)
