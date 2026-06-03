"""Greedy heuristic baseline.

Allocates equipment per (batch_id, eqp_model_cd) to the (plan_prod_key, oper_id)
with the highest remaining shortfall * UPH. Respects tool-group restrictions —
units stay within the same batch unless an explicit tool-group conversion is
beneficial. Used as the imitation-learning teacher and a baseline.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from .domain import Allocation, AllocationSet, SchedulingProblem


def greedy_allocate(
    problem: SchedulingProblem,
    wip_override: Dict[Tuple[str, str], float] | None = None,
    plan_override: Dict[Tuple[str, str], float] | None = None,
    treat_zero_as_unlimited: bool = True,
    ignore_wip: bool = False,
) -> AllocationSet:
    """단일 슬롯 그리디 할당. 매 반복 (target, bucket)의 marginal 점수가
    가장 큰 한 쌍을 골라 장비 1대 배정, 잔여 없을 때까지 반복.

    Marginal = min(잔여 plan, UPH, 잔여 WIP).

    Args:
        problem: 입력 스냅샷.
        wip_override / plan_override: (pk, op)별 WIP/plan을 override.
            멀티 피리어드 엔진이 매 슬롯 현재 상태로 그리디를 호출할 때 사용.
            생략 시 problem 안의 정적 값 사용 (single-snapshot 모드).
        treat_zero_as_unlimited: True면 WIP=0/누락을 "무제한"으로 해석 (단일
            스냅샷 기본). 멀티 피리어드는 False로 두어 실제 큐 비었음을 표현.
        ignore_wip: True이면 WIP 자체를 무시 (plan-only 모드).

    Returns:
        AllocationSet — Greedy 결정.

    Example:
        problem = load_problem_from_csv_dir("benchmarks/benchmark_01")
        alloc = greedy_allocate(problem)
        # AllocationSet(rule_timekey="benchmark_01", allocations=[
        #     Allocation("9C/92","P1","OP10","T5833", eqp_qty=4),
        # ])
    """
    pool: Dict[Tuple[str, str], int] = dict(problem.equipment_pool())
    remaining: Dict[Tuple[str, str], float] = {}
    for pk, op, qty in problem.plan_targets():
        if plan_override is not None and (pk, op) in plan_override:
            remaining[(pk, op)] = float(plan_override[(pk, op)])
        else:
            remaining[(pk, op)] = qty
    # WIP cap: each target can produce at most the WIP currently in queue.
    # `ignore_wip=True` reverts to the plan-only model (no queue constraint).
    wip_remaining: Dict[Tuple[str, str], float] = {}
    for (pk, op) in remaining:
        if ignore_wip:
            wip_remaining[(pk, op)] = float("inf")
            continue
        if wip_override is not None and (pk, op) in wip_override:
            w = float(wip_override[(pk, op)])
        else:
            w = problem.wip_of(pk, op)
        if treat_zero_as_unlimited:
            wip_remaining[(pk, op)] = w if w > 0 else float("inf")
        else:
            wip_remaining[(pk, op)] = w
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
