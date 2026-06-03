"""Exact (small-scale) optimal allocator used to produce benchmark ground truth.

For benchmark problems we solve the integer allocation by exhaustive search
over (batch, model) -> target assignments. To keep this tractable we exploit
two structural facts:

  * Within a (batch_id, eqp_model_cd) bucket, equipment units are
    interchangeable — the only decision is how to *split* the bucket across
    targets it is eligible for. So the search is over compositions of an
    integer into N parts.
  * Targets that can't consume a particular bucket (no UPH / not available)
    are excluded up front.

This is intentionally simple — benchmark problems are small (5-20 targets,
≤ 30 equipment units total). For production scale, the RL policy is used.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Dict, List, Tuple

from .domain import Allocation, AllocationSet, SchedulingProblem
from .simulator import Simulator


def _compositions(n: int, k: int):
    """정수 n을 k개 비음 정수로 분할하는 모든 약 composition.

    Example:
        list(_compositions(3, 2))
        # → [(0,3), (1,2), (2,1), (3,0)]
    """
    if k == 1:
        yield (n,)
        return
    for i in range(n + 1):
        for tail in _compositions(n - i, k - 1):
            yield (i,) + tail


def optimal_allocate(
    problem: SchedulingProblem,
    max_units_per_bucket: int = 20,
    ignore_wip: bool = False,
) -> AllocationSet:
    """소규모 문제에 대한 brute-force 최적 할당.

    각 bucket을 적격 target들에 정수 분할하는 모든 경우를 DFS로 탐색해
    평균 달성률 최대인 할당을 반환. 탐색 공간이 크면 greedy로 폴백.

    Args:
        problem: 입력 스냅샷.
        max_units_per_bucket: 단일 bucket이 이 수보다 많으면 greedy 폴백.
        ignore_wip: True면 WIP cap 무시 (plan-only 모드).

    Returns:
        AllocationSet — 최적 결정 (또는 greedy 폴백).

    Example:
        # benchmark_08: T_FAST 1대, T_SLOW 1대; P1과 P2 각 100개 계획
        alloc = optimal_allocate(problem)
        # → T_FAST→P2 (희소 옵션 먼저), T_SLOW→P1
        #   greedy는 T_FAST→P1로 빠져서 P2=0 (avg 0.5)
        #   optimal은 두 제품 모두 100% (avg 1.0)
    """
    """Brute-force optimal allocation maximizing average achievement.

    Falls back to the greedy heuristic if a bucket exceeds `max_units_per_bucket`
    or if the search space is too large.
    """
    from .heuristic import greedy_allocate

    pool = problem.equipment_pool()
    targets = problem.plan_targets()
    target_keys = [(pk, op) for pk, op, _ in targets]
    plan_by_pko = {(pk, op): qty for pk, op, qty in targets}

    # eligible targets per bucket
    bucket_targets: Dict[Tuple[str, str], List[int]] = {}
    for bm, free in pool.items():
        b_id, model = bm
        eligible = []
        for idx, (pk, op) in enumerate(target_keys):
            if not problem.is_available(pk, op, model):
                continue
            if b_id != problem.batch_of(pk, op) and model not in problem.model_group_of(model):
                continue
            eligible.append(idx)
        if eligible:
            bucket_targets[bm] = eligible
        if free > max_units_per_bucket:
            return greedy_allocate(problem, ignore_wip=ignore_wip)

    # rough search-space guard
    space = 1
    for bm, eligible in bucket_targets.items():
        free = pool[bm]
        # compositions count: C(free + k - 1, k - 1)
        k = len(eligible)
        c = 1
        for i in range(k - 1):
            c = c * (free + k - 1 - i) // (i + 1)
        space *= max(1, c)
        if space > 2_000_000:
            return greedy_allocate(problem, ignore_wip=ignore_wip)

    best_score = -1.0
    best_assign: Dict[Tuple[str, str, int], int] = {}

    buckets = list(bucket_targets.items())

    def recurse(i: int, assign: Dict[Tuple[str, str, int], int], produced: Dict[Tuple[str, str], float]):
        nonlocal best_score, best_assign
        if i == len(buckets):
            total = 0.0
            for key, plan_qty in plan_by_pko.items():
                actual = produced.get(key, 0.0)
                if not ignore_wip:
                    wip = problem.wip_of(*key)
                    if wip > 0:
                        actual = min(actual, wip)
                total += min(1.0, actual / plan_qty) if plan_qty > 0 else 1.0
            avg = total / len(plan_by_pko) if plan_by_pko else 0.0
            if avg > best_score:
                best_score = avg
                best_assign = dict(assign)
            return
        bm, eligible = buckets[i]
        b_id, model = bm
        free = pool[bm]
        for comp in _compositions(free, len(eligible)):
            new_assign = assign
            delta_produced: List[Tuple[Tuple[str, str], float]] = []
            for idx_in_eligible, qty in enumerate(comp):
                if qty == 0:
                    continue
                tgt_idx = eligible[idx_in_eligible]
                pk, op = target_keys[tgt_idx]
                uph = problem.uph_of(pk, op, model)
                produced[(pk, op)] = produced.get((pk, op), 0.0) + uph * qty
                delta_produced.append(((pk, op), uph * qty))
                new_assign = {**new_assign, (b_id, model, tgt_idx): qty}
            recurse(i + 1, new_assign, produced)
            for key, delta in delta_produced:
                produced[key] -= delta

    recurse(0, {}, {})

    if not best_assign:
        return greedy_allocate(problem)

    allocations: List[Allocation] = []
    for (b_id, model, tgt_idx), qty in best_assign.items():
        if qty == 0:
            continue
        pk, op = target_keys[tgt_idx]
        target_batch = problem.batch_of(pk, op) or b_id
        allocations.append(
            Allocation(
                batch_id=target_batch,
                plan_prod_key=pk,
                oper_id=op,
                eqp_model_cd=model,
                eqp_qty=int(qty),
            )
        )
    return AllocationSet(rule_timekey=problem.rule_timekey, allocations=allocations)
