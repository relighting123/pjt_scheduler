"""Multi-period (time-sliced) scheduling engine — phase 2.

The single-snapshot simulator answers "given one allocation, how much do we
produce?" That assumes every operation already has enough WIP. In reality a
downstream operation can start empty: you must first run the upstream op to
*build* its queue, then switch equipment downstream. This module models that.

Model
-----
* The horizon is split into ``num_slots`` slots of ``slot_hours`` each.
* WIP is a live state: producing ``q`` at operation k this slot DEPLETES the
  queue at k and ADDS ``q`` to the queue at the next operation (by OPER_SEQ) —
  available only in the NEXT slot (production latency). This latency is what
  forces build-ahead.
* Each slot a *policy* chooses an allocation from the current WIP/plan state.
  Re-allocating across slots is what lets the scheduler sequence work and, if
  it crosses batches, incurs a switch.

WIP convention here is literal: 0 means an empty queue, a large value (e.g.
999999) means an abundant raw-material feed. (The single-snapshot path keeps
its "0 == unlimited" back-compat default; see ``heuristic.greedy_allocate``.)
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from .domain import AllocationSet, SchedulingProblem
from .heuristic import greedy_allocate
from .simulator import count_switches

# A policy maps (problem, current_wip, remaining_plan, prev_alloc, slot_idx)
# to the allocation to run this slot.
Policy = Callable[
    [SchedulingProblem, Dict[Tuple[str, str], float], Dict[Tuple[str, str], float],
     Optional[AllocationSet], int],
    AllocationSet,
]


@dataclass
class FlowResult:
    avg_achievement: float = 0.0
    achievement_by_pko: Dict[Tuple[str, str], float] = field(default_factory=dict)
    produced_by_pko: Dict[Tuple[str, str], float] = field(default_factory=dict)
    total_switches: int = 0
    schedule: List[AllocationSet] = field(default_factory=list)
    wip_trace: List[Dict[Tuple[str, str], float]] = field(default_factory=list)


class MultiPeriodSimulator:
    def __init__(self, problem: SchedulingProblem, num_slots: int = 4, slot_hours: float = 1.0) -> None:
        self.problem = problem
        self.num_slots = int(num_slots)
        self.slot_hours = float(slot_hours)

    def _initial_wip(self) -> Dict[Tuple[str, str], float]:
        wip: Dict[Tuple[str, str], float] = {}
        for pk, op, _ in self.problem.plan_targets():
            wip[(pk, op)] = float(self.problem.wip_of(pk, op))
        return wip

    def run(self, policy: Policy) -> FlowResult:
        problem = self.problem
        wip = self._initial_wip()
        cumulative: Dict[Tuple[str, str], float] = defaultdict(float)
        plan = {(pk, op): qty for pk, op, qty in problem.plan_targets()}

        result = FlowResult()
        prev_alloc: Optional[AllocationSet] = None

        for slot in range(self.num_slots):
            remaining_plan = {k: max(0.0, plan[k] - cumulative[k]) for k in plan}
            alloc = policy(problem, dict(wip), remaining_plan, prev_alloc, slot)
            result.schedule.append(alloc)
            if prev_alloc is not None:
                result.total_switches += count_switches(prev_alloc, alloc)

            # capacity this slot per (pk, op), from the allocation
            capacity: Dict[Tuple[str, str], float] = defaultdict(float)
            for a in alloc.allocations:
                if not problem.is_available(a.plan_prod_key, a.oper_id, a.eqp_model_cd):
                    continue
                uph = problem.uph_of(a.plan_prod_key, a.oper_id, a.eqp_model_cd)
                capacity[(a.plan_prod_key, a.oper_id)] += uph * max(0, int(a.eqp_qty)) * self.slot_hours

            # produce from the queue snapshot at slot start (latency: this slot's
            # output is only visible downstream next slot)
            produced: Dict[Tuple[str, str], float] = {}
            for key, cap in capacity.items():
                produced[key] = min(cap, wip.get(key, 0.0))

            # apply depletion + downstream flow + accumulate
            for key, q in produced.items():
                if q <= 0:
                    continue
                wip[key] = wip.get(key, 0.0) - q
                cumulative[key] += q
                nxt = problem.next_oper_of(key[0], key[1])
                if nxt is not None:
                    wip[(key[0], nxt)] = wip.get((key[0], nxt), 0.0) + q

            result.wip_trace.append(dict(wip))
            prev_alloc = alloc

        # achievement vs total plan
        total = 0.0
        for (pk, op), plan_qty in plan.items():
            prod = cumulative[(pk, op)]
            rate = min(1.0, prod / plan_qty) if plan_qty > 0 else 1.0
            result.produced_by_pko[(pk, op)] = prod
            result.achievement_by_pko[(pk, op)] = rate
            total += rate
        result.avg_achievement = (total / len(plan)) if plan else 0.0
        return result


# --- policies --------------------------------------------------------------
def static_policy(problem, wip, remaining_plan, prev_alloc, slot) -> AllocationSet:
    """Phase-1 behaviour: decide once from the initial state, hold it forever."""
    if prev_alloc is not None:
        return prev_alloc
    return greedy_allocate(problem, wip_override=wip, treat_zero_as_unlimited=False)


def dynamic_greedy_policy(problem, wip, remaining_plan, prev_alloc, slot) -> AllocationSet:
    """Re-run greedy each slot against the live WIP queue and remaining plan."""
    return greedy_allocate(
        problem,
        wip_override=wip,
        plan_override=remaining_plan,
        treat_zero_as_unlimited=False,
    )


# --- small-case exact optimal ----------------------------------------------
def multiperiod_optimal(
    problem: SchedulingProblem,
    num_slots: int,
    slot_hours: float = 1.0,
    max_search: int = 20000,
) -> FlowResult:
    """Exact optimum for small instances by DFS over per-slot "pure" allocations.

    At each slot every equipment bucket is assigned wholly to one eligible
    operation (or idled). This enumerates the true optimum when buckets hold a
    single unit (the demonstrator case) and a strong lower bound otherwise. For
    anything larger it falls back to the dynamic-greedy policy.
    """
    pool = list(problem.equipment_pool().items())  # [((batch, model), qty), ...]
    targets = [(pk, op) for pk, op, _ in problem.plan_targets()]

    # eligible ops per bucket
    bucket_options: List[List[Optional[Tuple[str, str]]]] = []
    for (b_id, model), _ in pool:
        opts: List[Optional[Tuple[str, str]]] = [None]  # idle
        for (pk, op) in targets:
            if problem.is_available(pk, op, model) and (
                b_id == problem.batch_of(pk, op) or model in problem.model_group_of(model)
            ):
                opts.append((pk, op))
        bucket_options.append(opts)

    # search-space guard
    per_slot = 1
    for opts in bucket_options:
        per_slot *= max(1, len(opts))
    if per_slot ** num_slots > max_search:
        sim = MultiPeriodSimulator(problem, num_slots, slot_hours)
        return sim.run(dynamic_greedy_policy)

    # enumerate per-slot pure allocations
    from .domain import Allocation

    def slot_allocations() -> List[AllocationSet]:
        combos: List[List[Optional[Tuple[str, str]]]] = [[]]
        for opts in bucket_options:
            combos = [c + [o] for c in combos for o in opts]
        result: List[AllocationSet] = []
        for combo in combos:
            allocs: List[Allocation] = []
            for ((b_id, model), qty), choice in zip(pool, combo):
                if choice is None or qty <= 0:
                    continue
                pk, op = choice
                target_batch = problem.batch_of(pk, op) or b_id
                allocs.append(Allocation(target_batch, pk, op, model, int(qty)))
            result.append(AllocationSet(rule_timekey=problem.rule_timekey, allocations=allocs))
        return result

    candidates = slot_allocations()

    best: Optional[FlowResult] = None

    def dfs(slot: int, plan_so_far: List[AllocationSet]):
        nonlocal best
        if slot == num_slots:
            seq = list(plan_so_far)
            sim = MultiPeriodSimulator(problem, num_slots, slot_hours)
            idx = {"i": 0}

            def replay(problem, wip, remaining_plan, prev_alloc, s):
                a = seq[idx["i"]]
                idx["i"] += 1
                return a

            res = sim.run(replay)
            if best is None or res.avg_achievement > best.avg_achievement or (
                abs(res.avg_achievement - best.avg_achievement) < 1e-9
                and res.total_switches < best.total_switches
            ):
                best = res
            return
        for cand in candidates:
            plan_so_far.append(cand)
            dfs(slot + 1, plan_so_far)
            plan_so_far.pop()

    dfs(0, [])
    assert best is not None
    return best
