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

from ..domain import AllocationSet, SchedulingProblem
from ..policy.heuristic import greedy_allocate
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
    """`MultiPeriodSimulator.run()` 또는 `multiperiod_optimal()`의 출력.

    Fields:
        avg_achievement: 모든 (pk, op)에 대한 평균 달성률.
        achievement_by_pko: (pk, op) → 0.0~1.0.
        produced_by_pko:    (pk, op) → 호라이즌 누적 생산량.
        total_switches:     슬롯 간 (batch, model) 이동 횟수 총합.
        schedule:           슬롯별 AllocationSet 리스트 (길이 = num_slots).
        wip_trace:          각 슬롯 종료 시 WIP 상태 스냅샷.

    Example:
        FlowResult(
            avg_achievement=0.875,
            achievement_by_pko={("P1","OP10"):1.0, ("P2","OP10"):0.5, ...},
            total_switches=1,
            schedule=[AllocationSet(...), ...],  # 4개 (num_slots)
        )
    """
    avg_achievement: float = 0.0
    achievement_by_pko: Dict[Tuple[str, str], float] = field(default_factory=dict)
    produced_by_pko: Dict[Tuple[str, str], float] = field(default_factory=dict)
    total_switches: int = 0
    schedule: List[AllocationSet] = field(default_factory=list)
    wip_trace: List[Dict[Tuple[str, str], float]] = field(default_factory=list)


class MultiPeriodSimulator:
    """멀티 피리어드 (WIP 흐름 + 전환 비용) 시뮬레이터.

    horizon을 num_slots개로 쪼개고, 각 슬롯의 OP_k 생산이 OP_{k+1}
    (OPER_SEQ 순)의 WIP로 **다음 슬롯에** 흘러간다 (one-slot latency).
    배치 변경 시 switch_time_hours 만큼 슬롯 시간을 잃는다.

    Args:
        problem: 입력 스냅샷.
        num_slots: 호라이즌 슬롯 수 (예: 4시간 → 4슬롯).
        slot_hours: 슬롯당 시간 (기본 1시간).
        switch_time_hours: 신규 (batch, model)로 이동하는 장비 1대당 셋업 시간.

    Example:
        sim = MultiPeriodSimulator(problem, num_slots=4, slot_hours=1.0,
                                   switch_time_hours=0.5)
        result = sim.run(dynamic_greedy_policy)
        print(result.avg_achievement)  # 0.625
        print(result.total_switches)   # 3
    """

    def __init__(
        self,
        problem: SchedulingProblem,
        num_slots: int = 4,
        slot_hours: float = 1.0,
        switch_time_hours: float = 0.0,
    ) -> None:
        self.problem = problem
        self.num_slots = int(num_slots)
        self.slot_hours = float(slot_hours)
        self.switch_time_hours = float(switch_time_hours)

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
            slot_switches = count_switches(prev_alloc, alloc) if prev_alloc is not None else 0
            result.total_switches += slot_switches

            # Switch cost model: units that moved into a new (batch, model) this
            # slot lose `switch_time_hours` of productive time. Per-allocation:
            # fresh_units = max(0, current_qty - previous_qty_on_same_(batch,model))
            # effective_hours_for_fresh = max(0, slot_hours - switch_time_hours)
            # First slot has no prior state — every unit is assumed to start in
            # the right place (no setup cost). From slot 1 onward, units that
            # were not on the same (batch, model) last slot pay switch_time.
            first_slot = prev_alloc is None
            prev_by_bm: Dict[Tuple[str, str], int] = defaultdict(int)
            if not first_slot:
                for a in prev_alloc.allocations:  # type: ignore[union-attr]
                    prev_by_bm[(a.batch_id, a.eqp_model_cd)] += int(a.eqp_qty)

            # capacity this slot per (pk, op), from the allocation
            capacity: Dict[Tuple[str, str], float] = defaultdict(float)
            consumed_prev: Dict[Tuple[str, str], int] = defaultdict(int)
            for a in alloc.allocations:
                if not problem.is_available(a.plan_prod_key, a.oper_id, a.eqp_model_cd):
                    continue
                qty = max(0, int(a.eqp_qty))
                bm = (a.batch_id, a.eqp_model_cd)
                if first_slot:
                    kept_units, fresh_units = qty, 0
                else:
                    kept_available = max(0, prev_by_bm[bm] - consumed_prev[bm])
                    kept_units = min(qty, kept_available)
                    fresh_units = qty - kept_units
                    consumed_prev[bm] += kept_units
                uph = problem.uph_of(a.plan_prod_key, a.oper_id, a.eqp_model_cd)
                effective_fresh_hours = max(0.0, self.slot_hours - self.switch_time_hours)
                cap = uph * (kept_units * self.slot_hours + fresh_units * effective_fresh_hours)
                capacity[(a.plan_prod_key, a.oper_id)] += cap

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
    """Phase-1 정책: 첫 슬롯에 한 번 결정하고 모든 슬롯 동일 유지.

    Example:
        sim.run(static_policy)
        # build-ahead 시나리오에서 OP10만 계속 → OP20=0 → avg 0.5
    """
    if prev_alloc is not None:
        return prev_alloc
    return greedy_allocate(problem, wip_override=wip, treat_zero_as_unlimited=False)


def dynamic_greedy_policy(problem, wip, remaining_plan, prev_alloc, slot) -> AllocationSet:
    """슬롯마다 현재 (실시간) WIP/잔여계획 기반으로 greedy 재실행.

    Example:
        sim.run(dynamic_greedy_policy)
        # build-ahead: slot 0 OP10 → slot 1 OP20 (WIP 채워졌으니) → avg 1.0
        # thrashing : slot마다 batch 바뀌어 전환 비용 ↑ → avg 0.625 (3 swaps)
    """
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
    switch_time_hours: float = 0.0,
    max_search: int = 20000,
) -> FlowResult:
    """소규모 인스턴스에 대한 멀티 피리어드 정확해 (슬롯별 "pure" 할당 DFS).

    매 슬롯 각 bucket을 적격 op 1개에 통째로 배정 (또는 idle). bucket이
    1대일 때 진정한 최적해, 그 외엔 강한 하한. 탐색 공간 큰 경우
    dynamic_greedy_policy로 폴백.

    Args:
        problem: 입력 스냅샷.
        num_slots / slot_hours / switch_time_hours: 시뮬레이션 파라미터.
        max_search: 슬롯별 조합 수 ^ num_slots가 이를 넘으면 폴백.

    Returns:
        FlowResult — 최적 일정 + 누적 산출/달성률/전환수.

    Example:
        # thrashing: 4 제품 × 2 배치 × 1대 장비 × 4슬롯
        opt = multiperiod_optimal(problem, num_slots=4, slot_hours=1.0,
                                  switch_time_hours=0.5)
        # → schedule = [P1, P3, P2, P4]  (배치 A,A,B,B 묶음, 전환 1회)
        #   avg_achievement = 0.875
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
        sim = MultiPeriodSimulator(problem, num_slots, slot_hours, switch_time_hours)
        return sim.run(dynamic_greedy_policy)

    # enumerate per-slot pure allocations
    from ..domain import Allocation

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
            sim = MultiPeriodSimulator(problem, num_slots, slot_hours, switch_time_hours)
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
