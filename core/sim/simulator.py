"""Scheduling simulator.

Given a `SchedulingProblem` and an `AllocationSet`, compute production per
(plan_prod_key, oper_id) and the achievement rate vs. plan. The output is the
canonical reward signal for RL and the evaluation metric for benchmarks.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from ..domain import AllocationSet, SchedulingProblem


@dataclass
class SimulationResult:
    """`Simulator.simulate()`의 출력.

    Fields:
        produced_by_pko: (pk, op) → 실제 생산량 (WIP cap 적용 후)
        plan_by_pko:     (pk, op) → 계획량
        achievement_by_pko: (pk, op) → 달성률 (0.0~1.0)
        avg_achievement: 단순 평균 달성률
        over_allocations: 풀 초과 할당 경고 메시지

    Example:
        SimulationResult(
            produced_by_pko={("P1","OP10"): 600.0, ("P1","OP20"): 50.0},
            plan_by_pko    ={("P1","OP10"): 600.0, ("P1","OP20"): 200.0},
            achievement_by_pko={("P1","OP10"): 1.0, ("P1","OP20"): 0.25},
            avg_achievement=0.625,
            over_allocations=[],
        )
    """
    produced_by_pko: Dict[Tuple[str, str], float] = field(default_factory=dict)
    plan_by_pko: Dict[Tuple[str, str], float] = field(default_factory=dict)
    achievement_by_pko: Dict[Tuple[str, str], float] = field(default_factory=dict)
    avg_achievement: float = 0.0
    over_allocations: List[str] = field(default_factory=list)

    def __str__(self) -> str:  # pragma: no cover
        return f"avg_achievement={self.avg_achievement:.4f}, items={len(self.achievement_by_pko)}"


class Simulator:
    """단일 스냅샷 (지평 = horizon_hours) 생산 시뮬레이터.

    하나의 `SchedulingProblem`과 `AllocationSet`을 받아 (pk, op)별 실제
    생산량과 평균 달성률을 계산. RL 보상 신호 및 벤치마크 평가 지표의 근원.

    Args:
        problem: 입력 스냅샷.
        horizon_hours: 계획 지평 (기본 1시간). UPH × eqp × horizon = 산출.
        ignore_wip: True이면 WIP cap을 무시 (plan-only 모드).

    Example:
        sim = Simulator(problem, horizon_hours=1.0)
        result = sim.simulate(allocation)
        print(result.avg_achievement)   # 0.875
        print(result.achievement_by_pko[("P1","OP10")])  # 1.0
    """

    def __init__(self, problem: SchedulingProblem, horizon_hours: float = 1.0, ignore_wip: bool = False) -> None:
        self.problem = problem
        self.horizon_hours = float(horizon_hours)
        self.ignore_wip = bool(ignore_wip)

    def simulate(self, allocations: AllocationSet) -> SimulationResult:
        """할당 결정에 대한 시뮬레이션 실행.

        Args:
            allocations: 평가할 할당 결정.

        Returns:
            SimulationResult — 모든 (pk, op)에 대한 생산량/달성률.

        Example:
            allocation = AllocationSet("...", [
                Allocation("9C/92","P1","OP10","T5833", eqp_qty=3),
            ])
            result = sim.simulate(allocation)
            # → produced=600 (= 200uph × 3대), achievement=1.0
        """
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

        # achievement capped at 1.0 and also by WIP (you can't produce more than
        # the queue holds). Missing keys count as 0 achievement.
        total = 0.0
        counted = 0
        for pk_op, plan_qty in result.plan_by_pko.items():
            actual = produced.get(pk_op, 0.0)
            if not self.ignore_wip:
                wip = problem.wip_of(*pk_op)
                if wip > 0:
                    actual = min(actual, wip)
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
    """이전 할당 대비 tool 전환 수 계산.

    (batch, model) 쌍의 장비 수가 이전보다 증가했으면 그만큼 전환으로 간주.
    동일 batch_id 내에서는 plan_prod_key 변경은 전환 아님.

    Args:
        previous: 이전 스냅샷의 AllocationSet 또는 None (첫 슬롯).
        current: 현재 AllocationSet.

    Returns:
        총 전환 횟수 (int).

    Example:
        prev = AllocationSet("...", [Allocation("9C/92","P1","OP10","T5833",2)])
        curr = AllocationSet("...", [
            Allocation("9C/92","P1","OP10","T5833",1),
            Allocation("9C/102","P2","OP10","T5833",1),  # batch 바뀜
        ])
        count_switches(prev, curr)  # → 1
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
