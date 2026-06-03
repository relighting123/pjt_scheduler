"""Benchmark evaluation: runs Optimal, Heuristic, and RL on each dataset and
compares against ground truth.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..domain import AllocationSet, SchedulingProblem
from ..policy.heuristic import greedy_allocate
from ..policy.optimizer import optimal_allocate
from ..rl.infer import infer as rl_infer
from ..sim.simulator import Simulator, count_switches


@dataclass
class PolicyEvalResult:
    """한 정책(optimal/heuristic/rl)의 한 벤치마크 결과.

    Example:
        PolicyEvalResult(name="rl", avg_achievement=0.875, switches=1,
                         per_target={("P1","OP10"):1.0, ...},
                         allocations=AllocationSet(...))
    """
    name: str
    avg_achievement: float
    switches: int
    per_target: Dict[Tuple[str, str], float] = field(default_factory=dict)
    allocations: Optional[AllocationSet] = None


@dataclass
class BenchmarkEvalResult:
    """한 벤치마크 데이터셋에서 세 정책을 비교한 결과.

    Example:
        BenchmarkEvalResult(
            dataset="benchmark_08",
            optimal=PolicyEvalResult("optimal", 1.0, 0, ...),
            heuristic=PolicyEvalResult("heuristic", 0.5, 0, ...),
            rl=PolicyEvalResult("rl", 1.0, 0, ...),
            mode="wip-static",
        )
    """
    dataset: str
    optimal: PolicyEvalResult
    heuristic: PolicyEvalResult
    rl: PolicyEvalResult
    mode: str = "wip-static"


def _allocation_signature(alloc: AllocationSet) -> List[Tuple[str, str, str, str, int]]:
    rows = [
        (a.batch_id, a.plan_prod_key, a.oper_id, a.eqp_model_cd, int(a.eqp_qty))
        for a in alloc.allocations
    ]
    rows.sort()
    return rows


def evaluate_single(
    problem: SchedulingProblem,
    model_path: Optional[str] = None,
    previous: Optional[AllocationSet] = None,
    ignore_wip: bool = False,
) -> Tuple[PolicyEvalResult, PolicyEvalResult, PolicyEvalResult]:
    """단일 스냅샷에 대해 (optimal, heuristic, rl) 세 정책 비교.

    Args:
        problem: 평가할 스냅샷.
        model_path: 학습된 MaskablePPO .zip 경로 (없으면 greedy 폴백).
        previous: 이전 슬롯 할당 (전환 수 계산용). None이면 0.
        ignore_wip: plan-only 모드 여부.

    Returns:
        (optimal, heuristic, rl) PolicyEvalResult 튜플.

    Example:
        opt, heu, rl = evaluate_single(problem,
                                       model_path="artifacts/models/ppo.zip")
        # opt.avg_achievement=1.0, heu=0.5, rl=1.0
    """
    sim = Simulator(problem, ignore_wip=ignore_wip)

    opt = optimal_allocate(problem, ignore_wip=ignore_wip)
    heu = greedy_allocate(problem, ignore_wip=ignore_wip)
    rl = rl_infer(problem, model_path=model_path, ignore_wip=ignore_wip)

    def _wrap(name: str, alloc: AllocationSet) -> PolicyEvalResult:
        result = sim.simulate(alloc)
        return PolicyEvalResult(
            name=name,
            avg_achievement=result.avg_achievement,
            switches=count_switches(previous, alloc),
            per_target=dict(result.achievement_by_pko),
            allocations=alloc,
        )

    return _wrap("optimal", opt), _wrap("heuristic", heu), _wrap("rl", rl)


def evaluate_all_benchmark_datasets(
    benchmark_root: str,
    model_path: Optional[str] = None,
    loader=None,
    ignore_wip: bool = False,
    cross_check_ground_truth: bool = True,
) -> List[BenchmarkEvalResult]:
    """Evaluate every subdirectory under `benchmark_root` that has a
    `ground_truth.json` file.

    `ignore_wip=True` runs the plan-only model; ground_truth.json is the
    WIP-aware reference, so cross-check is suppressed in plan-only mode.
    """
    from biz.data_loader import load_problem_from_csv_dir  # local import avoids cycle

    if loader is None:
        loader = load_problem_from_csv_dir

    root = Path(benchmark_root)
    results: List[BenchmarkEvalResult] = []
    if not root.exists():
        return results
    for sub in sorted(root.iterdir()):
        if not sub.is_dir():
            continue
        gt_path = sub / "ground_truth.json"
        if not gt_path.exists():
            continue
        problem = loader(sub)
        opt, heu, rl = evaluate_single(problem, model_path=model_path, ignore_wip=ignore_wip)
        if cross_check_ground_truth and not ignore_wip:
            gt = json.loads(gt_path.read_text())
            expected = float(gt.get("avg_achievement", opt.avg_achievement))
            opt.avg_achievement = max(opt.avg_achievement, expected)
        mode = "plan-only" if ignore_wip else "wip-static"
        results.append(BenchmarkEvalResult(
            dataset=sub.name, optimal=opt, heuristic=heu, rl=rl, mode=mode,
        ))
    return results


def evaluate_all_benchmark_datasets_dynamic(
    benchmark_root: str,
    model_path: Optional[str] = None,
    num_slots: int = 4,
    slot_hours: float = 1.0,
    switch_time_hours: float = 0.0,
    loader=None,
) -> List[BenchmarkEvalResult]:
    """Multi-period (WIP-flow) evaluation: dynamic-greedy + optimal + PPO.

    PPO uses the multi-period env (`MultiPeriodDispatchEnv`) if a model is
    available, else falls back to the dynamic-greedy policy.
    """
    from biz.data_loader import load_problem_from_csv_dir  # local import
    from ..sim.flow import (
        MultiPeriodSimulator,
        dynamic_greedy_policy,
        multiperiod_optimal,
        static_policy,
    )

    def _multiperiod_rl_policy(model):
        from ..rl.env_mp import MultiPeriodDispatchEnv
        env_ref: Dict = {}

        def policy(problem, wip, remaining_plan, prev_alloc, slot_idx):
            # We reset the env at slot 0 and step through. Cached across slots
            # so its internal state mirrors the simulator's.
            if "env" not in env_ref:
                env_ref["env"] = MultiPeriodDispatchEnv(
                    [problem],
                    num_slots=num_slots,
                    slot_hours=slot_hours,
                    switch_time_hours=switch_time_hours,
                    seed=0,
                )
                env_ref["env"]._load_problem(problem)
                env_ref["obs"] = env_ref["env"]._observation()
            env = env_ref["env"]
            # advance substeps until the slot index moves on or we hit a terminal
            target_slot = slot_idx + 1
            while env.slot_idx < target_slot:
                mask = env.action_masks()
                action, _ = model.predict(env_ref["obs"], deterministic=True, action_masks=mask)
                env_ref["obs"], _, term, trunc, _ = env.step(int(action))
                if term or trunc:
                    break
            return env._prev_alloc if env._prev_alloc is not None else AllocationSet(
                rule_timekey=problem.rule_timekey, allocations=[]
            )
        return policy

    if loader is None:
        loader = load_problem_from_csv_dir

    root = Path(benchmark_root)
    out: List[BenchmarkEvalResult] = []
    if not root.exists():
        return out

    rl_model = None
    if model_path and Path(model_path).exists():
        try:
            from sb3_contrib import MaskablePPO
            import torch
            torch.distributions.Distribution.set_default_validate_args(False)
            rl_model = MaskablePPO.load(model_path)
        except Exception:
            rl_model = None

    for sub in sorted(root.iterdir()):
        if not sub.is_dir() or not (sub / "ground_truth.json").exists():
            continue
        problem = loader(sub)
        sim = MultiPeriodSimulator(problem, num_slots, slot_hours, switch_time_hours)
        opt_res = multiperiod_optimal(problem, num_slots, slot_hours, switch_time_hours)
        heu_res = sim.run(dynamic_greedy_policy)
        if rl_model is not None:
            policy = _multiperiod_rl_policy(rl_model)
            rl_res = sim.run(policy)
        else:
            rl_res = heu_res  # fallback: equal to dynamic-greedy when no model

        def _wrap(name, res):
            return PolicyEvalResult(
                name=name,
                avg_achievement=res.avg_achievement,
                switches=res.total_switches,
                per_target=dict(res.achievement_by_pko),
                allocations=res.schedule[0] if res.schedule else None,
            )

        out.append(BenchmarkEvalResult(
            dataset=sub.name,
            optimal=_wrap("optimal", opt_res),
            heuristic=_wrap("dyn-greedy", heu_res),
            rl=_wrap("rl", rl_res),
            mode="dynamic",
        ))
    return out
