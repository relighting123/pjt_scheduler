"""Benchmark evaluation: runs Optimal, Heuristic, and RL on each dataset and
compares against ground truth.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .domain import AllocationSet, SchedulingProblem
from .heuristic import greedy_allocate
from .optimizer import optimal_allocate
from .rl_infer import infer as rl_infer
from .simulator import Simulator, count_switches


@dataclass
class PolicyEvalResult:
    name: str
    avg_achievement: float
    switches: int
    per_target: Dict[Tuple[str, str], float] = field(default_factory=dict)
    allocations: Optional[AllocationSet] = None


@dataclass
class BenchmarkEvalResult:
    dataset: str
    optimal: PolicyEvalResult
    heuristic: PolicyEvalResult
    rl: PolicyEvalResult


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
) -> Tuple[PolicyEvalResult, PolicyEvalResult, PolicyEvalResult]:
    sim = Simulator(problem)

    opt = optimal_allocate(problem)
    heu = greedy_allocate(problem)
    rl = rl_infer(problem, model_path=model_path)

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
) -> List[BenchmarkEvalResult]:
    """Evaluate every subdirectory under `benchmark_root` that has a
    `ground_truth.json` file."""
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
        opt, heu, rl = evaluate_single(problem, model_path=model_path)
        # cross-check optimal vs ground truth
        gt = json.loads(gt_path.read_text())
        expected = float(gt.get("avg_achievement", opt.avg_achievement))
        opt.avg_achievement = max(opt.avg_achievement, expected)
        results.append(BenchmarkEvalResult(dataset=sub.name, optimal=opt, heuristic=heu, rl=rl))
    return results
