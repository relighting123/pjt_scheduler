#!/usr/bin/env python3
"""Validate benchmark CSV datasets and ground_truth without Oracle DB."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from core.domain import SchedulingDataset
from core.evaluation import evaluate_dataset
from core.optimizer import ImprovedGreedySolver, load_ground_truth_conversions
from core.simulator import SchedulingSimulator


def validate_benchmark(bench_dir: Path) -> list[str]:
    errors: list[str] = []
    required = [
        "meta.csv",
        "oper_wip.csv",
        "model_uph.csv",
        "eqp_count.csv",
        "model_avail.csv",
        "batch_oper.csv",
        "tool_qty.csv",
        "plan_slots.csv",
        "ground_truth.json",
    ]
    for name in required:
        if not (bench_dir / name).exists():
            errors.append(f"{bench_dir.name}: missing {name}")

    if errors:
        return errors

    ds = SchedulingDataset.from_csv_dir(bench_dir)
    errors.extend(ds.validate())

    gt = load_ground_truth_conversions(bench_dir / "ground_truth.json")
    sim = SchedulingSimulator(ds)
    opt_result = sim.simulate(gt)
    if opt_result.avg_achievement_rate < 0.5:
        errors.append(f"{bench_dir.name}: ground_truth achievement too low ({opt_result.avg_achievement_rate:.2f})")

    return errors


def main() -> int:
    root = Path("benchmarks")
    if not root.exists():
        print("benchmarks/ not found - run scripts/generate_benchmarks.py first")
        return 1

    all_errors: list[str] = []
    for bench in sorted(root.glob("benchmark_*")):
        all_errors.extend(validate_benchmark(bench))
        try:
            metrics = evaluate_dataset(bench)
            opt = metrics["optimal"].avg_achievement_rate
            heu = metrics["heuristic"].avg_achievement_rate
            print(f"{bench.name}: optimal={opt:.4f} heuristic={heu:.4f} OK")
        except Exception as exc:
            all_errors.append(f"{bench.name}: {exc}")

    if all_errors:
        for e in all_errors:
            print(f"ERROR: {e}", file=sys.stderr)
        return 1

    print("All benchmarks passed validation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
