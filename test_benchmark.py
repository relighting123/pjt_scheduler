"""DB-free validation: ensures the optimizer + heuristic + simulator agree
with each benchmark's ground_truth.json. Run:

    python test_benchmark.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from biz.data_loader import load_problem_from_csv_dir  # noqa: E402
from core.policy.heuristic import greedy_allocate  # noqa: E402
from core.policy.optimizer import optimal_allocate  # noqa: E402
from core.sim.snapshot import Simulator  # noqa: E402


def main() -> int:
    benchmark_root = ROOT / "benchmarks"
    if not benchmark_root.exists():
        print("No benchmark directory.", file=sys.stderr)
        return 1

    failures = 0
    print(f"{'dataset':<16} {'optimal':>8} {'expected':>9} {'heuristic':>10} {'status':>8}")
    print("-" * 64)
    for sub in sorted(benchmark_root.iterdir()):
        if not sub.is_dir():
            continue
        gt_path = sub / "ground_truth.json"
        if not gt_path.exists():
            continue
        problem = load_problem_from_csv_dir(sub)
        sim = Simulator(problem)
        opt = sim.simulate(optimal_allocate(problem))
        heu = sim.simulate(greedy_allocate(problem))
        expected = float(json.loads(gt_path.read_text()).get("avg_achievement", 0.0))
        ok = abs(opt.avg_achievement - expected) < 1e-3
        status = "OK" if ok else "MISMATCH"
        if not ok:
            failures += 1
        print(f"{sub.name:<16} {opt.avg_achievement:>8.3f} {expected:>9.3f} {heu.avg_achievement:>10.3f} {status:>8}")

    print("-" * 64)
    if failures:
        print(f"FAILED: {failures} datasets")
        return 1
    print("All benchmarks consistent with ground truth.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
