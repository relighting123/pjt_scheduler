"""Infer report builder (no DB)."""
from __future__ import annotations

from biz.data_loader import load_problem_from_csv_dir, resolve_fac_id
from biz.infer_report import build_infer_report
from biz.pipeline import run_infer
from core.policy.heuristic import greedy_allocate


def test_build_infer_report_benchmark():
    problem = load_problem_from_csv_dir("benchmarks/benchmark_01")
    alloc = greedy_allocate(problem)
    settings = {"infer": {"hours_per_day": 24.0, "horizon_hours": 1.0}}
    report = build_infer_report(problem, alloc, settings, "wip-static")
    assert report["targets"]
    assert 0.0 <= report["avg_achievement"] <= 1.0
    assert report["total_daily_capacity"] > 0
    t0 = report["targets"][0]
    assert "batch_id" in t0 and "eqp_qty_by_model" in t0


def test_resolve_fac_id():
    assert resolve_fac_id({"oracle": {"fac_id": "ICPRB"}}) == "ICPRB"
    assert resolve_fac_id({"oracle": {}}, override="X") == "X"


def test_run_infer_benchmark_includes_report():
    settings = {
        "model": {"mode": "wip-static", "artifact_dir": "artifacts/models", "policy_name": "ppo_dispatch"},
        "infer": {"hours_per_day": 24.0},
    }
    result = run_infer(
        settings,
        benchmark_dataset="benchmarks/benchmark_01",
        mode="wip-static",
    )
    assert "infer_report" in result
    assert result["infer_report"]["targets"]


if __name__ == "__main__":
    test_build_infer_report_benchmark()
    test_resolve_fac_id()
    test_run_infer_benchmark_includes_report()
    print("infer report tests OK")
