"""Infer report builder (no DB)."""
from __future__ import annotations

from pathlib import Path

from biz.data_loader import load_problem_from_csv_dir, resolve_fac_id
from biz.infer_report import build_infer_report, render_infer_report_html
from biz.pipeline import run_infer
from core.policy.heuristic import greedy_allocate


def test_build_infer_report_benchmark():
    problem = load_problem_from_csv_dir("benchmarks/benchmark_01")
    alloc = greedy_allocate(problem)
    settings = {"infer": {"hours_per_day": 24.0, "horizon_hours": 1.0}}
    report = build_infer_report(problem, alloc, settings, "wip-static")
    assert report["targets"]
    assert 0.0 <= report["avg_achievement"] <= 1.0
    assert 0.0 <= report["avg_utilization"] <= 1.0
    assert report["total_daily_capacity"] > 0
    t0 = report["targets"][0]
    assert "batch_id" in t0 and "eqp_qty_by_model" in t0
    assert "utilization_rate" in t0


def test_resolve_fac_id():
    assert resolve_fac_id({"oracle": {"fac_id": "ICPRB"}}) == "ICPRB"
    assert resolve_fac_id({"oracle": {}}, override="X") == "X"


def test_infer_report_html():
    problem = load_problem_from_csv_dir("benchmarks/benchmark_01")
    alloc = greedy_allocate(problem)
    settings = {"infer": {"hours_per_day": 24.0, "horizon_hours": 1.0}}
    report = build_infer_report(problem, alloc, settings, "wip-static")
    out = render_infer_report_html(
        report,
        "artifacts/reports/_test_infer_kpi.html",
        rule_timekey=problem.rule_timekey,
        mode="wip-static",
        fac_id="CJPRB",
        source="benchmark",
        rows=2,
        allocation_count=len(alloc.allocations),
        input_summary={"wip_rows": len(problem.wip)},
    )
    text = Path(out).read_text(encoding="utf-8")
    assert "평균 달성률" in text
    assert "평균 가동률" in text
    assert "공정별 KPI" in text
    assert problem.wip[0].plan_prod_key in text or "plan_prod_key" in text


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
    assert "report_html" in result
    assert Path(result["report_html"]).exists()


if __name__ == "__main__":
    test_build_infer_report_benchmark()
    test_resolve_fac_id()
    test_infer_report_html()
    test_run_infer_benchmark_includes_report()
    print("infer report tests OK")
