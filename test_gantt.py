"""Virtual equipment gantt (no DB)."""
from __future__ import annotations

from pathlib import Path

from biz.data_loader import load_problem_from_csv_dir
from biz.gantt_report import build_and_render_gantt
from biz.pipeline import run_infer
from biz.virtual_eqp import expand_allocation_to_virtual
from core.policy.heuristic import greedy_allocate


def test_virtual_expand_and_gantt():
    problem = load_problem_from_csv_dir("benchmarks/benchmark_01")
    alloc = greedy_allocate(problem)
    settings = {
        "infer": {"hours_per_day": 24, "report_dir": "artifacts/reports"},
        "virtual_eqp": {"gantt_slots": 24, "slot_hours": 1.0},
    }
    units = expand_allocation_to_virtual(alloc)
    assert len(units) == sum(a.eqp_qty for a in alloc.allocations)
    meta = build_and_render_gantt(
        problem, alloc, settings,
        "artifacts/reports/_test_gantt.html",
        rule_timekey="benchmark_01", mode="wip-static",
    )
    assert Path(meta["gantt_html"]).exists()
    html = Path(meta["gantt_html"]).read_text(encoding="utf-8")
    assert "호기 스케줄" in html
    assert "SEQ_NO" in html


def test_infer_includes_gantt_path():
    settings = {
        "model": {"mode": "wip-static", "artifact_dir": "artifacts/models", "policy_name": "ppo_dispatch"},
        "infer": {"write_gantt_html": True, "report_dir": "artifacts/reports"},
        "virtual_eqp": {"gantt_slots": 4, "slot_hours": 1.0},
    }
    result = run_infer(settings, benchmark_dataset="benchmarks/benchmark_01", mode="wip-static")
    assert "gantt_html" in result
    assert Path(result["gantt_html"]).exists()


if __name__ == "__main__":
    test_virtual_expand_and_gantt()
    test_infer_includes_gantt_path()
    print("gantt tests OK")
