"""Virtual equipment gantt (no DB)."""
from __future__ import annotations

from pathlib import Path

from biz.data_loader import load_problem_from_csv_dir
from biz.gantt_report import build_and_render_gantt
from biz.pipeline import run_infer
from biz.virtual_eqp import assign_virtual_units
from core.policy.heuristic import greedy_allocate


def test_virtual_expand_and_gantt():
    problem = load_problem_from_csv_dir("benchmarks/benchmark_01")
    alloc = greedy_allocate(problem)
    settings = {
        "infer": {"hours_per_day": 24, "horizon_hours": 1.0, "report_dir": "artifacts/reports"},
        "virtual_eqp": {"gantt_slots": 24, "slot_hours": 1.0},
    }
    vas = assign_virtual_units(problem, alloc, settings)
    assert len(vas) == sum(a.eqp_qty for a in alloc.allocations)
    meta = build_and_render_gantt(
        problem, alloc, settings,
        "artifacts/reports/_test_gantt.html",
        rule_timekey="benchmark_01", mode="wip-static",
    )
    assert Path(meta["gantt_html"]).exists()
    assert "gantt" in Path(meta["gantt_html"]).read_text(encoding="utf-8")


def test_unit_rules_block():
    problem = load_problem_from_csv_dir("benchmarks/benchmark_01")
    alloc = greedy_allocate(problem)
    if not alloc.allocations:
        return
    a0 = alloc.allocations[0]
    key = f"{a0.eqp_model_cd}|{a0.batch_id}"
    settings = {
        "virtual_eqp": {
            "unit_rules": {
                key: [[["WRONG_PK", "WRONG_OP"]]],
            },
        },
    }
    vas = assign_virtual_units(problem, alloc, settings)
    blocked = [v for v in vas if not v.allowed]
    assert blocked


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
    test_unit_rules_block()
    test_infer_includes_gantt_path()
    print("gantt tests OK")
