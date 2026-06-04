"""Round-trip test for infer JSON snapshots (no DB)."""
from __future__ import annotations

from biz.data_loader import load_problem_from_csv_dir
from biz.problem_snapshot import (
    dump_infer_snapshot,
    load_infer_snapshot,
    problem_from_dict,
    problem_to_dict,
)


def test_snapshot_roundtrip_benchmark():
    problem = load_problem_from_csv_dir("benchmarks/benchmark_01")
    restored = problem_from_dict(problem_to_dict(problem))
    assert restored.rule_timekey == problem.rule_timekey
    assert len(restored.wip) == len(problem.wip)
    assert len(restored.uph) == len(problem.uph)
    assert len(restored.plan_targets()) == len(problem.plan_targets())

    path = dump_infer_snapshot(
        "artifacts/inference/snapshots/_test_benchmark_01.json",
        problem,
        mode="wip-static",
        source="benchmark",
    )
    loaded, meta = load_infer_snapshot(path)
    assert meta["input_summary"]["wip_rows"] == len(problem.wip)
    assert len(loaded.equipment_pool()) == len(problem.equipment_pool())


if __name__ == "__main__":
    test_snapshot_roundtrip_benchmark()
    print("snapshot round-trip OK")
