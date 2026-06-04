"""SEQ schedule rows (no DB)."""
from __future__ import annotations

from pathlib import Path

from biz.data_loader import load_problem_from_csv_dir
from biz.schedule_output import SCHEDULE_COLUMNS, build_schedule_rows, write_schedule_csv
from biz.virtual_eqp import expand_allocation_to_virtual, merge_timeline_to_seq_segments
from core.policy.heuristic import greedy_allocate


def test_unique_virtual_eqp_ids():
    problem = load_problem_from_csv_dir("benchmarks/benchmark_05")
    alloc = greedy_allocate(problem)
    units = expand_allocation_to_virtual(alloc)
    ids = [u.virtual_eqp_id for u in units]
    assert len(ids) == len(set(ids))
    assert len(ids) == 4


def test_seq_merge_same_product():
    from biz.virtual_eqp import EqSlotAssignment

    slots = [
        EqSlotAssignment("EQ1", "B", "M", "P1", "OP10", 0),
        EqSlotAssignment("EQ1", "B", "M", "P1", "OP20", 1),
        EqSlotAssignment("EQ1", "B", "M", "P2", "OP10", 2),
        EqSlotAssignment("EQ1", "B", "M", "P2", "OP10", 3),
    ]
    segs = merge_timeline_to_seq_segments(slots)
    assert len(segs) == 2
    assert segs[0].seq_no == 1 and segs[0].plan_prod_key == "P1"
    assert segs[1].seq_no == 2 and segs[1].plan_prod_key == "P2"


def test_schedule_rows_benchmark_01():
    problem = load_problem_from_csv_dir("benchmarks/benchmark_01")
    alloc = greedy_allocate(problem)
    settings = {"infer": {}, "virtual_eqp": {"gantt_slots": 24, "slot_hours": 1.0}}
    rows = build_schedule_rows(problem, alloc, settings, mode="wip-static")
    assert len(rows) == 3
    assert all(r.seq_no == 1 for r in rows)
    assert rows[0].plan_prod_key == "P1"
    assert rows[0].start_time != rows[0].end_time or rows[0].produce_qty >= 0


def test_schedule_csv_columns():
    problem = load_problem_from_csv_dir("benchmarks/benchmark_01")
    alloc = greedy_allocate(problem)
    settings = {"infer": {}, "virtual_eqp": {"gantt_slots": 4, "slot_hours": 1.0}}
    rows = build_schedule_rows(problem, alloc, settings, mode="wip-static")
    path = write_schedule_csv("artifacts/reports/_test_schedule.csv", rows)
    text = Path(path).read_text(encoding="utf-8")
    header = text.splitlines()[0]
    assert list(SCHEDULE_COLUMNS) == header.split(",")


if __name__ == "__main__":
    test_unique_virtual_eqp_ids()
    test_seq_merge_same_product()
    test_schedule_rows_benchmark_01()
    test_schedule_csv_columns()
    print("schedule tests OK")
