"""Split SQL row mappers produce same problem shape as pivot (no DB)."""
from __future__ import annotations

from biz.data_loader import (
    _assemble_scheduling_problem,
    _map_equipment_rows,
    _map_plan_rows,
    _map_tool_qty_rows,
    _map_uph_rows,
    _map_wip_rows,
    _rows_to_problem,
    load_problem_from_csv_dir,
    resolve_query_mode,
)


def test_split_mappers_match_csv_benchmark():
    """benchmark_01 CSV == split row mappers on synthetic tuples."""
    ref = load_problem_from_csv_dir("benchmarks/benchmark_01")
    rk = ref.rule_timekey

    wip_rows = [
        (rk, a.batch_id, a.plan_prod_key, a.oper_id, w.oper_seq, w.wip_qty)
        for w, a in zip(ref.wip, ref.tool_groups)
        for _ in [0]
    ]
    # rebuild wip rows from records
    wip_rows = []
    pko_batch = {(g.plan_prod_key, g.oper_id): g.batch_id for g in ref.tool_groups}
    for w in ref.wip:
        wip_rows.append((rk, pko_batch[(w.plan_prod_key, w.oper_id)],
                         w.plan_prod_key, w.oper_id, w.oper_seq, w.wip_qty))

    uph_rows = [
        (rk, u.plan_prod_key, u.oper_id, u.eqp_model_cd, u.uph) for u in ref.uph
    ]
    eqp_rows = [
        (rk, e.batch_id, e.eqp_model_cd, e.eqp_qty) for e in ref.equipment
    ]
    plan_rows = [
        (rk, p.plan_prod_key, p.oper_id, p.start_time, p.end_time, p.plan_qty)
        for p in ref.plans
    ]
    tool_rows = [
        (rk, t.batch_id, t.eqp_model_cd, t.tool_qty) for t in ref.tool_qty
    ]

    wip, seen = _map_wip_rows(rk, wip_rows)
    uph = _map_uph_rows(rk, uph_rows)
    eqp_map = _map_equipment_rows(rk, eqp_rows)
    plan_map = _map_plan_rows(rk, plan_rows)
    tool_qty = _map_tool_qty_rows(rk, tool_rows)
    split_problem = _assemble_scheduling_problem(
        rk, wip, uph, eqp_map, plan_map, tool_qty, seen, ref.eqp_model_groups,
    )

    assert len(split_problem.wip) == len(ref.wip)
    assert len(split_problem.uph) == len(ref.uph)
    assert len(split_problem.plan_targets()) == len(ref.plan_targets())
    assert split_problem.equipment_pool() == ref.equipment_pool()


def test_pivot_rows_equivalent_to_split_for_inf_shape():
    rk = "benchmark_01"
    pivot_rows = [
        (rk, "9C/92", "P1", "OP10", 1, "T5833", "AVAIL_WIP_QTY", "100"),
        (rk, "9C/92", "P1", "OP10", 1, "T5833", "EQUIP_UPH", "200"),
        (rk, "9C/92", "P1", "OP10", 1, "T5833", "ASSIGN_EQUIP_CNT", "4"),
        (rk, "9C/92", "P1", "OP10", 1, "T5833", "D0_PLAN", "600"),
    ]
    pivot = _rows_to_problem(rk, pivot_rows, {})
    wip, seen = _map_wip_rows(rk, [(rk, "9C/92", "P1", "OP10", 1, 100.0)])
    uph = _map_uph_rows(rk, [(rk, "P1", "OP10", "T5833", 200.0)])
    eqp = _map_equipment_rows(rk, [(rk, "9C/92", "T5833", 4)])
    plan = _map_plan_rows(rk, [(rk, "P1", "OP10", rk, rk, 600.0)])
    split = _assemble_scheduling_problem(rk, wip, uph, eqp, plan, [], seen, {})
    assert pivot.plan_qty_of("P1", "OP10") == split.plan_qty_of("P1", "OP10")


def test_resolve_query_mode():
    assert resolve_query_mode({"oracle": {"query_mode": "split"}}) == "split"
    assert resolve_query_mode({"oracle": {}}) == "pivot"


if __name__ == "__main__":
    test_split_mappers_match_csv_benchmark()
    test_pivot_rows_equivalent_to_split_for_inf_shape()
    test_resolve_query_mode()
    print("split loader tests OK")
