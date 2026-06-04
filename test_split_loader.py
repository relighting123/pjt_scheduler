"""Oracle input row mappers (no DB)."""
from __future__ import annotations

from biz.data_loader import (
    _assemble_scheduling_problem,
    _map_equipment_rows,
    _map_plan_rows,
    _map_tool_qty_rows,
    _map_uph_rows,
    _map_wip_rows,
    load_problem_from_csv_dir,
)


def test_mappers_match_csv_benchmark():
    ref = load_problem_from_csv_dir("benchmarks/benchmark_01")
    rk = ref.rule_timekey
    pko_batch = {(g.plan_prod_key, g.oper_id): g.batch_id for g in ref.tool_groups}
    wip_rows = [
        (rk, pko_batch[(w.plan_prod_key, w.oper_id)],
         w.plan_prod_key, w.oper_id, w.oper_seq, w.wip_qty)
        for w in ref.wip
    ]
    uph_rows = [(rk, u.plan_prod_key, u.oper_id, u.eqp_model_cd, u.uph) for u in ref.uph]
    eqp_rows = [(rk, e.batch_id, e.eqp_model_cd, e.eqp_qty) for e in ref.equipment]
    plan_rows = [
        (rk, p.plan_prod_key, p.oper_id, p.start_time, p.end_time, p.plan_qty)
        for p in ref.plans
    ]
    tool_rows = [(rk, t.batch_id, t.eqp_model_cd, t.tool_qty) for t in ref.tool_qty]

    wip, seen = _map_wip_rows(rk, wip_rows)
    split = _assemble_scheduling_problem(
        rk, wip, _map_uph_rows(rk, uph_rows),
        _map_equipment_rows(rk, eqp_rows), _map_plan_rows(rk, plan_rows),
        _map_tool_qty_rows(rk, tool_rows), seen, ref.eqp_model_groups,
    )
    assert len(split.wip) == len(ref.wip)
    assert split.equipment_pool() == ref.equipment_pool()


if __name__ == "__main__":
    test_mappers_match_csv_benchmark()
    print("input mapper tests OK")
