"""DB-free validation of the multi-period (WIP-flow) engine — phase 2.

Demonstrates the "build-ahead" dynamic that the single-snapshot model cannot
capture: a downstream operation starts with an empty queue, so equipment must
first run upstream to build WIP, then switch downstream in a later slot.

Run:
    python test_multiperiod.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from core.domain import (  # noqa: E402
    AvailabilityRecord, EquipmentRecord, PlanRecord, SchedulingProblem,
    ToolGroupRecord, ToolQtyRecord, UphRecord, WipRecord,
)
from core.flow import (  # noqa: E402
    MultiPeriodSimulator, dynamic_greedy_policy, multiperiod_optimal, static_policy,
)


def build_buildahead_problem() -> SchedulingProblem:
    """One product, two ops on the same batch, a single versatile unit.

    - OP10 has an abundant feed (raw material), OP20 starts empty.
    - Plan: 200 at OP10 and 200 at OP20 over a 2-slot horizon (UPH 200/slot).
    - Only way to hit both: run OP10 in slot 1 (builds OP20's queue), then run
      OP20 in slot 2. A single fixed allocation can satisfy at most one op.
    """
    rk = "2026051707000100"
    targets = [("P1", "OP10", 200.0), ("P1", "OP20", 200.0)]
    wip = {("P1", "OP10"): 999999.0, ("P1", "OP20"): 0.0}
    return SchedulingProblem(
        rule_timekey=rk,
        wip=[WipRecord(rk, pk, op, seq, wip[(pk, op)])
             for (pk, op, _), seq in zip(targets, (1, 2))],
        uph=[UphRecord(rk, "P1", "OP10", "T_GP", 200.0),
             UphRecord(rk, "P1", "OP20", "T_GP", 200.0)],
        equipment=[EquipmentRecord(rk, "9C/92", "T_GP", 1)],
        availability=[AvailabilityRecord(rk, "P1", "OP10", "T_GP", True),
                      AvailabilityRecord(rk, "P1", "OP20", "T_GP", True)],
        tool_groups=[ToolGroupRecord(rk, "9C/92", "P1", "OP10"),
                     ToolGroupRecord(rk, "9C/92", "P1", "OP20")],
        tool_qty=[ToolQtyRecord(rk, "9C/92", "T_GP", 1)],
        plans=[PlanRecord(rk, pk, op, rk, rk, qty) for pk, op, qty in targets],
        eqp_model_groups={"G001": ["9C/92"]},
    )


def build_thrashing_problem() -> SchedulingProblem:
    """Four products alternating between two batches (G001 = {A, B}).

    Plans, UPH and slot length are sized so that each (pk, op) needs *exactly*
    one full slot to be satisfied. Greedy processes targets in dict order
    (P1, P2, P3, P4), which alternates batches every slot and pays the switch
    cost on slots 1/2/3 — total production = 4·100 − 3·50 = 250 → avg 0.625.

    The optimum groups by batch (P1, P3 on A first, then P2, P4 on B), paying
    the cost on slot 2 only — total = 4·100 − 1·50 = 350 → avg 0.875.
    """
    rk = "2026051707000200"
    pairs = [("P1", "A"), ("P2", "B"), ("P3", "A"), ("P4", "B")]
    targets = [(pk, "OP10", 100.0) for pk, _ in pairs]
    return SchedulingProblem(
        rule_timekey=rk,
        wip=[WipRecord(rk, pk, "OP10", 1, 999999.0) for pk, _ in pairs],
        uph=[UphRecord(rk, pk, "OP10", "T_GP", 100.0) for pk, _ in pairs],
        # All units physically the same model; "batch" toggles. We pool the
        # single unit under batch A; the tool-group lets it serve B too.
        equipment=[EquipmentRecord(rk, "A", "T_GP", 1)],
        availability=[AvailabilityRecord(rk, pk, "OP10", "T_GP", True) for pk, _ in pairs],
        tool_groups=[ToolGroupRecord(rk, batch, pk, "OP10") for pk, batch in pairs],
        tool_qty=[ToolQtyRecord(rk, "A", "T_GP", 1)],
        plans=[PlanRecord(rk, pk, "OP10", rk, rk, qty) for pk, _, qty in
               [(pk, b, q) for (pk, b), (_, _, q) in zip(pairs, targets)]],
        eqp_model_groups={"G001": ["A", "B"]},
    )


def _run_scenario(name, problem, num_slots, slot_hours, switch_time_hours,
                  expected_static, expected_dynamic, expected_optimal) -> int:
    sim = MultiPeriodSimulator(problem, num_slots, slot_hours, switch_time_hours)
    static = sim.run(static_policy)
    dynamic = sim.run(dynamic_greedy_policy)
    optimal = multiperiod_optimal(problem, num_slots, slot_hours, switch_time_hours)

    print(f"\n=== {name} (slots={num_slots}, switch_time={switch_time_hours}h) ===")
    print(f"{'policy':<18} {'avg_achv':>9} {'switches':>9}")
    print("-" * 40)
    for label, r in (("static (phase-1)", static), ("dynamic greedy", dynamic), ("optimal", optimal)):
        print(f"{label:<18} {r.avg_achievement:>9.3f} {r.total_switches:>9}")

    print("Optimal schedule:")
    for i, alloc in enumerate(optimal.schedule):
        items = [(a.plan_prod_key, a.oper_id, a.batch_id, a.eqp_qty) for a in alloc.allocations] or ["idle"]
        print(f"  slot {i}: {items}")

    failures = 0
    for label, got, want in (("static", static.avg_achievement, expected_static),
                             ("dynamic", dynamic.avg_achievement, expected_dynamic),
                             ("optimal", optimal.avg_achievement, expected_optimal)):
        if abs(got - want) > 1e-6:
            print(f"FAIL [{name}/{label}]: expected {want}, got {got}")
            failures += 1
    return failures


def main() -> int:
    failures = 0

    # Build-ahead — dynamic recovers what static can't.
    failures += _run_scenario(
        "build-ahead (OP20 starts empty)",
        build_buildahead_problem(),
        num_slots=2, slot_hours=1.0, switch_time_hours=0.0,
        expected_static=0.5, expected_dynamic=1.0, expected_optimal=1.0,
    )

    # Thrashing — dynamic greedy pays per-slot switch cost; optimal batches.
    # 4 slots, switch_time 0.5h. Plan 200 each, UPH 100/slot.
    #   alternating P1 P2 P1 P2 (3 switches): production 100+50+50+50=250 split
    #     → P1≈150, P2≈100 → avg 0.625
    #   batched     P1 P1 P2 P2 (1 switch ):  100+100+50+100=350 split
    #     → P1=200 (1.0), P2=150 (0.75) → avg 0.875
    failures += _run_scenario(
        "thrashing (switch cost makes batching cheaper)",
        build_thrashing_problem(),
        num_slots=4, slot_hours=1.0, switch_time_hours=0.5,
        expected_static=0.25, expected_dynamic=0.625, expected_optimal=0.875,
    )

    print("-" * 40)
    if failures:
        print(f"FAILED: {failures} checks")
        return 1
    print("OK: dynamic > static (build-ahead), optimal > dynamic (thrashing).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
