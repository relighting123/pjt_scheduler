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


def main() -> int:
    problem = build_buildahead_problem()
    num_slots, slot_hours = 2, 1.0
    sim = MultiPeriodSimulator(problem, num_slots=num_slots, slot_hours=slot_hours)

    static = sim.run(static_policy)
    dynamic = sim.run(dynamic_greedy_policy)
    optimal = multiperiod_optimal(problem, num_slots=num_slots, slot_hours=slot_hours)

    print("Build-ahead scenario (1 unit, OP10 abundant, OP20 empty, 2 slots)")
    print(f"{'policy':<18} {'avg_achv':>9} {'switches':>9}")
    print("-" * 40)
    for name, r in (("static (phase-1)", static), ("dynamic greedy", dynamic), ("optimal", optimal)):
        print(f"{name:<18} {r.avg_achievement:>9.3f} {r.total_switches:>9}")

    print("\nOptimal schedule (slot -> allocations):")
    for i, alloc in enumerate(optimal.schedule):
        items = [(a.oper_id, a.eqp_model_cd, a.eqp_qty) for a in alloc.allocations] or ["idle"]
        print(f"  slot {i}: {items}")

    # assertions: static can satisfy only one op; dynamic/optimal satisfy both
    failures = 0
    if not (abs(static.avg_achievement - 0.5) < 1e-6):
        print(f"FAIL: static expected 0.5, got {static.avg_achievement}")
        failures += 1
    if not (abs(dynamic.avg_achievement - 1.0) < 1e-6):
        print(f"FAIL: dynamic expected 1.0, got {dynamic.avg_achievement}")
        failures += 1
    if not (abs(optimal.avg_achievement - 1.0) < 1e-6):
        print(f"FAIL: optimal expected 1.0, got {optimal.avg_achievement}")
        failures += 1

    print("-" * 40)
    if failures:
        print(f"FAILED: {failures} checks")
        return 1
    print("OK: dynamic re-allocation recovers the build-ahead schedule "
          "(0.5 -> 1.0) that a static single allocation cannot.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
