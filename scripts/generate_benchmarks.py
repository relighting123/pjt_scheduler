"""Generate 7 benchmark datasets with known optimal answers.

Each dataset is a directory containing:
  wip_info.csv, uph.csv, equipment.csv, availability.csv,
  tool_group.csv, tool_qty.csv, plan.csv, tool_groups.json, ground_truth.json

Run:
  python scripts/generate_benchmarks.py
"""
from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1] / "benchmarks"


@dataclass
class BenchmarkSpec:
    name: str
    description: str
    rule_timekey: str
    targets: List[Tuple[str, str, float]]  # (plan_prod_key, oper_id, plan_qty)
    oper_seq: Dict[Tuple[str, str], int]
    batch_map: Dict[Tuple[str, str], str]  # (pk, op) -> batch_id
    equipment: List[Tuple[str, str, int]]  # (batch_id, model, qty)
    uph: Dict[Tuple[str, str, str], float]  # (pk, op, model) -> uph
    tool_qty: List[Tuple[str, str, int]]
    eqp_model_groups: Dict[str, List[str]] = field(default_factory=dict)
    wip: Dict[Tuple[str, str], float] = field(default_factory=dict)
    ground_truth_avg: float = 1.0  # filled by solver


def _write_csv(path: Path, header: List[str], rows: List[list]) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)


def _solve_optimal(spec: BenchmarkSpec) -> float:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from core.domain import (
        AvailabilityRecord, EquipmentRecord, PlanRecord, SchedulingProblem,
        ToolGroupRecord, ToolQtyRecord, UphRecord, WipRecord,
    )
    from core.optimizer import optimal_allocate
    from core.simulator import Simulator

    wip = [WipRecord(spec.rule_timekey, pk, op, spec.oper_seq.get((pk, op), 1), spec.wip.get((pk, op), 999999.0))
           for (pk, op, _) in spec.targets]
    uph = [UphRecord(spec.rule_timekey, pk, op, m, v) for (pk, op, m), v in spec.uph.items()]
    eqp = [EquipmentRecord(spec.rule_timekey, b, m, q) for (b, m, q) in spec.equipment]
    avail = [AvailabilityRecord(spec.rule_timekey, pk, op, m, True) for (pk, op, m) in spec.uph]
    tgrp = [ToolGroupRecord(spec.rule_timekey, b, pk, op) for (pk, op), b in spec.batch_map.items()]
    tqty = [ToolQtyRecord(spec.rule_timekey, b, m, q) for (b, m, q) in spec.tool_qty]
    plans = [PlanRecord(spec.rule_timekey, pk, op, spec.rule_timekey, spec.rule_timekey, qty)
             for (pk, op, qty) in spec.targets]
    problem = SchedulingProblem(
        rule_timekey=spec.rule_timekey,
        wip=wip, uph=uph, equipment=eqp, availability=avail,
        tool_groups=tgrp, tool_qty=tqty, plans=plans,
        eqp_model_groups=spec.eqp_model_groups,
    )
    alloc = optimal_allocate(problem)
    sim = Simulator(problem).simulate(alloc)
    return sim.avg_achievement


def _write_dataset(spec: BenchmarkSpec) -> None:
    d = ROOT / spec.name
    d.mkdir(parents=True, exist_ok=True)

    _write_csv(d / "wip_info.csv",
               ["rule_timekey", "plan_prod_key", "oper_id", "oper_seq", "wip_qty"],
               [[spec.rule_timekey, pk, op, spec.oper_seq.get((pk, op), 1), spec.wip.get((pk, op), 999999.0)]
                for (pk, op, _) in spec.targets])

    _write_csv(d / "uph.csv",
               ["rule_timekey", "plan_prod_key", "oper_id", "eqp_model_cd", "uph"],
               [[spec.rule_timekey, pk, op, m, v] for (pk, op, m), v in spec.uph.items()])

    _write_csv(d / "equipment.csv",
               ["rule_timekey", "batch_id", "eqp_model_cd", "eqp_qty"],
               [[spec.rule_timekey, b, m, q] for (b, m, q) in spec.equipment])

    _write_csv(d / "availability.csv",
               ["rule_timekey", "plan_prod_key", "oper_id", "eqp_model_cd", "avail_yn"],
               [[spec.rule_timekey, pk, op, m, "Y"] for (pk, op, m) in spec.uph])

    _write_csv(d / "tool_group.csv",
               ["rule_timekey", "batch_id", "plan_prod_key", "oper_id"],
               [[spec.rule_timekey, b, pk, op] for (pk, op), b in spec.batch_map.items()])

    _write_csv(d / "tool_qty.csv",
               ["rule_timekey", "batch_id", "eqp_model_cd", "tool_qty"],
               [[spec.rule_timekey, b, m, q] for (b, m, q) in spec.tool_qty])

    _write_csv(d / "plan.csv",
               ["rule_timekey", "plan_prod_key", "oper_id", "start_time", "end_time", "plan_qty"],
               [[spec.rule_timekey, pk, op, spec.rule_timekey, spec.rule_timekey, qty]
                for (pk, op, qty) in spec.targets])

    (d / "tool_groups.json").write_text(json.dumps(spec.eqp_model_groups, indent=2))

    gt_avg = _solve_optimal(spec)
    (d / "ground_truth.json").write_text(json.dumps({
        "description": spec.description,
        "avg_achievement": round(gt_avg, 6),
    }, indent=2))


def _specs() -> List[BenchmarkSpec]:
    specs: List[BenchmarkSpec] = []

    # 1) Balanced — one product, one op, plenty of equipment
    specs.append(BenchmarkSpec(
        name="benchmark_01",
        description="Balanced single-target case — clearly solvable at 100%.",
        rule_timekey="2026051707000001",
        targets=[("P1", "OP10", 600.0)],
        oper_seq={("P1", "OP10"): 1},
        batch_map={("P1", "OP10"): "9C/92"},
        equipment=[("9C/92", "T5833", 4)],
        uph={("P1", "OP10", "T5833"): 200.0},
        tool_qty=[("9C/92", "T5833", 4)],
        eqp_model_groups={"G001": ["9C/92", "9C/102"]},
    ))

    # 2) Bottleneck — split equipment between two ops on the same batch
    specs.append(BenchmarkSpec(
        name="benchmark_02",
        description="Split a model pool across two operations on the same batch.",
        rule_timekey="2026051707000002",
        targets=[("P1", "OP10", 400.0), ("P1", "OP20", 200.0)],
        oper_seq={("P1", "OP10"): 1, ("P1", "OP20"): 2},
        batch_map={("P1", "OP10"): "9C/92", ("P1", "OP20"): "9C/92"},
        equipment=[("9C/92", "T5833", 3)],
        uph={("P1", "OP10", "T5833"): 200.0, ("P1", "OP20", "T5833"): 200.0},
        tool_qty=[("9C/92", "T5833", 3)],
        eqp_model_groups={"G001": ["9C/92", "9C/102"]},
    ))

    # 3) Cross-batch conversion REQUIRED — tools in wrong batch
    specs.append(BenchmarkSpec(
        name="benchmark_03",
        description="Equipment lives in batch G001 partner but is needed elsewhere; conversion required.",
        rule_timekey="2026051707000003",
        targets=[("P1", "OP10", 400.0)],
        oper_seq={("P1", "OP10"): 1},
        batch_map={("P1", "OP10"): "9C/92"},
        equipment=[("9C/102", "T5833", 2), ("9C/92", "T5833", 0)],
        uph={("P1", "OP10", "T5833"): 200.0},
        tool_qty=[("9C/102", "T5833", 2), ("9C/92", "T5833", 0)],
        eqp_model_groups={"G001": ["9C/92", "9C/102"]},
    ))

    # 4) Multi-model — pick the higher-UPH model
    specs.append(BenchmarkSpec(
        name="benchmark_04",
        description="Pick the higher-UPH model among multiple options.",
        rule_timekey="2026051707000004",
        targets=[("P1", "OP10", 300.0)],
        oper_seq={("P1", "OP10"): 1},
        batch_map={("P1", "OP10"): "9C/92"},
        equipment=[("9C/92", "T5833", 2), ("9C/92", "MAGNUM5", 2)],
        uph={("P1", "OP10", "T5833"): 100.0, ("P1", "OP10", "MAGNUM5"): 150.0},
        tool_qty=[("9C/92", "T5833", 2), ("9C/92", "MAGNUM5", 2)],
        eqp_model_groups={"G001": ["9C/92"]},
    ))

    # 5) Multi-product competing for the same model — must split optimally
    specs.append(BenchmarkSpec(
        name="benchmark_05",
        description="Two products on the same batch competing for the same model.",
        rule_timekey="2026051707000005",
        targets=[("P1", "OP10", 400.0), ("P2", "OP10", 400.0)],
        oper_seq={("P1", "OP10"): 1, ("P2", "OP10"): 1},
        batch_map={("P1", "OP10"): "9C/92", ("P2", "OP10"): "9C/92"},
        equipment=[("9C/92", "T5833", 4)],
        uph={("P1", "OP10", "T5833"): 200.0, ("P2", "OP10", "T5833"): 200.0},
        tool_qty=[("9C/92", "T5833", 4)],
        eqp_model_groups={"G001": ["9C/92", "9C/102"]},
    ))

    # 6) Asymmetric capability — only some equipment can run some products
    specs.append(BenchmarkSpec(
        name="benchmark_06",
        description="Asymmetric capability: T5833 runs P1 only; MAGNUM5 runs P2 only.",
        rule_timekey="2026051707000006",
        targets=[("P1", "OP10", 200.0), ("P2", "OP10", 200.0)],
        oper_seq={("P1", "OP10"): 1, ("P2", "OP10"): 1},
        batch_map={("P1", "OP10"): "9C/92", ("P2", "OP10"): "9C/92"},
        equipment=[("9C/92", "T5833", 2), ("9C/92", "MAGNUM5", 2)],
        uph={("P1", "OP10", "T5833"): 100.0, ("P2", "OP10", "MAGNUM5"): 100.0},
        tool_qty=[("9C/92", "T5833", 2), ("9C/92", "MAGNUM5", 2)],
        eqp_model_groups={"G001": ["9C/92"]},
    ))

    # 7) Tight capacity — optimal is partial achievement; tests calibration
    specs.append(BenchmarkSpec(
        name="benchmark_07",
        description="Capacity-limited: plan exceeds full UPH; partial achievement expected.",
        rule_timekey="2026051707000007",
        targets=[("P1", "OP10", 800.0), ("P1", "OP20", 800.0)],
        oper_seq={("P1", "OP10"): 1, ("P1", "OP20"): 2},
        batch_map={("P1", "OP10"): "9C/92", ("P1", "OP20"): "9C/92"},
        equipment=[("9C/92", "T5833", 2)],
        uph={("P1", "OP10", "T5833"): 200.0, ("P1", "OP20", "T5833"): 200.0},
        tool_qty=[("9C/92", "T5833", 2)],
        eqp_model_groups={"G001": ["9C/92", "9C/102"]},
    ))

    # --- greedy-fail cases (RL must learn to avoid local-optimum traps) ---

    # 8) Specialist mismatch — greedy gives the fast generalist to product 1
    #    because the marginal is highest there; optimal sends the generalist
    #    to product 2 (only tool that can run it) and the specialist to P1.
    specs.append(BenchmarkSpec(
        name="benchmark_08",
        description="Greedy trap: versatile tool wasted on wrong target. Greedy 0.5, optimal 1.0.",
        rule_timekey="2026051707000008",
        targets=[("P1", "OP10", 100.0), ("P2", "OP10", 100.0)],
        oper_seq={("P1", "OP10"): 1, ("P2", "OP10"): 1},
        batch_map={("P1", "OP10"): "9C/92", ("P2", "OP10"): "9C/92"},
        equipment=[("9C/92", "T_FAST", 1), ("9C/92", "T_SLOW", 1)],
        uph={
            ("P1", "OP10", "T_FAST"): 300.0,
            ("P2", "OP10", "T_FAST"): 100.0,
            ("P1", "OP10", "T_SLOW"): 100.0,
        },
        tool_qty=[("9C/92", "T_FAST", 1), ("9C/92", "T_SLOW", 1)],
        eqp_model_groups={"G001": ["9C/92"]},
    ))

    # 9) Multi-op frontloading — greedy fills the upstream OP with the
    #    generalist; the specialist that can only run OP10 then has no work
    #    and OP20 is empty. Optimal swaps the assignments.
    specs.append(BenchmarkSpec(
        name="benchmark_09",
        description="Multi-op frontloading trap. Greedy 0.5, optimal 1.0.",
        rule_timekey="2026051707000009",
        targets=[("P1", "OP10", 200.0), ("P1", "OP20", 200.0)],
        oper_seq={("P1", "OP10"): 1, ("P1", "OP20"): 2},
        batch_map={("P1", "OP10"): "9C/92", ("P1", "OP20"): "9C/92"},
        equipment=[("9C/92", "T_GP", 1), ("9C/92", "T_OP10", 1)],
        uph={
            ("P1", "OP10", "T_GP"): 200.0,
            ("P1", "OP20", "T_GP"): 200.0,
            ("P1", "OP10", "T_OP10"): 200.0,
        },
        tool_qty=[("9C/92", "T_GP", 1), ("9C/92", "T_OP10", 1)],
        eqp_model_groups={"G001": ["9C/92"]},
    ))

    # 10) Downstream bottleneck — 3 ops, GP tool can run all but specialists
    #     only cover OP10 and OP20. Greedy uses the GP tool for OP10 (because
    #     it iterates targets in order) and leaves OP30 empty.
    specs.append(BenchmarkSpec(
        name="benchmark_10",
        description="3-op line; GP tool must go to terminal OP. Greedy 0.667, optimal 1.0.",
        rule_timekey="2026051707000010",
        targets=[("P1", "OP10", 100.0), ("P1", "OP20", 100.0), ("P1", "OP30", 100.0)],
        oper_seq={("P1", "OP10"): 1, ("P1", "OP20"): 2, ("P1", "OP30"): 3},
        batch_map={("P1", "OP10"): "9C/92", ("P1", "OP20"): "9C/92", ("P1", "OP30"): "9C/92"},
        equipment=[("9C/92", "T_GP", 1), ("9C/92", "T_OP10", 1), ("9C/92", "T_OP20", 1)],
        uph={
            ("P1", "OP10", "T_GP"): 100.0,
            ("P1", "OP20", "T_GP"): 100.0,
            ("P1", "OP30", "T_GP"): 100.0,
            ("P1", "OP10", "T_OP10"): 100.0,
            ("P1", "OP20", "T_OP20"): 100.0,
        },
        tool_qty=[("9C/92", "T_GP", 1), ("9C/92", "T_OP10", 1), ("9C/92", "T_OP20", 1)],
        eqp_model_groups={"G001": ["9C/92"]},
    ))

    # 11) WIP shortage at downstream op — even with enough equipment, the
    #     queue at OP20 only holds 50 units, so OP20 cannot exceed 50/200=0.25.
    #     Validates that the simulator/heuristic/optimizer respect WIP.
    specs.append(BenchmarkSpec(
        name="benchmark_11",
        description="WIP-limited downstream: OP20 has only 50 WIP. Optimal 0.625, not 1.0.",
        rule_timekey="2026051707000011",
        targets=[("P1", "OP10", 200.0), ("P1", "OP20", 200.0)],
        oper_seq={("P1", "OP10"): 1, ("P1", "OP20"): 2},
        batch_map={("P1", "OP10"): "9C/92", ("P1", "OP20"): "9C/92"},
        equipment=[("9C/92", "T5833", 2)],
        uph={("P1", "OP10", "T5833"): 200.0, ("P1", "OP20", "T5833"): 200.0},
        tool_qty=[("9C/92", "T5833", 2)],
        eqp_model_groups={"G001": ["9C/92"]},
        wip={("P1", "OP10"): 999999.0, ("P1", "OP20"): 50.0},
    ))
    return specs


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    for spec in _specs():
        _write_dataset(spec)
        print(f"generated {spec.name}")


if __name__ == "__main__":
    main()
