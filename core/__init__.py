"""Core components grouped by purpose.

Sub-packages:

    core.sim         simulators (single-snapshot + multi-period)
    core.policy      non-RL policies (greedy heuristic, brute-force optimal)
    core.rl          reinforcement learning (env, train, infer)
    core.evaluation  benchmark evaluation + HTML/MD report

Single-file modules:

    core.domain      domain model (record dataclasses, SchedulingProblem)
    core.db          Oracle connection helpers (lazy oracledb import)

Top-level shortcuts re-export the most commonly used types so simple
callers can stick to `from core import X`.
"""
from .domain import (
    SchedulingProblem,
    WipRecord,
    UphRecord,
    EquipmentRecord,
    AvailabilityRecord,
    ToolGroupRecord,
    ToolQtyRecord,
    PlanRecord,
    Allocation,
    AllocationSet,
)

__all__ = [
    "SchedulingProblem",
    "WipRecord",
    "UphRecord",
    "EquipmentRecord",
    "AvailabilityRecord",
    "ToolGroupRecord",
    "ToolQtyRecord",
    "PlanRecord",
    "Allocation",
    "AllocationSet",
]
