"""RL inference. Falls back to greedy when PPO/SB3 isn't installed."""
from __future__ import annotations

from typing import Optional

from ..domain import AllocationSet, SchedulingProblem
from ..policy.heuristic import greedy_allocate


def infer(
    problem: SchedulingProblem,
    model_path: Optional[str] = None,
    ignore_wip: bool = False,
) -> AllocationSet:
    """단일 스냅샷 RL 추론. 모델 없으면/실패하면 greedy로 자동 폴백.

    Args:
        problem: 추론할 SchedulingProblem.
        model_path: 학습된 MaskablePPO .zip 경로 (없으면 None).
        ignore_wip: True면 plan-only 모드.

    Returns:
        AllocationSet — RL이 결정한 할당 (또는 greedy 폴백).

    Example:
        alloc = infer(problem, "artifacts/models/ppo_dispatch_wip_static.zip")
        # AllocationSet(rule_timekey="...", allocations=[Allocation(...), ...])
    """
    if not model_path:
        return greedy_allocate(problem, ignore_wip=ignore_wip)
    try:
        from .env import DispatchEnv
        from .train import load_policy
        import torch
        # MaskableCategorical validates before masking; disable global
        # distribution validation (see core/rl_train.py).
        torch.distributions.Distribution.set_default_validate_args(False)
    except Exception:
        return greedy_allocate(problem, ignore_wip=ignore_wip)

    model = load_policy(model_path)
    if model is None:
        return greedy_allocate(problem, ignore_wip=ignore_wip)
    env = DispatchEnv([problem], ignore_wip=ignore_wip)
    obs, _ = env.reset()
    done = False
    while not done:
        mask = env.action_masks()
        action, _ = model.predict(obs, deterministic=True, action_masks=mask)
        obs, _, terminated, truncated, _ = env.step(int(action))
        done = terminated or truncated
    allocation = env.current_allocation()
    # safety net: empty allocation falls back to greedy
    if not allocation.allocations:
        return greedy_allocate(problem, ignore_wip=ignore_wip)
    return allocation
