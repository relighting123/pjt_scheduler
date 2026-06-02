"""RL inference. Falls back to greedy when PPO/SB3 isn't installed."""
from __future__ import annotations

from typing import Optional

from .domain import AllocationSet, SchedulingProblem
from .heuristic import greedy_allocate


def infer(
    problem: SchedulingProblem,
    model_path: Optional[str] = None,
    ignore_wip: bool = False,
) -> AllocationSet:
    if not model_path:
        return greedy_allocate(problem, ignore_wip=ignore_wip)
    try:
        from stable_baselines3 import PPO  # noqa: F401
        from .rl_env import DispatchEnv
        from .rl_train import load_policy
    except Exception:
        return greedy_allocate(problem, ignore_wip=ignore_wip)

    model = load_policy(model_path)
    if model is None:
        return greedy_allocate(problem, ignore_wip=ignore_wip)
    env = DispatchEnv([problem], ignore_wip=ignore_wip)
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(int(action))
        done = terminated or truncated
    allocation = env.current_allocation()
    # safety net: empty allocation falls back to greedy
    if not allocation.allocations:
        return greedy_allocate(problem, ignore_wip=ignore_wip)
    return allocation
