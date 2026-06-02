"""Gymnasium environment wrapping `SchedulingProblem` for PPO training.

Action structure: at each step, the agent assigns one equipment unit from a
(batch, eqp_model) bucket to a (plan_prod_key, oper) target — or NO-OP. The
episode ends when all equipment is placed or NO-OP is selected.

We keep the action space discrete and bounded across the env's lifetime so
PPO sees a stable head. Problems with shapes larger than the bounds fall back
to greedy at inference time.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .domain import Allocation, AllocationSet, SchedulingProblem
from .simulator import Simulator, count_switches

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception:  # pragma: no cover
    gym = None  # type: ignore
    spaces = None  # type: ignore


class DispatchEnv(gym.Env if gym is not None else object):
    """Single-snapshot dispatch environment."""

    metadata = {"render_modes": []}

    MAX_BUCKETS = 16
    MAX_TARGETS = 32

    def __init__(
        self,
        problems: List[SchedulingProblem],
        switch_penalty: float = 0.02,
        achievement_weight: float = 1.0,
        seed: Optional[int] = None,
    ) -> None:
        if gym is None:
            raise RuntimeError("gymnasium is required for DispatchEnv; install pjt_scheduler[rl].")
        super().__init__()
        if not problems:
            raise ValueError("At least one SchedulingProblem is required.")
        self.problems = problems
        self.switch_penalty = float(switch_penalty)
        self.achievement_weight = float(achievement_weight)
        self._rng = np.random.default_rng(seed)

        # action: bucket_idx * (MAX_TARGETS + 1) + target_idx_or_noop
        self.action_space = spaces.Discrete(self.MAX_BUCKETS * (self.MAX_TARGETS + 1))

        # observation: for each bucket, [free_units]; for each target, [shortfall];
        # plus availability/UPH matrix flattened.
        obs_size = (
            self.MAX_BUCKETS
            + self.MAX_TARGETS
            + self.MAX_BUCKETS * self.MAX_TARGETS
        )
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(obs_size,), dtype=np.float32)

        self._reset_internals()

    # ------------------------------------------------------------------
    def _reset_internals(self) -> None:
        self.problem: Optional[SchedulingProblem] = None
        self.bucket_keys: List[Tuple[str, str]] = []
        self.target_keys: List[Tuple[str, str]] = []
        self.bucket_free: np.ndarray = np.zeros(self.MAX_BUCKETS, dtype=np.int32)
        self.target_shortfall: np.ndarray = np.zeros(self.MAX_TARGETS, dtype=np.float32)
        self.target_plan: np.ndarray = np.zeros(self.MAX_TARGETS, dtype=np.float32)
        self.uph_matrix: np.ndarray = np.zeros((self.MAX_BUCKETS, self.MAX_TARGETS), dtype=np.float32)
        self.avail_matrix: np.ndarray = np.zeros((self.MAX_BUCKETS, self.MAX_TARGETS), dtype=np.float32)
        self._allocations: List[Allocation] = []
        self._steps = 0

    def _load_problem(self, problem: SchedulingProblem) -> None:
        self._reset_internals()
        self.problem = problem
        pool = problem.equipment_pool()
        bucket_items = list(pool.items())[: self.MAX_BUCKETS]
        self.bucket_keys = [bm for bm, _ in bucket_items]
        for i, (_, qty) in enumerate(bucket_items):
            self.bucket_free[i] = int(qty)

        targets = problem.plan_targets()[: self.MAX_TARGETS]
        self.target_keys = [(pk, op) for pk, op, _ in targets]
        for j, (_, _, plan_qty) in enumerate(targets):
            self.target_plan[j] = float(plan_qty)
            self.target_shortfall[j] = float(plan_qty)

        plan_scale = max(1.0, float(self.target_plan.max()))
        for i, (b_id, model) in enumerate(self.bucket_keys):
            for j, (pk, op) in enumerate(self.target_keys):
                if problem.is_available(pk, op, model) and (
                    b_id == problem.batch_of(pk, op) or model in problem.model_group_of(model)
                ):
                    self.avail_matrix[i, j] = 1.0
                    self.uph_matrix[i, j] = problem.uph_of(pk, op, model) / plan_scale

    def _observation(self) -> np.ndarray:
        plan_scale = max(1.0, float(self.target_plan.max())) if self.target_plan.size else 1.0
        bucket_scale = max(1.0, float(self.bucket_free.max())) if self.bucket_free.size else 1.0
        bucket_obs = self.bucket_free.astype(np.float32) / bucket_scale
        target_obs = self.target_shortfall.astype(np.float32) / plan_scale
        matrix_obs = (self.uph_matrix * self.avail_matrix).reshape(-1)
        return np.concatenate([bucket_obs, target_obs, matrix_obs]).astype(np.float32)

    # ------------------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        idx = int(self._rng.integers(0, len(self.problems)))
        self._load_problem(self.problems[idx])
        return self._observation(), {}

    def step(self, action: int):
        assert self.problem is not None
        bucket_idx, target_or_noop = divmod(int(action), self.MAX_TARGETS + 1)
        is_noop = target_or_noop == self.MAX_TARGETS
        reward = 0.0
        terminated = False
        info: Dict = {}

        self._steps += 1
        if is_noop or bucket_idx >= len(self.bucket_keys):
            terminated = True
        else:
            target_idx = target_or_noop
            if (
                target_idx >= len(self.target_keys)
                or self.bucket_free[bucket_idx] <= 0
                or self.avail_matrix[bucket_idx, target_idx] <= 0.0
            ):
                reward -= 0.01  # invalid action shaping
            else:
                b_id, model = self.bucket_keys[bucket_idx]
                pk, op = self.target_keys[target_idx]
                uph = self.problem.uph_of(pk, op, model)
                self.bucket_free[bucket_idx] -= 1
                marginal = min(self.target_shortfall[target_idx], uph)
                self.target_shortfall[target_idx] = max(0.0, self.target_shortfall[target_idx] - uph)
                plan = max(1.0, float(self.target_plan[target_idx]))
                reward += self.achievement_weight * (marginal / plan) / max(1, len(self.target_keys))
                # log allocation
                target_batch = self.problem.batch_of(pk, op) or b_id
                merged = False
                for a in self._allocations:
                    if (
                        a.plan_prod_key == pk
                        and a.oper_id == op
                        and a.batch_id == target_batch
                        and a.eqp_model_cd == model
                    ):
                        a.eqp_qty += 1
                        merged = True
                        break
                if not merged:
                    self._allocations.append(
                        Allocation(
                            batch_id=target_batch,
                            plan_prod_key=pk,
                            oper_id=op,
                            eqp_model_cd=model,
                            eqp_qty=1,
                        )
                    )
                # switch penalty when cross-batch
                if b_id != target_batch:
                    reward -= self.switch_penalty

        if int(self.bucket_free.sum()) == 0:
            terminated = True
        if self._steps >= self.MAX_BUCKETS * self.MAX_TARGETS:
            terminated = True

        info["allocation_count"] = len(self._allocations)
        return self._observation(), float(reward), terminated, False, info

    # ------------------------------------------------------------------
    def current_allocation(self) -> AllocationSet:
        return AllocationSet(
            rule_timekey=self.problem.rule_timekey if self.problem else "",
            allocations=list(self._allocations),
        )
