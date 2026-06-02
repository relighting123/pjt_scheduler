"""Multi-period Gym environment for PPO over the WIP-flow simulator.

Each Gym step is *one unit assignment* (same as the single-slot env), but the
episode now spans multiple slots:

  bucket * (MAX_TARGETS+1) + target   == assign one unit
  bucket * (MAX_TARGETS+1) + MAX_TARGETS  == NO-OP, finalizes the current slot

When NO-OP is chosen (or all buckets in the slot are exhausted), we commit the
slot: production = sum of assigned UPH this slot, capped by WIP; cumulative
production accrues; downstream WIP receives this slot's output for the *next*
slot (one-slot latency, matching `MultiPeriodSimulator`). Then `prev_alloc`,
`bucket_free`, `target_wip` and `target_shortfall` are refreshed for the next
slot. The episode terminates after `num_slots` commits.

Reward shape:
  - per substep:  capped marginal toward remaining plan, normalised per-target
    (positive shaping — does not double-count beyond actual production).
  - at commit:    switch penalty for units that moved into a new (batch, model)
    vs. last slot, reflecting the production lost to `switch_time_hours` in the
    real simulator.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .domain import Allocation, AllocationSet, SchedulingProblem

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception:  # pragma: no cover
    gym = None  # type: ignore
    spaces = None  # type: ignore


class MultiPeriodDispatchEnv(gym.Env if gym is not None else object):
    metadata = {"render_modes": []}

    MAX_BUCKETS = 16
    MAX_TARGETS = 32

    def __init__(
        self,
        problems: List[SchedulingProblem],
        num_slots: int = 4,
        slot_hours: float = 1.0,
        switch_time_hours: float = 0.0,
        achievement_weight: float = 1.0,
        seed: Optional[int] = None,
    ) -> None:
        if gym is None:
            raise RuntimeError("gymnasium is required; install pjt_scheduler[rl].")
        super().__init__()
        if not problems:
            raise ValueError("At least one SchedulingProblem is required.")
        self.problems = problems
        self.num_slots = int(num_slots)
        self.slot_hours = float(slot_hours)
        self.switch_time_hours = float(switch_time_hours)
        self.achievement_weight = float(achievement_weight)
        self._rng = np.random.default_rng(seed)

        self.action_space = spaces.Discrete(self.MAX_BUCKETS * (self.MAX_TARGETS + 1))
        # observation: bucket_free, target_shortfall, target_wip, uph_matrix, prev_mask, slot_progress
        obs_size = (
            self.MAX_BUCKETS
            + self.MAX_TARGETS * 2
            + self.MAX_BUCKETS * self.MAX_TARGETS
            + self.MAX_BUCKETS
            + 1
        )
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(obs_size,), dtype=np.float32)
        self._reset_internals()

    # ------------------------------------------------------------------
    def _reset_internals(self) -> None:
        self.problem: Optional[SchedulingProblem] = None
        self.bucket_keys: List[Tuple[str, str]] = []
        self.target_keys: List[Tuple[str, str]] = []
        self.bucket_free: np.ndarray = np.zeros(self.MAX_BUCKETS, dtype=np.int32)
        self._bucket_initial: np.ndarray = np.zeros(self.MAX_BUCKETS, dtype=np.int32)
        self.target_shortfall: np.ndarray = np.zeros(self.MAX_TARGETS, dtype=np.float32)
        self.target_plan: np.ndarray = np.zeros(self.MAX_TARGETS, dtype=np.float32)
        self.target_wip: np.ndarray = np.zeros(self.MAX_TARGETS, dtype=np.float32)
        self.uph_matrix: np.ndarray = np.zeros((self.MAX_BUCKETS, self.MAX_TARGETS), dtype=np.float32)
        self.avail_matrix: np.ndarray = np.zeros((self.MAX_BUCKETS, self.MAX_TARGETS), dtype=np.float32)
        self.prev_bucket_target: np.ndarray = np.full(self.MAX_BUCKETS, -1, dtype=np.int32)
        self.slot_idx: int = 0
        self._this_slot_alloc: List[Allocation] = []
        self._prev_alloc: Optional[AllocationSet] = None
        self._cumulative: Dict[Tuple[str, str], float] = {}
        self._slot_consumed_prev: Dict[Tuple[str, str], int] = {}

    # ------------------------------------------------------------------
    def _load_problem(self, problem: SchedulingProblem) -> None:
        self._reset_internals()
        self.problem = problem
        pool = problem.equipment_pool()
        bucket_items = list(pool.items())[: self.MAX_BUCKETS]
        self.bucket_keys = [bm for bm, _ in bucket_items]
        for i, (_, qty) in enumerate(bucket_items):
            self.bucket_free[i] = int(qty)
            self._bucket_initial[i] = int(qty)

        targets = problem.plan_targets()[: self.MAX_TARGETS]
        self.target_keys = [(pk, op) for pk, op, _ in targets]
        plan_scale = 1.0
        for j, (pk, op, plan_qty) in enumerate(targets):
            self.target_plan[j] = float(plan_qty)
            self.target_shortfall[j] = float(plan_qty)
            w = problem.wip_of(pk, op)
            self.target_wip[j] = float(w) if w > 0 else 0.0  # 0 == empty queue (multi-period semantics)
            plan_scale = max(plan_scale, float(plan_qty))

        for i, (b_id, model) in enumerate(self.bucket_keys):
            for j, (pk, op) in enumerate(self.target_keys):
                if problem.is_available(pk, op, model) and (
                    b_id == problem.batch_of(pk, op) or model in problem.model_group_of(model)
                ):
                    self.avail_matrix[i, j] = 1.0
                    self.uph_matrix[i, j] = problem.uph_of(pk, op, model) * self.slot_hours / plan_scale

        self._cumulative = {k: 0.0 for k in self.target_keys}

    # ------------------------------------------------------------------
    def _observation(self) -> np.ndarray:
        plan_scale = max(1.0, float(self.target_plan.max())) if self.target_plan.size else 1.0
        bucket_scale = max(1.0, float(self._bucket_initial.max())) if self._bucket_initial.size else 1.0
        bucket_obs = self.bucket_free.astype(np.float32) / bucket_scale
        shortfall_obs = self.target_shortfall.astype(np.float32) / plan_scale
        wip_obs = np.clip(self.target_wip.astype(np.float32) / plan_scale, 0.0, 1.0)
        matrix_obs = (self.uph_matrix * self.avail_matrix).reshape(-1)
        prev_obs = (self.prev_bucket_target.astype(np.float32) + 1.0) / (self.MAX_TARGETS + 1)
        slot_obs = np.array([self.slot_idx / max(1, self.num_slots)], dtype=np.float32)
        return np.concatenate([bucket_obs, shortfall_obs, wip_obs, matrix_obs, prev_obs, slot_obs]).astype(np.float32)

    # ------------------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        idx = int(self._rng.integers(0, len(self.problems)))
        self._load_problem(self.problems[idx])
        return self._observation(), {}

    # ------------------------------------------------------------------
    def _commit_slot(self) -> float:
        """Apply this slot's allocation to the live state. Returns switch-cost reward (<=0)."""
        problem = self.problem
        assert problem is not None
        alloc = AllocationSet(rule_timekey=problem.rule_timekey, allocations=list(self._this_slot_alloc))

        # WIP flow + cumulative update. Production this slot already incorporates
        # switch cost via the per-substep shaping (UPH * effective_hours rolled in
        # by capping with target_wip / shortfall at substep time).
        # Here we just count switches for the negative reward and advance state.
        first_slot = self._prev_alloc is None
        prev_by_bm: Dict[Tuple[str, str], int] = {}
        if not first_slot:
            for a in self._prev_alloc.allocations:  # type: ignore[union-attr]
                prev_by_bm[(a.batch_id, a.eqp_model_cd)] = prev_by_bm.get((a.batch_id, a.eqp_model_cd), 0) + a.eqp_qty
        consumed: Dict[Tuple[str, str], int] = {}
        switch_reward = 0.0
        for a in alloc.allocations:
            bm = (a.batch_id, a.eqp_model_cd)
            if first_slot:
                continue
            kept_avail = max(0, prev_by_bm.get(bm, 0) - consumed.get(bm, 0))
            kept = min(a.eqp_qty, kept_avail)
            fresh = a.eqp_qty - kept
            consumed[bm] = consumed.get(bm, 0) + kept
            if fresh > 0:
                # production lost to setup ≈ uph * fresh * switch_time_hours / plan
                uph = problem.uph_of(a.plan_prod_key, a.oper_id, a.eqp_model_cd)
                plan = max(1.0, float(problem.plan_qty_of(a.plan_prod_key, a.oper_id)))
                lost = uph * fresh * self.switch_time_hours
                switch_reward -= self.achievement_weight * (lost / plan) / max(1, len(self.target_keys))

        # propagate produced -> downstream WIP (one-slot latency)
        produced_this_slot: Dict[Tuple[str, str], float] = {}
        # produced amount per (pk, op) = sum over alloc of effective UPH (already
        # bounded by shortfall + wip at substep time)
        for a in alloc.allocations:
            uph = problem.uph_of(a.plan_prod_key, a.oper_id, a.eqp_model_cd)
            # use the actual depletion the substep performed: difference between
            # the slot-start wip/shortfall and current value would be ideal, but
            # cheaper to recompute as min(uph*qty, remaining)
            produced_this_slot[(a.plan_prod_key, a.oper_id)] = (
                produced_this_slot.get((a.plan_prod_key, a.oper_id), 0.0)
                + uph * a.eqp_qty * self.slot_hours
            )
        # cap each by the slot-start WIP (which has already been depleted by substeps;
        # we therefore cap by initial slot wip stored separately — see _slot_start_wip)
        for key, q in produced_this_slot.items():
            actual = min(q, self._slot_start_wip.get(key, q), self._slot_start_shortfall.get(key, q))
            self._cumulative[key] = self._cumulative.get(key, 0.0) + actual
            nxt = problem.next_oper_of(key[0], key[1])
            if nxt is not None:
                # downstream WIP receives this slot's output, visible next slot
                # by finding its index in target_keys
                if (key[0], nxt) in self.target_keys:
                    j = self.target_keys.index((key[0], nxt))
                    self.target_wip[j] += actual

        # advance to next slot
        self.slot_idx += 1
        self._prev_alloc = alloc
        self._this_slot_alloc = []
        self.bucket_free[:] = self._bucket_initial
        # refresh shortfall = plan - cumulative
        for j, key in enumerate(self.target_keys):
            self.target_shortfall[j] = max(0.0, self.target_plan[j] - self._cumulative.get(key, 0.0))
        # mark slot-start snapshots for the new slot
        self._slot_start_wip = {k: float(self.target_wip[i]) for i, k in enumerate(self.target_keys)}
        self._slot_start_shortfall = {k: float(self.target_shortfall[i]) for i, k in enumerate(self.target_keys)}
        return switch_reward

    # ------------------------------------------------------------------
    def step(self, action: int):
        assert self.problem is not None
        if not hasattr(self, "_slot_start_wip"):
            self._slot_start_wip = {k: float(self.target_wip[i]) for i, k in enumerate(self.target_keys)}
            self._slot_start_shortfall = {k: float(self.target_shortfall[i]) for i, k in enumerate(self.target_keys)}

        bucket_idx, target_or_noop = divmod(int(action), self.MAX_TARGETS + 1)
        is_noop = target_or_noop == self.MAX_TARGETS
        reward = 0.0
        terminated = False
        info: Dict = {}

        commit_now = is_noop or bucket_idx >= len(self.bucket_keys)
        if not commit_now:
            target_idx = target_or_noop
            if (
                target_idx >= len(self.target_keys)
                or self.bucket_free[bucket_idx] <= 0
                or self.avail_matrix[bucket_idx, target_idx] <= 0.0
                or float(self.target_wip[target_idx]) <= 0.0
            ):
                # invalid / wasteful substep: no allocation, small shaping
                reward -= 0.01
            else:
                b_id, model = self.bucket_keys[bucket_idx]
                pk, op = self.target_keys[target_idx]
                uph = self.problem.uph_of(pk, op, model)
                self.bucket_free[bucket_idx] -= 1
                wip_left = float(self.target_wip[target_idx])
                effective_uph = min(uph * self.slot_hours, wip_left)
                marginal = min(float(self.target_shortfall[target_idx]), effective_uph)
                self.target_shortfall[target_idx] = max(0.0, float(self.target_shortfall[target_idx]) - effective_uph)
                self.target_wip[target_idx] = max(0.0, wip_left - effective_uph)
                plan = max(1.0, float(self.target_plan[target_idx]))
                reward += self.achievement_weight * (marginal / plan) / max(1, len(self.target_keys))
                # record allocation
                target_batch = self.problem.batch_of(pk, op) or b_id
                merged = False
                for a in self._this_slot_alloc:
                    if (a.plan_prod_key == pk and a.oper_id == op
                            and a.batch_id == target_batch and a.eqp_model_cd == model):
                        a.eqp_qty += 1
                        merged = True
                        break
                if not merged:
                    self._this_slot_alloc.append(
                        Allocation(batch_id=target_batch, plan_prod_key=pk,
                                   oper_id=op, eqp_model_cd=model, eqp_qty=1)
                    )
                self.prev_bucket_target[bucket_idx] = target_idx

        # auto-commit when buckets are empty so the agent never spends a step
        # on a no-op forced by saturation.
        if commit_now or self.bucket_free.sum() == 0:
            reward += self._commit_slot()
            if self.slot_idx >= self.num_slots:
                terminated = True

        info["slot"] = self.slot_idx
        info["cumulative"] = dict(self._cumulative)
        return self._observation(), float(reward), terminated, False, info

    # ------------------------------------------------------------------
    def current_schedule(self) -> List[AllocationSet]:
        sched: List[AllocationSet] = []
        if self._prev_alloc is not None:
            sched.append(self._prev_alloc)
        return sched
