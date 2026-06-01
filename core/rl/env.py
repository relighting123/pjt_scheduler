"""Gymnasium environment for equipment allocation actions."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from core.domain import ConversionRecord, SchedulingDataset
from core.optimizer import ImprovedGreedySolver
from core.simulator import SchedulingSimulator

# Fixed sizes so PPO observation/action spaces match across all benchmarks and DB snapshots.
MAX_OPERS = 8
MAX_MODELS = 8
OBS_DIM = MAX_OPERS * 3 + MAX_MODELS * 2 + 2


class SchedulingEnv(gym.Env):
    """Discrete action: assign next oper to one equipment model slot (fixed action space)."""

    metadata = {"render_modes": []}

    def __init__(self, dataset: SchedulingDataset, max_steps: int = 32):
        super().__init__()
        self.dataset = dataset
        self.max_steps = max_steps
        self.oper_keys = dataset.oper_keys()
        self.models = sorted({m.eqp_model_cd for m in dataset.model_uph})
        if not self.models:
            self.models = ["DEFAULT"]
        self.n_opers = len(self.oper_keys)
        self.n_models = len(self.models)
        self.action_space = spaces.Discrete(MAX_MODELS)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32
        )
        self._step_idx = 0
        self._conversions: list[ConversionRecord] = []
        self._oper_idx = 0

    def _model_index(self, action: int) -> int:
        if self.n_models <= 0:
            return 0
        return int(action) % self.n_models

    def _obs(self) -> np.ndarray:
        plan = self.dataset.plan_qty_by_oper()
        max_plan = max(plan.values()) if plan else 1.0
        vec: list[float] = []

        for i in range(MAX_OPERS):
            if i < self.n_opers:
                key = self.oper_keys[i]
                pq = plan.get(key, 0.0) / max_plan
                batch = self.dataset.batch_for(key[0], key[1]) or ""
                vec.extend([pq, float(hash(batch) % 100) / 100.0, 1.0 if pq > 0 else 0.0])
            else:
                vec.extend([0.0, 0.0, 0.0])

        for i in range(MAX_MODELS):
            if i < self.n_models:
                model = self.models[i]
                vec.append(self.dataset.total_eqp_qty(model) / 10.0)
                vec.append(
                    1.0
                    if any(
                        self.dataset.is_available(k[0], k[1], model) for k in self.oper_keys
                    )
                    else 0.0
                )
            else:
                vec.extend([0.0, 0.0])

        sim = SchedulingSimulator(self.dataset)
        res = sim.simulate(self._conversions)
        vec.extend([res.avg_achievement_rate, self._step_idx / self.max_steps])

        out = np.array(vec, dtype=np.float32)
        assert out.shape == (OBS_DIM,), f"obs shape {out.shape} != ({OBS_DIM},)"
        return out

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._step_idx = 0
        self._oper_idx = 0
        self._conversions = []
        return self._obs(), {}

    def step(self, action: int):
        self._step_idx += 1
        terminated = False
        truncated = self._step_idx >= self.max_steps

        if self._oper_idx < self.n_opers:
            plan_prod_key, oper_id = self.oper_keys[self._oper_idx]
            model = self.models[self._model_index(action)]
            batch_id = self.dataset.batch_for(plan_prod_key, oper_id)
            if batch_id and self.dataset.is_available(plan_prod_key, oper_id, model):
                self._conversions.append(
                    ConversionRecord(
                        rule_timekey=self.dataset.rule_timekey,
                        from_batch="",
                        from_plan_prod_key=plan_prod_key,
                        from_oper_id=oper_id,
                        eqp_model_cd=model,
                        to_batch_id=batch_id,
                        to_plan_prod_key=plan_prod_key,
                        to_oper_id=oper_id,
                        start_conv_time=self.dataset.rule_timekey,
                        eqp_qty=1,
                    )
                )
            self._oper_idx += 1

        sim = SchedulingSimulator(self.dataset)
        result = sim.simulate(self._conversions)
        reward = result.avg_achievement_rate * 10.0 - 0.05 * result.conversion_count

        if self._oper_idx >= self.n_opers:
            terminated = True

        return self._obs(), float(reward), terminated, truncated, {"conversions": self._conversions}

    def expert_conversions(self) -> list[ConversionRecord]:
        return ImprovedGreedySolver().solve(self.dataset)
