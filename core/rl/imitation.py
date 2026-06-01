"""Behavioral cloning warm-start from expert (improved greedy) trajectories."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from core.domain import SchedulingDataset
from core.rl.env import MAX_MODELS, SchedulingEnv


class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def collect_demonstrations(dataset: SchedulingDataset, n_episodes: int = 5) -> tuple[np.ndarray, np.ndarray]:
    env = SchedulingEnv(dataset)
    obs_list: list[np.ndarray] = []
    act_list: list[int] = []
    expert = env.expert_conversions()
    # Action index aligns with sorted model slot (same as env._model_index).
    model_to_idx = {m: i for i, m in enumerate(env.models)}

    for conv in expert:
        obs, _ = env.reset()
        action = model_to_idx.get(conv.eqp_model_cd, 0)
        obs_list.append(obs)
        act_list.append(action)
        env.step(action)

    while len(obs_list) < n_episodes * max(1, len(env.oper_keys)):
        obs, _ = env.reset()
        for i, key in enumerate(env.oper_keys):
            plan_prod_key, oper_id = key
            models = [
                m.eqp_model_cd
                for m in dataset.model_uph
                if m.plan_prod_key == plan_prod_key and m.oper_id == oper_id
            ]
            best = (
                max(models, key=lambda m: dataset.uph(plan_prod_key, oper_id, m) or 0)
                if models
                else env.models[0]
            )
            action = min(model_to_idx.get(best, 0), MAX_MODELS - 1)
            obs_list.append(obs)
            act_list.append(action)
            obs, _, term, trunc, _ = env.step(action)
            if term or trunc:
                break

    return np.stack(obs_list), np.array(act_list, dtype=np.int64)


def train_behavioral_cloning(
    datasets: list[SchedulingDataset],
    save_path: str | Path,
    epochs: int = 20,
    batch_size: int = 32,
) -> Path:
    all_obs: list[np.ndarray] = []
    all_act: list[int] = []
    for ds in datasets:
        o, a = collect_demonstrations(ds)
        all_obs.append(o)
        all_act.append(a)

    obs = np.concatenate(all_obs, axis=0)
    acts = np.concatenate(all_act, axis=0)
    env0 = SchedulingEnv(datasets[0])
    model = PolicyNetwork(obs.shape[1], env0.action_space.n)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loader = DataLoader(TensorDataset(torch.from_numpy(obs), torch.from_numpy(acts)), batch_size=batch_size, shuffle=True)

    for _ in range(epochs):
        for batch_obs, batch_act in loader:
            logits = model(batch_obs)
            loss = nn.functional.cross_entropy(logits, batch_act)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "obs_dim": obs.shape[1], "n_actions": env0.action_space.n}, path)
    return path
