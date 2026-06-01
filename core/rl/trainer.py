"""PPO training with optional imitation-learning warm start."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Callable

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from core.domain import ConversionRecord, SchedulingDataset
from core.optimizer import ImprovedGreedySolver
from core.rl.env import SchedulingEnv
from core.rl.imitation import train_behavioral_cloning


def _make_env(dataset: SchedulingDataset):
    return SchedulingEnv(dataset)


def train_ppo(
    datasets: list[SchedulingDataset],
    total_timesteps: int = 50_000,
    model_path: str | Path = "artifacts/models/ppo_scheduling",
    imitation_path: str | Path | None = None,
) -> Path:
    if not datasets:
        raise ValueError("At least one dataset required for training")

    def env_fn():
        ds = random.choice(datasets)
        return SchedulingEnv(ds)

    vec_env = DummyVecEnv([env_fn])
    model = PPO("MlpPolicy", vec_env, verbose=1, learning_rate=3e-4, n_steps=512, batch_size=64)
    model.learn(total_timesteps=total_timesteps)

    out = Path(model_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(out))
    return out


def train_with_imitation_then_ppo(
    datasets: list[SchedulingDataset],
    total_timesteps: int = 50_000,
    imitation_epochs: int = 20,
    model_dir: str | Path = "artifacts/models",
) -> Path:
    model_dir = Path(model_dir)
    il_path = model_dir / "bc_policy.pt"
    train_behavioral_cloning(datasets, il_path, epochs=imitation_epochs)
    ppo_path = train_ppo(datasets, total_timesteps=total_timesteps, model_path=model_dir / "ppo_scheduling", imitation_path=il_path)
    return ppo_path


def infer_conversions(
    dataset: SchedulingDataset,
    model_path: str | Path = "artifacts/models/ppo_scheduling",
) -> list[ConversionRecord]:
    path = Path(model_path)
    if path.with_suffix(".zip").exists():
        model = PPO.load(str(path))
    elif path.exists():
        model = PPO.load(str(path))
    else:
        return ImprovedGreedySolver().solve(dataset)

    env = SchedulingEnv(dataset)
    if model.observation_space.shape != env.observation_space.shape:
        # Old checkpoint trained with variable obs size — retrain or use heuristic.
        return ImprovedGreedySolver().solve(dataset)

    obs, _ = env.reset()
    conversions: list[ConversionRecord] = []
    terminated = truncated = False
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(int(action))
        conversions = list(info.get("conversions", conversions))
    return conversions
