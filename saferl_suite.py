from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("WANDB_SILENT", "true")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import gymnasium as gym
import highway_env  # noqa: F401
import matplotlib
import numpy as np
import torch
import torch.nn as nn
from fsrl.data import FastCollector
from fsrl.policy import CPO, PPOLagrangian
from fsrl.trainer import OnpolicyTrainer
from fsrl.utils import BaseLogger
from tianshou.data import Batch, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from torch.distributions import Categorical

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager


def configure_matplotlib_fonts() -> None:
    windows_font_dir = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
    candidates = ["arial.ttf", "segoeui.ttf", "tahoma.ttf"]
    for filename in candidates:
        font_path = windows_font_dir / filename
        if font_path.exists():
            font_manager.fontManager.addfont(str(font_path))
            font_name = font_manager.FontProperties(fname=str(font_path)).get_name()
            matplotlib.rcParams["font.family"] = font_name
            matplotlib.rcParams["font.sans-serif"] = [font_name]
            return


configure_matplotlib_fonts()

DEFAULT_SCENARIOS = ("merge-v0", "highway-v0", "roundabout-v0", "intersection-v1")
DEFAULT_ALGORITHMS = ("ppo", "ppolag", "cpo")

BASE_ENV_CONFIG: dict[str, Any] = {
    "collision_reward": 0,
    "action": {"type": "DiscreteMetaAction"},
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 5,
        "features": ["presence", "x", "y", "vx", "vy"],
        "absolute": False,
        "normalize": True,
    },
}

SCENARIO_OVERRIDES: dict[str, dict[str, Any]] = {
    "merge-v0": {
        "vehicles_count": 20,
        "vehicles_density": 1.5,
        "high_speed_reward": 0.4,
    },
    "highway-v0": {
        "lanes_count": 4,
        "vehicles_count": 35,
        "vehicles_density": 1.2,
        "high_speed_reward": 0.4,
    },
    "roundabout-v0": {
        "vehicles_count": 16,
        "vehicles_density": 1.0,
        "high_speed_reward": 0.25,
    },
    "intersection-v1": {
        "vehicles_count": 10,
        "initial_vehicle_count": 10,
        "spawn_probability": 0.6,
    },
}

ALGORITHM_LABELS = {
    "ppo": "PPO",
    "ppolag": "PPOLag",
    "cpo": "CPO",
}

SCENARIO_LABELS = {
    "merge-v0": "Merge",
    "highway-v0": "Highway",
    "roundabout-v0": "Roundabout",
    "intersection-v1": "Intersection",
}


def normalize_algorithm(name: str) -> str:
    normalized = name.strip().lower().replace("-", "")
    if normalized in {"ppo", "unconstrainedppo"}:
        return "ppo"
    if normalized in {"ppolag", "ppolagrangian", "lagrangianppo"}:
        return "ppolag"
    if normalized == "cpo":
        return "cpo"
    raise ValueError(f"Unsupported algorithm: {name}")


def algorithm_label(name: str) -> str:
    return ALGORITHM_LABELS[normalize_algorithm(name)]


def scenario_label(env_id: str) -> str:
    return SCENARIO_LABELS.get(env_id, env_id)


@dataclass
class ExperimentConfig:
    algorithm: str = "ppolag"
    env_id: str = "merge-v0"
    seed: int = 42
    epochs: int = 20
    steps_per_epoch: int = 4000
    hidden_sizes: list[int] = field(default_factory=lambda: [64, 64])
    pi_lr: float = 3e-4
    vf_lr: float = 1e-3
    batch_size: int = 99999
    repeat_per_collect: int = 80
    episode_per_collect: int = 100
    buffer_size: int = 100000
    cost_limit: float = 0.1
    train_envs: int = 8
    test_envs: int = 4
    test_episodes: int = 10
    eval_episodes: int = 100
    device: str = "auto"
    output_dir: str = "results"
    quick: bool = False
    cpo_critic_iters: int = 20

    def __post_init__(self) -> None:
        self.algorithm = normalize_algorithm(self.algorithm)
        self.hidden_sizes = [int(size) for size in self.hidden_sizes]
        if self.quick:
            self.epochs = min(self.epochs, 10)
            self.eval_episodes = min(self.eval_episodes, 20)

    @property
    def tag(self) -> str:
        return self.algorithm

    @property
    def display_name(self) -> str:
        return algorithm_label(self.algorithm)

    @property
    def resolved_device(self) -> str:
        if self.algorithm == "cpo":
            return "cpu"
        if self.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device

    @property
    def trainer_repeat_per_collect(self) -> int:
        return 1 if self.algorithm == "cpo" else self.repeat_per_collect

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["device"] = self.resolved_device
        data["algorithm_label"] = self.display_name
        return data


class CostWrapper(gym.Wrapper):
    """Expose binary crash cost expected by the safe-RL trainers."""

    def step(self, action: Any):  # type: ignore[override]
        obs, reward, terminated, truncated, info = self.env.step(action)
        info["cost"] = 1.0 if info.get("crashed", False) else 0.0
        return obs, reward, terminated, truncated, info


def build_env_config(env_id: str) -> dict[str, Any]:
    config = dict(BASE_ENV_CONFIG)
    config.update(SCENARIO_OVERRIDES.get(env_id, {}))
    return config


def make_env(env_id: str = "merge-v0", render_mode: str | None = None) -> gym.Env:
    env = gym.make(env_id, render_mode=render_mode)
    env.unwrapped.configure(build_env_config(env_id))
    env.reset()
    env = gym.wrappers.FlattenObservation(env)
    env = CostWrapper(env)
    return env


class DiscreteActor(nn.Module):
    """Categorical policy network for highway-env discrete controls."""

    def __init__(self, state_dim: int, action_dim: int, hidden_sizes: Sequence[int] = (64, 64)):
        super().__init__()
        layers: list[nn.Module] = []
        prev = state_dim
        for hidden in hidden_sizes:
            layers.extend([nn.Linear(prev, hidden), nn.Tanh()])
            prev = hidden
        layers.append(nn.Linear(prev, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, obs, state=None, **kwargs):
        if isinstance(obs, np.ndarray):
            obs = torch.as_tensor(obs, dtype=torch.float32, device=next(self.parameters()).device)
        return self.net(obs), state


class CriticNet(nn.Module):
    """Value or cost critic network."""

    def __init__(self, state_dim: int, hidden_sizes: Sequence[int] = (64, 64)):
        super().__init__()
        layers: list[nn.Module] = []
        prev = state_dim
        for hidden in hidden_sizes:
            layers.extend([nn.Linear(prev, hidden), nn.Tanh()])
            prev = hidden
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, obs, **kwargs):
        if isinstance(obs, np.ndarray):
            obs = torch.as_tensor(obs, dtype=torch.float32, device=next(self.parameters()).device)
        return self.net(obs)


def set_global_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def init_model_weights(modules: Sequence[nn.Module]) -> None:
    for module in modules:
        for layer in module.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight)
                nn.init.zeros_(layer.bias)


def build_networks(
    state_dim: int,
    action_dim: int,
    hidden_sizes: Sequence[int],
    device: str,
) -> tuple[nn.Module, list[nn.Module]]:
    actor = DiscreteActor(state_dim, action_dim, hidden_sizes).to(device)
    critics = [CriticNet(state_dim, hidden_sizes).to(device) for _ in range(2)]
    init_model_weights([actor, *critics])
    return actor, critics


def build_policy(
    config: ExperimentConfig,
    actor: nn.Module,
    critics: list[nn.Module],
    sample_env: gym.Env,
):
    optim = torch.optim.Adam(
        [
            {"params": actor.parameters(), "lr": config.pi_lr},
            {"params": critics[0].parameters(), "lr": config.vf_lr},
            {"params": critics[1].parameters(), "lr": config.vf_lr},
        ]
    )

    def dist_fn(*logits):
        return Categorical(logits=logits[0])

    common_kwargs = {
        "logger": BaseLogger(),
        "gae_lambda": 0.97,
        "gamma": 0.99,
        "advantage_normalization": True,
        "cost_limit": config.cost_limit,
        "reward_normalization": False,
        "deterministic_eval": True,
        "action_scaling": False,
        "action_bound_method": "",
        "observation_space": sample_env.observation_space,
        "action_space": sample_env.action_space,
        "max_batchsize": config.batch_size,
    }

    if config.algorithm in {"ppo", "ppolag"}:
        return PPOLagrangian(
            actor=actor,
            critics=critics,
            optim=optim,
            dist_fn=dist_fn,
            target_kl=0.01,
            vf_coef=0.25,
            max_grad_norm=0.5,
            eps_clip=0.2,
            use_lagrangian=config.algorithm == "ppolag",
            lagrangian_pid=(10.0, 1.0, 1.0),
            rescaling=True,
            **common_kwargs,
        )

    if config.algorithm == "cpo":
        return CPO(
            actor=actor,
            critics=critics,
            optim=optim,
            dist_fn=dist_fn,
            target_kl=0.01,
            backtrack_coeff=0.8,
            damping_coeff=0.1,
            max_backtracks=15,
            optim_critic_iters=config.cpo_critic_iters,
            l2_reg=1e-3,
            **common_kwargs,
        )

    raise ValueError(f"Unsupported algorithm: {config.algorithm}")


def train_policy(config: ExperimentConfig):
    set_global_seed(config.seed)
    device = config.resolved_device

    print(f"\n{'=' * 60}")
    print(f"Training {config.display_name} on {config.env_id}")
    print(f"  Total steps: {config.epochs * config.steps_per_epoch}")
    print(f"  Network: MLP {config.hidden_sizes}, tanh")
    print(f"  Policy LR: {config.pi_lr}, Value LR: {config.vf_lr}")
    print(f"  Updates per collection: {config.trainer_repeat_per_collect}")
    print(f"  Cost limit: {config.cost_limit}")
    print(f"  Device: {device}")
    print(f"{'=' * 60}\n")

    train_envs = DummyVectorEnv([lambda: make_env(config.env_id) for _ in range(config.train_envs)])
    test_envs = DummyVectorEnv([lambda: make_env(config.env_id) for _ in range(config.test_envs)])

    sample_env = make_env(config.env_id)
    state_dim = sample_env.observation_space.shape[0]
    action_dim = sample_env.action_space.n
    print(f"State dim: {state_dim}, Action dim: {action_dim}")

    actor, critics = build_networks(state_dim, action_dim, tuple(config.hidden_sizes), device)
    policy = build_policy(config, actor, critics, sample_env)
    sample_env.close()

    buf = VectorReplayBuffer(config.buffer_size, config.train_envs)
    train_collector = FastCollector(policy, train_envs, buf, exploration_noise=True)
    test_collector = FastCollector(policy, test_envs)

    trainer = OnpolicyTrainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=config.epochs,
        batch_size=config.batch_size,
        cost_limit=config.cost_limit,
        step_per_epoch=config.steps_per_epoch,
        repeat_per_collect=config.trainer_repeat_per_collect,
        episode_per_test=config.test_episodes,
        episode_per_collect=config.episode_per_collect,
        stop_fn=lambda r, c: False,
        logger=BaseLogger(),
        verbose=True,
        show_progress=True,
    )

    history = {
        "epoch": [],
        "train_reward": [],
        "train_cost": [],
        "test_reward": [],
        "test_cost": [],
        "test_length": [],
    }

    policy.train()
    for epoch, epoch_stat, info in trainer:
        history["epoch"].append(epoch)
        history["train_reward"].append(epoch_stat.get("train/reward", 0.0))
        history["train_cost"].append(epoch_stat.get("train/cost", 0.0))
        history["test_reward"].append(epoch_stat.get("test/reward", 0.0))
        history["test_cost"].append(epoch_stat.get("test/cost", 0.0))
        history["test_length"].append(epoch_stat.get("test/length", 0.0))

        if epoch % 5 == 0 or epoch == config.epochs:
            print(
                f"\n>>> Epoch {epoch}: "
                f"test_rew={epoch_stat.get('test/reward', 0.0):.2f}, "
                f"test_cost={epoch_stat.get('test/cost', 0.0):.3f}, "
                f"train_cost={epoch_stat.get('train/cost', 0.0):.3f}"
            )

    train_envs.close()
    test_envs.close()
    return policy, history


def evaluate_policy(policy, env_id: str, n_episodes: int = 100, seed: int = 0) -> dict[str, Any]:
    rewards: list[float] = []
    costs: list[float] = []
    lengths: list[int] = []
    crashed_episodes: list[int] = []

    policy.eval()
    for episode_idx in range(n_episodes):
        env = make_env(env_id)
        obs, _ = env.reset(seed=seed + episode_idx)
        total_reward = 0.0
        total_cost = 0.0
        steps = 0

        while True:
            batch = Batch(obs=obs.reshape(1, -1), info={})
            with torch.no_grad():
                result = policy(batch)
            action = int(result.act[0])
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            total_cost += float(info.get("cost", 0.0))
            steps += 1
            if terminated or truncated:
                break

        rewards.append(total_reward)
        costs.append(total_cost)
        lengths.append(steps)
        crashed_episodes.append(1 if total_cost > 0 else 0)
        env.close()

    collision_rate = float(np.mean(crashed_episodes) * 100)
    avg_reward = float(np.mean(rewards))
    avg_cost = float(np.mean(costs))
    avg_length = float(np.mean(lengths))

    print(f"\n{'=' * 50}")
    print(f"Evaluation ({n_episodes} episodes):")
    print(f"  Avg Reward:     {avg_reward:.3f} +/- {np.std(rewards):.3f}")
    print(f"  Avg Cost:       {avg_cost:.3f}")
    print(f"  Collision Rate: {collision_rate:.1f}%")
    print(f"  Avg Length:     {avg_length:.1f}")
    print(f"{'=' * 50}")

    return {
        "rewards": rewards,
        "costs": costs,
        "lengths": lengths,
        "collision_rate": collision_rate,
        "avg_reward": avg_reward,
        "avg_cost": avg_cost,
        "avg_length": avg_length,
    }


def plot_training(history: dict[str, list[float]], title: str, save_path: os.PathLike[str] | str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(history["epoch"], history["test_reward"], "b-", alpha=0.7, label="test")
    axes[0].plot(history["epoch"], history["train_reward"], "b--", alpha=0.4, label="train")
    axes[0].set_title("Episode Reward")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Reward")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history["epoch"], history["test_cost"], "r-", alpha=0.7, label="test")
    axes[1].plot(history["epoch"], history["train_cost"], "r--", alpha=0.4, label="train")
    axes[1].set_title("Episode Cost")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Cost")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(history["epoch"], history["test_length"], "g-", alpha=0.7)
    axes[2].set_title("Episode Length")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Steps")
    axes[2].grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def save_json(path: os.PathLike[str] | str, data: dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def result_paths(config: ExperimentConfig) -> dict[str, Path]:
    output_dir = Path(config.output_dir)
    return {
        "output_dir": output_dir,
        "training_plot": output_dir / f"{config.tag}_training.png",
        "results_json": output_dir / f"{config.tag}_results.json",
        "policy": output_dir / f"{config.tag}_policy.pth",
    }


def build_run_summary(
    config: ExperimentConfig,
    history: dict[str, list[float]],
    eval_results: dict[str, Any],
) -> dict[str, Any]:
    final_train_reward = history["train_reward"][-1] if history["train_reward"] else math.nan
    final_train_cost = history["train_cost"][-1] if history["train_cost"] else math.nan
    final_test_reward = history["test_reward"][-1] if history["test_reward"] else math.nan
    final_test_cost = history["test_cost"][-1] if history["test_cost"] else math.nan
    final_test_length = history["test_length"][-1] if history["test_length"] else math.nan

    return {
        "algorithm": config.algorithm,
        "algorithm_label": config.display_name,
        "env_id": config.env_id,
        "scenario_label": scenario_label(config.env_id),
        "seed": config.seed,
        "avg_reward": eval_results["avg_reward"],
        "avg_cost": eval_results["avg_cost"],
        "avg_length": eval_results["avg_length"],
        "collision_rate": eval_results["collision_rate"],
        "final_train_reward": final_train_reward,
        "final_train_cost": final_train_cost,
        "final_test_reward": final_test_reward,
        "final_test_cost": final_test_cost,
        "final_test_length": final_test_length,
    }


def load_run_from_json(path: os.PathLike[str] | str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def run_experiment(config: ExperimentConfig) -> dict[str, Any]:
    paths = result_paths(config)
    paths["output_dir"].mkdir(parents=True, exist_ok=True)

    policy, history = train_policy(config)
    plot_training(
        history,
        f"{config.display_name} on {config.env_id}",
        paths["training_plot"],
    )
    eval_results = evaluate_policy(policy, config.env_id, config.eval_episodes, seed=1000 + config.seed)
    torch.save(policy.state_dict(), paths["policy"])
    print(f"Policy saved to {paths['policy']}")

    summary = build_run_summary(config, history, eval_results)
    save_payload = {
        "config": config.to_dict(),
        "history": history,
        "eval": {
            key: value
            for key, value in eval_results.items()
            if key not in {"rewards", "costs", "lengths"}
        },
        "summary": summary,
    }
    save_json(paths["results_json"], save_payload)
    print(f"Results saved to {paths['results_json']}")
    summary["output_dir"] = str(paths["output_dir"])
    return save_payload
