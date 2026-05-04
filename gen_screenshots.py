"""Generate environment rendering screenshots for midterm demo PPT."""
import gymnasium as gym
import highway_env
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

out_dir = "screenshots"
os.makedirs(out_dir, exist_ok=True)

envs = {
    "merge-v0": "Merge",
    "highway-v0": "Highway",
    "roundabout-v0": "Roundabout",
    "intersection-v1": "Intersection",
    "racetrack-v0": "Racetrack",
}

for env_id, label in envs.items():
    print(f"Rendering {env_id}...")
    env = gym.make(env_id, render_mode="rgb_array")
    obs, info = env.reset(seed=42)
    # Step a few times to get interesting state
    for _ in range(5):
        obs, _, terminated, truncated, _ = env.step(env.action_space.sample())
        if terminated or truncated:
            obs, info = env.reset(seed=42)
    frame = env.render()
    env.close()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.imshow(frame)
    ax.set_title(f"{label} ({env_id})", fontsize=14, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()
    path = os.path.join(out_dir, f"{env_id.replace('-', '_')}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

# Also create a combined grid
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()
for i, (env_id, label) in enumerate(envs.items()):
    env = gym.make(env_id, render_mode="rgb_array")
    obs, _ = env.reset(seed=42)
    for _ in range(5):
        obs, _, terminated, truncated, _ = env.step(env.action_space.sample())
        if terminated or truncated:
            obs, _ = env.reset(seed=42)
    frame = env.render()
    env.close()
    axes[i].imshow(frame)
    axes[i].set_title(label, fontsize=12, fontweight="bold")
    axes[i].axis("off")

axes[-1].axis("off")  # empty 6th cell
plt.suptitle("Highway-Env Scenarios", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "all_envs_grid.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved: screenshots/all_envs_grid.png")
