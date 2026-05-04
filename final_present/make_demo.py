"""
Generate ~30 second side-by-side PPO vs PPOLag driving demo on merge-v0.

Loads only actor weights (no fsrl/tianshou dependency).
Renders both policies in parallel, stitches frames side-by-side, writes mp4.
"""
import os
# NOTE: do NOT set SDL_VIDEODRIVER=dummy on macOS — it makes highway-env
# render solid-black frames.

import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import highway_env  # noqa: F401  registers envs
import imageio
from PIL import Image, ImageDraw, ImageFont

# ---------- network ----------
class Actor(nn.Module):
    def __init__(self, sd, prefix="actor.net"):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(25, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 5),
        )
        # map state_dict keys
        with torch.no_grad():
            self.net[0].weight.copy_(sd[f"{prefix}.0.weight"])
            self.net[0].bias.copy_(sd[f"{prefix}.0.bias"])
            self.net[2].weight.copy_(sd[f"{prefix}.2.weight"])
            self.net[2].bias.copy_(sd[f"{prefix}.2.bias"])
            self.net[4].weight.copy_(sd[f"{prefix}.4.weight"])
            self.net[4].bias.copy_(sd[f"{prefix}.4.bias"])
        self.eval()

    @torch.no_grad()
    def act(self, obs):
        x = torch.as_tensor(obs.flatten(), dtype=torch.float32).unsqueeze(0)
        logits = self.net(x)
        return int(torch.argmax(logits, dim=-1).item())


def load_actor(path):
    sd = torch.load(path, map_location="cpu", weights_only=False)
    return Actor(sd)


# ---------- env ----------
def make_env(seed):
    config = {
        "collision_reward": 0,
        "vehicles_count": 20,
        "vehicles_density": 1.5,
        "high_speed_reward": 0.4,
        "offscreen_rendering": True,    # avoid pygame.display issues on macOS
        "screen_width": 720,
        "screen_height": 200,
        "scaling": 5.5,
    }
    env = gym.make("merge-v0", config=config, render_mode="rgb_array")
    env = gym.wrappers.FlattenObservation(env)
    return env


def run_episode(actor, seed, max_steps=120):
    env = make_env(seed)
    obs, _ = env.reset(seed=seed)
    frames = [env.render()]
    crashed = False
    steps = 0
    while steps < max_steps:
        action = actor.act(obs)
        obs, _, terminated, truncated, info = env.step(action)
        frames.append(env.render())
        steps += 1
        if info.get("crashed", False):
            crashed = True
        if terminated or truncated:
            # auto reset to keep video going
            obs, _ = env.reset(seed=seed + 1000 + steps)
            crashed = False if not crashed else crashed  # keep flag if happened
    env.close()
    return frames


# ---------- compose ----------
def label_frame(frame, label, color, info_text=""):
    """Add a label banner above the frame."""
    h, w = frame.shape[:2]
    BANNER_H = 50
    INFO_H = 32
    canvas = np.ones((h + BANNER_H + INFO_H, w, 3), dtype=np.uint8) * 255
    canvas[BANNER_H:BANNER_H + h] = frame
    img = Image.fromarray(canvas)
    draw = ImageDraw.Draw(img)
    # label band
    draw.rectangle([(0, 0), (w, BANNER_H)], fill=color)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
        font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
    except Exception:
        font = ImageFont.load_default()
        font_small = font
    draw.text((12, 12), label, fill="white", font=font)
    # info band below image
    draw.rectangle([(0, BANNER_H + h), (w, BANNER_H + h + INFO_H)],
                   fill=(245, 245, 245))
    draw.text((12, BANNER_H + h + 6), info_text, fill=(40, 40, 40),
              font=font_small)
    return np.array(img)


def stitch(left, right, gap=12):
    h = max(left.shape[0], right.shape[0])
    w = left.shape[1] + right.shape[1] + gap
    canvas = np.ones((h, w, 3), dtype=np.uint8) * 255
    canvas[:left.shape[0], :left.shape[1]] = left
    canvas[:right.shape[0], left.shape[1] + gap:] = right
    return canvas


def main():
    base = "/Users/bonianhan/Projects/CS699/project"
    ppo = load_actor(f"{base}/results_final/ppo_policy.pth")
    ppolag = load_actor(f"{base}/results_final/ppolag_policy.pth")

    # Run both with same seed sequence so trajectories are comparable.
    # Need ~30s of footage. merge-v0 ep is short (~15 steps); chain a few seeds.
    ppo_frames = []
    ppolag_frames = []
    ppo_crash_count = 0
    ppolag_crash_count = 0
    ppo_episodes = 0
    ppolag_episodes = 0

    # Re-use a single env per actor (pygame on macOS dislikes repeated init).
    SEEDS = list(range(100, 200))
    env_ppo = make_env(0)
    env_ppolag = make_env(0)

    for seed in SEEDS:
        # PPO episode
        obs, _ = env_ppo.reset(seed=seed)
        while True:
            ppo_frames.append(env_ppo.render())
            action = ppo.act(obs)
            obs, _, terminated, truncated, info = env_ppo.step(action)
            if info.get("crashed", False):
                ppo_crash_count += 1
            if terminated or truncated:
                ppo_episodes += 1
                break

        # PPOLag episode
        obs, _ = env_ppolag.reset(seed=seed)
        while True:
            ppolag_frames.append(env_ppolag.render())
            action = ppolag.act(obs)
            obs, _, terminated, truncated, info = env_ppolag.step(action)
            if info.get("crashed", False):
                ppolag_crash_count += 1
            if terminated or truncated:
                ppolag_episodes += 1
                break

        # Stop once we have enough frames for ~30s at 12fps = ~360 frames
        if len(ppo_frames) >= 360 and len(ppolag_frames) >= 360:
            break

    env_ppo.close()
    env_ppolag.close()

    # Pad shorter list
    n = max(len(ppo_frames), len(ppolag_frames))
    while len(ppo_frames) < n:
        ppo_frames.append(ppo_frames[-1])
    while len(ppolag_frames) < n:
        ppolag_frames.append(ppolag_frames[-1])

    print(f"PPO   frames={len(ppo_frames)} crashes={ppo_crash_count}/{ppo_episodes}")
    print(f"PPOLag frames={len(ppolag_frames)} crashes={ppolag_crash_count}/{ppolag_episodes}")

    # also report action distributions so we can see whether the policies
    # actually use lane-change actions
    import collections
    def action_dist(actor, n_episodes=8):
        c = collections.Counter()
        env = make_env(0)
        for s in range(n_episodes):
            obs, _ = env.reset(seed=s + 500)
            while True:
                a = actor.act(obs)
                c[a] += 1
                obs, _, term, trunc, _ = env.step(a)
                if term or trunc:
                    break
        env.close()
        return c
    NAMES = {0: "LANE_LEFT", 1: "IDLE", 2: "LANE_RIGHT", 3: "FASTER", 4: "SLOWER"}
    for name, actor in [("PPO", ppo), ("PPOLag", ppolag)]:
        d = action_dist(actor)
        total = sum(d.values())
        print(f"{name} action distribution:")
        for k in sorted(d):
            print(f"  {NAMES[k]:11s} {d[k]/total*100:5.1f}%  ({d[k]}/{total})")

    # Stitch frames
    composed = []
    # rolling crash counters
    ppo_running_crashes = 0
    ppolag_running_crashes = 0
    ppo_ep_idx = 0
    ppolag_ep_idx = 0

    for i in range(n):
        info_l = f"PPO (unconstrained)  |  crashes: {ppo_crash_count}/{ppo_episodes} eps  ({ppo_crash_count/max(1,ppo_episodes)*100:.0f}%)"
        info_r = f"PPOLag (constrained) | crashes: {ppolag_crash_count}/{ppolag_episodes} eps  ({ppolag_crash_count/max(1,ppolag_episodes)*100:.0f}%)"
        L = label_frame(ppo_frames[i], "PPO  (unconstrained)",
                        (220, 50, 50), info_l)
        R = label_frame(ppolag_frames[i], "PPOLag  (constrained)",
                        (40, 160, 70), info_r)
        composed.append(stitch(L, R))

    out_path = f"{base}/figures/ppo_vs_ppolag_demo.mp4"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    writer = imageio.get_writer(out_path, fps=12, codec="libx264",
                                output_params=["-pix_fmt", "yuv420p"])
    for f in composed:
        writer.append_data(f)
    writer.close()
    print(f"Wrote {out_path}  ({len(composed)} frames @ 12fps = {len(composed)/12:.1f}s)")
    return ppo_crash_count, ppo_episodes, ppolag_crash_count, ppolag_episodes


if __name__ == "__main__":
    main()
