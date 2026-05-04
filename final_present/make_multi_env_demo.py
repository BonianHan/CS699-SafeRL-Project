"""
Record a 2x2 grid demo showing one trained policy per scenario:
  merge       : the full PPOLag from results_final/
  highway     : quick-PPO trained on highway-v0
  roundabout  : quick-PPO trained on roundabout-v0
  intersection: quick-PPO trained on intersection-v1
"""
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import highway_env  # noqa: F401
import imageio
from PIL import Image, ImageDraw, ImageFont
import collections


PROJECT = Path("/Users/bonianhan/Projects/CS699/project")
OUT = PROJECT / "final_present" / "all_envs_demo.mp4"


# ------------------ models ------------------
class MergeActor(nn.Module):
    """Loaded from the full PPOLagrangian state_dict (results_final/)."""
    def __init__(self, sd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(25, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 5),
        )
        with torch.no_grad():
            self.net[0].weight.copy_(sd["actor.net.0.weight"])
            self.net[0].bias.copy_(sd["actor.net.0.bias"])
            self.net[2].weight.copy_(sd["actor.net.2.weight"])
            self.net[2].bias.copy_(sd["actor.net.2.bias"])
            self.net[4].weight.copy_(sd["actor.net.4.weight"])
            self.net[4].bias.copy_(sd["actor.net.4.bias"])
        self.eval()

    @torch.no_grad()
    def act(self, obs):
        x = torch.as_tensor(obs.flatten(), dtype=torch.float32).unsqueeze(0)
        return int(torch.argmax(self.net(x), dim=-1).item())


class QuickActor(nn.Module):
    """Loaded from the train_quick.py state_dict (actor+critic in one Policy)."""
    def __init__(self, sd, obs_dim, n_act=5):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, n_act),
        )
        with torch.no_grad():
            self.actor[0].weight.copy_(sd["actor.0.weight"])
            self.actor[0].bias.copy_(sd["actor.0.bias"])
            self.actor[2].weight.copy_(sd["actor.2.weight"])
            self.actor[2].bias.copy_(sd["actor.2.bias"])
            self.actor[4].weight.copy_(sd["actor.4.weight"])
            self.actor[4].bias.copy_(sd["actor.4.bias"])
        self.eval()

    @torch.no_grad()
    def act(self, obs):
        x = torch.as_tensor(obs.flatten(), dtype=torch.float32).unsqueeze(0)
        return int(torch.argmax(self.actor(x), dim=-1).item())


# ------------------ env factory ------------------
def make_env(env_id):
    cfg = {
        "action": {"type": "DiscreteMetaAction"},
        "offscreen_rendering": True,
        "screen_width": 720,
        "screen_height": 200,
        "scaling": 5.5,
    }
    if env_id == "merge-v0":
        cfg["collision_reward"] = 0
        cfg["vehicles_count"] = 20
        cfg["vehicles_density"] = 1.5
        cfg["high_speed_reward"] = 0.4
    env = gym.make(env_id, config=cfg, render_mode="rgb_array")
    env = gym.wrappers.FlattenObservation(env)
    return env


# ------------------ recorder ------------------
def record_env(env_id, actor, max_frames=240, seed_start=200):
    env = make_env(env_id)
    frames, actions = [], []
    crashes, episodes = 0, 0
    seed = seed_start
    while len(frames) < max_frames:
        obs, _ = env.reset(seed=seed)
        seed += 1
        while len(frames) < max_frames:
            frames.append(env.render())
            a = actor.act(obs)
            actions.append(a)
            obs, _, term, trunc, info = env.step(a)
            if info.get("crashed", False):
                crashes += 1
            if term or trunc:
                episodes += 1
                break
    env.close()
    return frames, actions, crashes, episodes


# ------------------ compose ------------------
NAMES = {0: "LANE_LEFT", 1: "IDLE", 2: "LANE_RIGHT", 3: "FASTER", 4: "SLOWER"}

def label_panel(frame, title, info, color):
    h, w = frame.shape[:2]
    BANNER = 48
    INFO = 36
    canvas = np.ones((h + BANNER + INFO, w, 3), dtype=np.uint8) * 255
    canvas[BANNER:BANNER + h] = frame
    img = Image.fromarray(canvas)
    draw = ImageDraw.Draw(img)
    draw.rectangle([(0, 0), (w, BANNER)], fill=color)
    try:
        f1 = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 22)
        f2 = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
    except Exception:
        f1 = ImageFont.load_default(); f2 = f1
    draw.text((12, 12), title, fill="white", font=f1)
    draw.rectangle([(0, BANNER + h), (w, BANNER + h + INFO)],
                   fill=(245, 245, 245))
    draw.text((12, BANNER + h + 8), info, fill=(40, 40, 40), font=f2)
    return np.array(img)


def grid_2x2(top_left, top_right, bottom_left, bottom_right, gap=8):
    h = max(top_left.shape[0], top_right.shape[0])
    w = max(top_left.shape[1], top_right.shape[1])
    H = h * 2 + gap
    W = w * 2 + gap
    canvas = np.ones((H, W, 3), dtype=np.uint8) * 255
    canvas[:h, :w] = top_left
    canvas[:h, w + gap:w + gap + top_right.shape[1]] = top_right
    canvas[h + gap:h + gap + bottom_left.shape[0], :w] = bottom_left
    canvas[h + gap:h + gap + bottom_right.shape[0],
           w + gap:w + gap + bottom_right.shape[1]] = bottom_right
    return canvas


def main():
    print("Loading policies...")
    merge_sd = torch.load(PROJECT / "results_final" / "ppolag_policy.pth",
                          map_location="cpu", weights_only=False)
    merge_actor = MergeActor(merge_sd)

    actors = {"merge-v0": merge_actor}
    obs_dims = {"highway-v0": 25, "roundabout-v0": 25, "intersection-v1": 40}
    for eid, dim in obs_dims.items():
        path = Path("/tmp/policies") / f"{eid.split('-')[0]}.pth"
        sd = torch.load(path, map_location="cpu", weights_only=True)
        actors[eid] = QuickActor(sd, dim)
        print(f"  loaded {path.name}  obs_dim={dim}")

    # record each env
    layout = [
        ("merge-v0",         "Merge",        "Full PPOLag (80k steps)",        (40, 160, 70)),
        ("highway-v0",       "Highway",      "Quick PPO (3k steps)",            (37, 99, 235)),
        ("roundabout-v0",    "Roundabout",   "Quick PPO (3k steps)",            (200, 90, 60)),
        ("intersection-v1",  "Intersection", "Quick PPO (3k steps)",            (220, 50, 50)),
    ]

    panels = {}
    for env_id, _, _, _ in layout:
        print(f"Recording {env_id}...")
        frames, acts, crashes, eps = record_env(env_id, actors[env_id],
                                                max_frames=200,
                                                seed_start=300)
        c = collections.Counter(acts)
        total = sum(c.values()) or 1
        # top action
        top = sorted(c.items(), key=lambda kv: -kv[1])[:3]
        top_str = "  ".join(f"{NAMES[k][:4]} {v/total*100:.0f}%" for k, v in top)
        info = (f"{crashes}/{eps} crash ({crashes/max(1,eps)*100:.0f}%)  "
                f"·  actions: {top_str}")
        panels[env_id] = (frames, info)

    n = min(len(p[0]) for p in panels.values())
    print(f"composing {n} frames")

    composed = []
    for i in range(n):
        labeled = {}
        for env_id, label, sub, color in layout:
            frames, info = panels[env_id]
            f = frames[i]
            title = f"{label}  —  {sub}"
            labeled[env_id] = label_panel(f, title, info, color)
        grid = grid_2x2(labeled["merge-v0"], labeled["highway-v0"],
                        labeled["roundabout-v0"], labeled["intersection-v1"])
        composed.append(grid)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(OUT), fps=12, codec="libx264",
                                output_params=["-pix_fmt", "yuv420p"])
    for f in composed:
        writer.append_data(f)
    writer.close()
    print(f"wrote {OUT}  ({len(composed)} frames @ 12fps = {len(composed)/12:.1f}s)")

    # also dump per-env action distribution to a json so we can drop it in slides
    import json
    dist = {}
    for env_id, label, sub, _ in layout:
        frames, info = panels[env_id]
        dist[env_id] = info
    with open(PROJECT / "final_present" / "all_envs_demo_meta.json", "w") as f:
        json.dump(dist, f, indent=2)
    print("wrote meta json")


if __name__ == "__main__":
    main()
