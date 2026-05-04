"""
Sequential (not 2x2) demo: each env plays for ~5 seconds in turn.
Total ~20 seconds at 12 fps.
"""
from pathlib import Path
import collections
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import highway_env  # noqa: F401
import imageio
from PIL import Image, ImageDraw, ImageFont

PROJECT = Path("/Users/bonianhan/Projects/CS699/project")
OUT = PROJECT / "final_present" / "all_envs_demo.mp4"
PER_ENV_FRAMES = 60   # 5 seconds at 12 fps
FPS = 12


# ---------- networks ----------
class MergeActor(nn.Module):
    def __init__(self, sd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(25, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 5),
        )
        with torch.no_grad():
            for i, k in zip([0, 2, 4], ["0", "2", "4"]):
                self.net[i].weight.copy_(sd[f"actor.net.{k}.weight"])
                self.net[i].bias.copy_(sd[f"actor.net.{k}.bias"])
        self.eval()

    @torch.no_grad()
    def act(self, obs):
        x = torch.as_tensor(obs.flatten(), dtype=torch.float32).unsqueeze(0)
        return int(torch.argmax(self.net(x), dim=-1).item())


class QuickActor(nn.Module):
    def __init__(self, sd, obs_dim, n_act=5):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, n_act),
        )
        with torch.no_grad():
            for i, k in zip([0, 2, 4], ["0", "2", "4"]):
                self.actor[i].weight.copy_(sd[f"actor.{k}.weight"])
                self.actor[i].bias.copy_(sd[f"actor.{k}.bias"])
        self.eval()

    @torch.no_grad()
    def act(self, obs):
        x = torch.as_tensor(obs.flatten(), dtype=torch.float32).unsqueeze(0)
        return int(torch.argmax(self.actor(x), dim=-1).item())


# ---------- env ----------
def make_env(env_id):
    cfg = {
        "action": {"type": "DiscreteMetaAction"},
        "offscreen_rendering": True,
        "screen_width": 960,
        "screen_height": 280,
        "scaling": 6.5,
    }
    if env_id == "merge-v0":
        cfg["collision_reward"] = 0
        cfg["vehicles_count"] = 20
        cfg["vehicles_density"] = 1.5
        cfg["high_speed_reward"] = 0.4
    env = gym.make(env_id, config=cfg, render_mode="rgb_array")
    env = gym.wrappers.FlattenObservation(env)
    return env


def record_env(env_id, actor, n_frames, seed_start=300):
    env = make_env(env_id)
    frames, actions = [], []
    crashes, episodes = 0, 0
    seed = seed_start
    while len(frames) < n_frames:
        obs, _ = env.reset(seed=seed); seed += 1
        while len(frames) < n_frames:
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
    return frames[:n_frames], actions[:n_frames], crashes, episodes


# ---------- composition ----------
NAMES = {0: "LANE_LEFT", 1: "IDLE", 2: "LANE_RIGHT", 3: "FASTER", 4: "SLOWER"}

def font(size):
    try:
        return ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size)
    except Exception:
        return ImageFont.load_default()


def label_frame(frame, label, sub, info, color, total_w=960):
    """Wrap one env frame with a coloured banner top + info row bottom."""
    h, w = frame.shape[:2]
    BANNER = 70
    INFO = 56
    # pad to total_w if narrower
    if w < total_w:
        padded = np.ones((h, total_w, 3), dtype=np.uint8) * 60
        offset = (total_w - w) // 2
        padded[:, offset:offset + w] = frame
        frame = padded
        w = total_w
    canvas = np.ones((h + BANNER + INFO, w, 3), dtype=np.uint8) * 255
    canvas[BANNER:BANNER + h] = frame
    img = Image.fromarray(canvas)
    draw = ImageDraw.Draw(img)
    draw.rectangle([(0, 0), (w, BANNER)], fill=color)
    f1 = font(28); f2 = font(15); f3 = font(14)
    draw.text((20, 12), label, fill="white", font=f1)
    draw.text((20, 44), sub, fill=(220, 220, 220), font=f2)
    draw.rectangle([(0, BANNER + h), (w, BANNER + h + INFO)],
                   fill=(245, 245, 245))
    draw.text((20, BANNER + h + 16), info, fill=(40, 40, 40), font=f3)
    return np.array(img)


def main():
    print("Loading policies...")
    merge_sd = torch.load(PROJECT / "results_final" / "ppolag_policy.pth",
                          map_location="cpu", weights_only=False)
    actors = {"merge-v0": MergeActor(merge_sd)}
    obs_dims = {"highway-v0": 25, "roundabout-v0": 25, "intersection-v1": 40}
    for eid, dim in obs_dims.items():
        path = Path("/tmp/policies") / f"{eid.split('-')[0]}.pth"
        sd = torch.load(path, map_location="cpu", weights_only=True)
        actors[eid] = QuickActor(sd, dim)
        print(f"  loaded {path.name}")

    layout = [
        ("merge-v0",        "Merge",        "Full PPOLag  (80k steps)",     (40, 160, 70)),
        ("highway-v0",      "Highway",      "Quick PPO  (3k steps)",         (37, 99, 235)),
        ("roundabout-v0",   "Roundabout",   "Quick PPO  (3k steps)",         (200, 90, 60)),
        ("intersection-v1", "Intersection", "Quick PPO  (3k steps)",         (220, 50, 50)),
    ]

    sequential_frames = []
    meta = {}
    for env_id, label, sub, color in layout:
        print(f"Recording {env_id}...")
        frames, acts, crashes, eps = record_env(env_id, actors[env_id],
                                                n_frames=PER_ENV_FRAMES)
        c = collections.Counter(acts)
        total = sum(c.values()) or 1
        top = sorted(c.items(), key=lambda kv: -kv[1])[:3]
        top_str = "  ".join(f"{NAMES[k][:5]} {v/total*100:.0f}%" for k, v in top)
        info = (f"{crashes}/{eps} crashes  ({crashes/max(1,eps)*100:.0f}%)"
                f"     ·     actions: {top_str}")
        meta[env_id] = info
        for f in frames:
            sequential_frames.append(label_frame(f, label, sub, info, color))

    print(f"composing {len(sequential_frames)} frames")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(OUT), fps=FPS, codec="libx264",
                                output_params=["-pix_fmt", "yuv420p"])
    for f in sequential_frames:
        writer.append_data(f)
    writer.close()
    print(f"wrote {OUT}  ({len(sequential_frames)} frames @ {FPS}fps "
          f"= {len(sequential_frames)/FPS:.1f}s)")

    import json
    with open(PROJECT / "final_present" / "all_envs_demo_meta.json", "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
