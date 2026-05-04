"""Generate video of highway-env environments running with an agent."""
import gymnasium as gym
import highway_env
import numpy as np
import os

try:
    import imageio
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "imageio[ffmpeg]"])
    import imageio

out_dir = "videos"
os.makedirs(out_dir, exist_ok=True)

envs_to_record = {
    "merge-v0": {"steps": 80, "label": "Merge"},
    "highway-v0": {"steps": 80, "label": "Highway"},
    "roundabout-v0": {"steps": 80, "label": "Roundabout"},
    "intersection-v1": {"steps": 80, "label": "Intersection"},
}

all_frames = []

for env_id, cfg in envs_to_record.items():
    print(f"Recording {env_id}...")
    env = gym.make(env_id, render_mode="rgb_array")
    obs, info = env.reset(seed=42)

    frames = []
    for step in range(cfg["steps"]):
        frame = env.render()
        frames.append(frame)
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset(seed=42 + step)

    env.close()

    # Save individual env video
    video_path = os.path.join(out_dir, f"{env_id.replace('-', '_')}.mp4")
    writer = imageio.get_writer(video_path, fps=15, codec="libx264",
                                 output_params=["-pix_fmt", "yuv420p"])
    for f in frames:
        writer.append_data(f)
    writer.close()
    print(f"  Saved: {video_path}")

    # Collect frames for combined video (take first 40 frames each)
    all_frames.append((cfg["label"], frames[:40]))

# Create combined video: show each env sequentially with label overlay
print("Creating combined video...")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

combined_frames = []
for label, frames in all_frames:
    for i, frame in enumerate(frames):
        fig = Figure(figsize=(10, 5.625), dpi=100)
        canvas = FigureCanvasAgg(fig)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(frame)
        ax.axis("off")
        # Add label overlay
        ax.text(0.02, 0.95, label, transform=ax.transAxes,
                fontsize=24, fontweight="bold", color="white",
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.7))
        canvas.draw()
        w, h = canvas.get_width_height()
        buf = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
        combined_frames.append(buf[:, :, :3])  # drop alpha
        plt.close(fig)

combined_path = os.path.join(out_dir, "all_envs_demo.mp4")
writer = imageio.get_writer(combined_path, fps=15, codec="libx264",
                             output_params=["-pix_fmt", "yuv420p"])
for f in combined_frames:
    writer.append_data(f)
writer.close()
print(f"Saved combined: {combined_path}")
print("Done!")
