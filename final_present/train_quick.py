"""
Minimal discrete PPO trainer (no fsrl/tianshou).  Used only to get a
non-random policy on highway-v0, roundabout-v0, intersection-v1 for the
demo video.  ~3000 steps per env on CPU ≈ 2-3 min.
"""
import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import highway_env  # noqa: F401


def make_env(env_id, seed):
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
    env = gym.make(env_id, config=cfg)
    env = gym.wrappers.FlattenObservation(env)
    env.reset(seed=seed)
    return env


class Policy(nn.Module):
    def __init__(self, obs_dim, n_act, hidden=64):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, n_act),
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, obs):
        return self.actor(obs), self.critic(obs).squeeze(-1)

    def act(self, obs, deterministic=False):
        with torch.no_grad():
            x = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            logits, v = self.forward(x)
            if deterministic:
                a = int(torch.argmax(logits, dim=-1).item())
                logp = float(F.log_softmax(logits, dim=-1)[0, a].item())
            else:
                dist = torch.distributions.Categorical(logits=logits)
                a = int(dist.sample().item())
                logp = float(dist.log_prob(torch.tensor([a])).item())
        return a, logp, float(v.item())


def gae(rewards, values, dones, gamma=0.99, lam=0.95):
    adv = np.zeros_like(rewards, dtype=np.float32)
    gae_acc = 0.0
    next_v = 0.0
    for t in reversed(range(len(rewards))):
        nonterm = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_v * nonterm - values[t]
        gae_acc = delta + gamma * lam * nonterm * gae_acc
        adv[t] = gae_acc
        next_v = values[t]
    returns = adv + np.array(values, dtype=np.float32)
    return adv, returns


def train(env_id, total_steps=3000, batch=512, lr=3e-4, seed=42):
    env = make_env(env_id, seed)
    obs_dim = env.observation_space.shape[0]
    n_act = env.action_space.n
    pol = Policy(obs_dim, n_act)
    opt = torch.optim.Adam(pol.parameters(), lr=lr)
    print(f"[{env_id}] obs={obs_dim} act={n_act} target={total_steps} steps")

    obs, _ = env.reset(seed=seed)
    step = 0
    last_log = time.time()
    ep_returns = []
    cur_ret = 0.0
    while step < total_steps:
        # collect batch
        buf_obs, buf_act, buf_logp, buf_val = [], [], [], []
        buf_rew, buf_done = [], []
        for _ in range(batch):
            a, logp, v = pol.act(obs)
            o2, r, term, trunc, _ = env.step(a)
            buf_obs.append(obs)
            buf_act.append(a); buf_logp.append(logp); buf_val.append(v)
            buf_rew.append(float(r)); buf_done.append(float(term or trunc))
            cur_ret += float(r)
            if term or trunc:
                ep_returns.append(cur_ret); cur_ret = 0.0
                obs, _ = env.reset()
            else:
                obs = o2
            step += 1
            if step >= total_steps:
                break

        # PPO update
        obs_t = torch.as_tensor(np.array(buf_obs), dtype=torch.float32)
        act_t = torch.as_tensor(buf_act, dtype=torch.long)
        old_logp = torch.as_tensor(buf_logp, dtype=torch.float32)
        rews = np.array(buf_rew, dtype=np.float32)
        dones = np.array(buf_done, dtype=np.float32)
        vals = np.array(buf_val, dtype=np.float32)
        adv, ret = gae(rews, vals, dones)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        adv_t = torch.as_tensor(adv, dtype=torch.float32)
        ret_t = torch.as_tensor(ret, dtype=torch.float32)

        for _ in range(4):
            logits, v = pol.forward(obs_t)
            dist = torch.distributions.Categorical(logits=logits)
            new_logp = dist.log_prob(act_t)
            ratio = (new_logp - old_logp).exp()
            clip = ratio.clamp(0.8, 1.2)
            policy_loss = -torch.min(ratio * adv_t, clip * adv_t).mean()
            value_loss = F.mse_loss(v, ret_t)
            entropy = dist.entropy().mean()
            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(pol.parameters(), 0.5)
            opt.step()

        if time.time() - last_log > 5:
            recent = ep_returns[-10:] if ep_returns else [0.0]
            print(f"  step {step:5d} ep_ret(last10)={np.mean(recent):+.2f}")
            last_log = time.time()

    env.close()
    return pol


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    pol = train(args.env, total_steps=args.steps, seed=args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(pol.state_dict(), args.out)
    print(f"saved {args.out}")
