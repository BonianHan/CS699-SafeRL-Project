"""Single-run training entrypoint for the Safe RL highway-env experiments."""

from __future__ import annotations

import argparse

from saferl_suite import DEFAULT_ALGORITHMS, ExperimentConfig, run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Safe RL algorithms on highway-env discrete control tasks.",
    )
    parser.add_argument("--algorithm", choices=DEFAULT_ALGORITHMS, default="ppolag")
    parser.add_argument("--env_id", type=str, default="merge-v0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--steps_per_epoch", type=int, default=4000)
    parser.add_argument("--hidden_sizes", type=int, nargs="+", default=[64, 64])
    parser.add_argument("--pi_lr", type=float, default=3e-4)
    parser.add_argument("--vf_lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=99999)
    parser.add_argument("--repeat_per_collect", type=int, default=80)
    parser.add_argument("--episode_per_collect", type=int, default=100)
    parser.add_argument("--buffer_size", type=int, default=100000)
    parser.add_argument("--cost_limit", type=float, default=0.1)
    parser.add_argument("--train_envs", type=int, default=8)
    parser.add_argument("--test_envs", type=int, default=4)
    parser.add_argument("--test_episodes", type=int, default=10)
    parser.add_argument("--eval_episodes", type=int, default=100)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--no_lagrangian",
        action="store_true",
        help="Backwards-compatible alias for --algorithm ppo.",
    )
    parser.add_argument(
        "--cpo_critic_iters",
        type=int,
        default=20,
        help="Number of critic updates inside each CPO policy update.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    algorithm = "ppo" if args.no_lagrangian else args.algorithm
    config = ExperimentConfig(
        algorithm=algorithm,
        env_id=args.env_id,
        seed=args.seed,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        hidden_sizes=args.hidden_sizes,
        pi_lr=args.pi_lr,
        vf_lr=args.vf_lr,
        batch_size=args.batch_size,
        repeat_per_collect=args.repeat_per_collect,
        episode_per_collect=args.episode_per_collect,
        buffer_size=args.buffer_size,
        cost_limit=args.cost_limit,
        train_envs=args.train_envs,
        test_envs=args.test_envs,
        test_episodes=args.test_episodes,
        eval_episodes=args.eval_episodes,
        device=args.device,
        output_dir=args.output_dir,
        quick=args.quick,
        cpo_critic_iters=args.cpo_critic_iters,
    )
    run_experiment(config)


if __name__ == "__main__":
    main()
