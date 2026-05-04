"""Run the multi-scenario, multi-seed Safe RL roadmap sweep and generate a report."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

from saferl_suite import (
    DEFAULT_ALGORITHMS,
    DEFAULT_SCENARIOS,
    ExperimentConfig,
    algorithm_label,
    load_run_from_json,
    normalize_algorithm,
    result_paths,
    run_experiment,
    scenario_label,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager


def configure_matplotlib_fonts() -> None:
    windows_font_dir = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
    for filename in ["arial.ttf", "segoeui.ttf", "tahoma.ttf"]:
        font_path = windows_font_dir / filename
        if font_path.exists():
            font_manager.fontManager.addfont(str(font_path))
            font_name = font_manager.FontProperties(fname=str(font_path)).get_name()
            matplotlib.rcParams["font.family"] = font_name
            matplotlib.rcParams["font.sans-serif"] = [font_name]
            return


configure_matplotlib_fonts()

SUMMARY_METRICS = [
    "avg_reward",
    "avg_cost",
    "avg_length",
    "collision_rate",
    "final_test_reward",
    "final_test_cost",
    "final_test_length",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Safe RL roadmap sweep across scenarios, algorithms, and seeds.",
    )
    parser.add_argument("--algorithms", nargs="+", default=list(DEFAULT_ALGORITHMS))
    parser.add_argument("--scenarios", nargs="+", default=list(DEFAULT_SCENARIOS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--steps_per_epoch", type=int, default=1000)
    parser.add_argument("--repeat_per_collect", type=int, default=5)
    parser.add_argument("--episode_per_collect", type=int, default=50)
    parser.add_argument("--train_envs", type=int, default=4)
    parser.add_argument("--test_envs", type=int, default=2)
    parser.add_argument("--test_episodes", type=int, default=5)
    parser.add_argument("--eval_episodes", type=int, default=25)
    parser.add_argument("--cost_limit", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output_dir", type=str, default="roadmap_results")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--cpo_critic_iters", type=int, default=20)
    parser.add_argument(
        "--baseline",
        type=str,
        default="ppo",
        help="Reference algorithm for significance tests. Use 'none' for pairwise tests.",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace, scenario: str, algorithm: str, seed: int) -> ExperimentConfig:
    run_dir = Path(args.output_dir) / "runs" / scenario / algorithm / f"seed_{seed}"
    return ExperimentConfig(
        algorithm=algorithm,
        env_id=scenario,
        seed=seed,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        repeat_per_collect=args.repeat_per_collect,
        episode_per_collect=args.episode_per_collect,
        cost_limit=args.cost_limit,
        train_envs=args.train_envs,
        test_envs=args.test_envs,
        test_episodes=args.test_episodes,
        eval_episodes=args.eval_episodes,
        device=args.device,
        output_dir=str(run_dir),
        cpo_critic_iters=args.cpo_critic_iters,
    )


def existing_or_run(config: ExperimentConfig, skip_existing: bool) -> dict[str, Any]:
    results_path = result_paths(config)["results_json"]
    if skip_existing and results_path.exists():
        print(f"Reusing: {results_path}")
        return load_run_from_json(results_path)
    return run_experiment(config)


def confidence_interval(values: pd.Series) -> tuple[float, float, float, float]:
    array = values.astype(float).to_numpy()
    n = int(array.size)
    mean = float(array.mean())
    if n <= 1:
        return mean, 0.0, mean, mean
    std = float(array.std(ddof=1))
    sem = std / math.sqrt(n)
    half_width = float(stats.t.ppf(0.975, df=n - 1) * sem)
    return mean, std, mean - half_width, mean + half_width


def summarize_runs(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (scenario, algorithm), group in df.groupby(["env_id", "algorithm"], sort=False):
        row: dict[str, Any] = {
            "env_id": scenario,
            "scenario_label": scenario_label(scenario),
            "algorithm": algorithm,
            "algorithm_label": algorithm_label(algorithm),
            "n": int(group.shape[0]),
        }
        for metric in SUMMARY_METRICS:
            mean, std, ci_low, ci_high = confidence_interval(group[metric])
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
            row[f"{metric}_ci95_low"] = ci_low
            row[f"{metric}_ci95_high"] = ci_high
            row[f"{metric}_ci95_half"] = ci_high - mean
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_overall(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for algorithm, group in df.groupby("algorithm", sort=False):
        row: dict[str, Any] = {
            "algorithm": algorithm,
            "algorithm_label": algorithm_label(algorithm),
            "n": int(group.shape[0]),
        }
        for metric in ["avg_reward", "avg_cost", "collision_rate"]:
            mean, std, ci_low, ci_high = confidence_interval(group[metric])
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
            row[f"{metric}_ci95_low"] = ci_low
            row[f"{metric}_ci95_high"] = ci_high
        rows.append(row)
    return pd.DataFrame(rows)


def compute_significance(df: pd.DataFrame, baseline: str | None = "ppo") -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    metrics = ["collision_rate", "avg_reward", "avg_cost"]

    if baseline is None:
        for scenario, scenario_group in df.groupby("env_id", sort=False):
            algorithms = list(dict.fromkeys(scenario_group["algorithm"].tolist()))
            for idx, left in enumerate(algorithms):
                for right in algorithms[idx + 1 :]:
                    left_group = scenario_group[scenario_group["algorithm"] == left]
                    right_group = scenario_group[scenario_group["algorithm"] == right]
                    row = {
                        "env_id": scenario,
                        "scenario_label": scenario_label(scenario),
                        "comparison": f"{algorithm_label(left)} vs {algorithm_label(right)}",
                    }
                    for metric in metrics:
                        if len(left_group) < 2 or len(right_group) < 2:
                            row[f"{metric}_pvalue"] = math.nan
                        else:
                            _, pvalue = stats.ttest_ind(
                                left_group[metric].to_numpy(),
                                right_group[metric].to_numpy(),
                                equal_var=False,
                            )
                            row[f"{metric}_pvalue"] = float(pvalue)
                    rows.append(row)
        return pd.DataFrame(rows)

    for scenario, scenario_group in df.groupby("env_id", sort=False):
        baseline_group = scenario_group[scenario_group["algorithm"] == baseline]
        if baseline_group.empty:
            continue
        for algorithm, group in scenario_group.groupby("algorithm", sort=False):
            if algorithm == baseline:
                continue
            row = {
                "env_id": scenario,
                "scenario_label": scenario_label(scenario),
                "baseline": baseline,
                "algorithm": algorithm,
                "algorithm_label": algorithm_label(algorithm),
                "comparison": f"{algorithm_label(algorithm)} vs {algorithm_label(baseline)}",
            }
            for metric in metrics:
                if len(group) < 2 or len(baseline_group) < 2:
                    row[f"{metric}_pvalue"] = math.nan
                else:
                    _, pvalue = stats.ttest_ind(
                        group[metric].to_numpy(),
                        baseline_group[metric].to_numpy(),
                        equal_var=False,
                    )
                    row[f"{metric}_pvalue"] = float(pvalue)
            rows.append(row)
    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame, columns: list[str], float_digits: int = 3) -> str:
    if df.empty:
        return "_No data_"
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = [header, separator]
    for _, row in df[columns].iterrows():
        cells: list[str] = []
        for value in row:
            if isinstance(value, float):
                if math.isnan(value):
                    cells.append("nan")
                else:
                    cells.append(f"{value:.{float_digits}f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def plot_grouped_metric(summary_df: pd.DataFrame, metric: str, ylabel: str, save_path: Path) -> None:
    scenarios = list(dict.fromkeys(summary_df["env_id"].tolist()))
    algorithms = list(dict.fromkeys(summary_df["algorithm"].tolist()))
    x = np.arange(len(scenarios))
    width = 0.22

    fig, ax = plt.subplots(figsize=(11, 5))
    for idx, algorithm in enumerate(algorithms):
        subset = (
            summary_df[summary_df["algorithm"] == algorithm]
            .set_index("env_id")
            .reindex(scenarios)
        )
        offset = (idx - (len(algorithms) - 1) / 2) * width
        ax.bar(
            x + offset,
            subset[f"{metric}_mean"],
            width=width,
            yerr=subset[f"{metric}_ci95_half"],
            capsize=4,
            label=algorithm_label(algorithm),
        )

    ax.set_xticks(x)
    ax.set_xticklabels([scenario_label(name) for name in scenarios])
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} by Scenario")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_reward_safety(summary_df: pd.DataFrame, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    for _, row in summary_df.iterrows():
        ax.scatter(
            row["collision_rate_mean"],
            row["avg_reward_mean"],
            s=110,
            label=f"{row['algorithm_label']} / {row['scenario_label']}",
        )
        ax.annotate(
            f"{row['algorithm_label']} / {row['scenario_label']}",
            (row["collision_rate_mean"], row["avg_reward_mean"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )
    ax.set_xlabel("Collision Rate (%)")
    ax.set_ylabel("Average Reward")
    ax.set_title("Reward vs. Safety Frontier")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_report(
    report_path: Path,
    args: argparse.Namespace,
    raw_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    overall_df: pd.DataFrame,
    significance_df: pd.DataFrame,
) -> None:
    scenario_table = summary_df[
        [
            "scenario_label",
            "algorithm_label",
            "n",
            "collision_rate_mean",
            "avg_reward_mean",
            "avg_cost_mean",
            "avg_length_mean",
        ]
    ].sort_values(["scenario_label", "collision_rate_mean", "avg_reward_mean"])
    overall_table = overall_df[
        [
            "algorithm_label",
            "n",
            "collision_rate_mean",
            "avg_reward_mean",
            "avg_cost_mean",
        ]
    ].sort_values("collision_rate_mean")
    significance_columns = [
        "scenario_label",
        "comparison",
        "collision_rate_pvalue",
        "avg_reward_pvalue",
        "avg_cost_pvalue",
    ]
    significance_table = (
        significance_df[significance_columns].sort_values(["scenario_label", "comparison"])
        if not significance_df.empty
        else significance_df
    )

    best_safety_rows = []
    for scenario_name, group in summary_df.groupby("scenario_label", sort=False):
        safest = group.loc[group["collision_rate_mean"].idxmin()]
        highest_reward = group.loc[group["avg_reward_mean"].idxmax()]
        best_safety_rows.append(
            f"- {scenario_name}: safest was {safest['algorithm_label']} ({safest['collision_rate_mean']:.2f}% collision), while highest reward was {highest_reward['algorithm_label']} ({highest_reward['avg_reward_mean']:.2f})."
        )

    significance_title = "## Significance Tests"
    significance_note = "Welch t-tests were computed per scenario when at least two seeds were available for both methods."
    if str(args.baseline).lower() != "none":
        significance_title = f"## Significance vs {algorithm_label(normalize_algorithm(args.baseline))}"

    lines = [
        "# Safe RL Roadmap Report",
        "",
        "## Sweep Configuration",
        f"- Algorithms: {', '.join(algorithm_label(a) for a in args.algorithms)}",
        f"- Scenarios: {', '.join(scenario_label(s) for s in args.scenarios)}",
        f"- Seeds: {', '.join(str(seed) for seed in args.seeds)}",
        f"- Training budget per run: {args.epochs} epochs x {args.steps_per_epoch} steps",
        f"- Evaluation budget per run: {args.eval_episodes} episodes",
        "",
        "## Scenario Summary",
        markdown_table(
            scenario_table,
            [
                "scenario_label",
                "algorithm_label",
                "n",
                "collision_rate_mean",
                "avg_reward_mean",
                "avg_cost_mean",
                "avg_length_mean",
            ],
        ),
        "",
        "## Overall Summary",
        markdown_table(
            overall_table,
            [
                "algorithm_label",
                "n",
                "collision_rate_mean",
                "avg_reward_mean",
                "avg_cost_mean",
            ],
        ),
        "",
        "## Tradeoff Takeaways",
        *best_safety_rows,
        "",
        significance_title,
        significance_note,
        markdown_table(significance_table, significance_columns),
        "",
        "## Roadmap Status",
        "- [x] Cross-scenario evaluation",
        "- [x] Algorithm comparison in the discrete-action setting (PPO, PPOLag, CPO)",
        "- [x] Multi-seed runs",
        "- [x] Statistical analysis with confidence intervals and pairwise tests",
        "- [x] Final report",
        "",
        "## OmniSafe Compatibility Note",
        "- OmniSafe 0.5.0 was audited in the new environment.",
        "- The current release exposes `CPO`, `PPOLag`, `SACLag`, and `SACPID`, but not `WCSAC`.",
        "- Its adapters wrap environments with `ActionScale`, which asserts `spaces.Box` actions, while this repo uses discrete-action highway-env tasks.",
        "- A faithful OmniSafe off-policy comparison therefore requires a separate continuous-action refactor rather than a drop-in baseline.",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.algorithms = [normalize_algorithm(name) for name in args.algorithms]
    baseline = None if str(args.baseline).lower() == "none" else normalize_algorithm(args.baseline)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    for scenario in args.scenarios:
        for algorithm in args.algorithms:
            for seed in args.seeds:
                config = build_config(args, scenario, algorithm, seed)
                payload = existing_or_run(config, args.skip_existing)
                records.append(payload["summary"])

    raw_df = pd.DataFrame(records).sort_values(["env_id", "algorithm", "seed"])
    summary_df = summarize_runs(raw_df).sort_values(["env_id", "algorithm"])
    overall_df = summarize_overall(raw_df).sort_values("algorithm")
    significance_df = compute_significance(raw_df, baseline=baseline)
    if not significance_df.empty:
        significance_df = significance_df.sort_values(["env_id", "comparison"])

    raw_df.to_csv(output_dir / "run_metrics.csv", index=False)
    summary_df.to_csv(output_dir / "scenario_summary.csv", index=False)
    overall_df.to_csv(output_dir / "overall_summary.csv", index=False)
    significance_df.to_csv(output_dir / "significance_vs_ppo.csv", index=False)
    significance_df.to_csv(output_dir / "significance_tests.csv", index=False)
    (output_dir / "run_metrics.json").write_text(
        json.dumps(records, indent=2),
        encoding="utf-8",
    )
    (output_dir / "sweep_config.json").write_text(
        json.dumps(
            {
                "algorithms": args.algorithms,
                "scenarios": args.scenarios,
                "seeds": args.seeds,
                "epochs": args.epochs,
                "steps_per_epoch": args.steps_per_epoch,
                "repeat_per_collect": args.repeat_per_collect,
                "episode_per_collect": args.episode_per_collect,
                "train_envs": args.train_envs,
                "test_envs": args.test_envs,
                "test_episodes": args.test_episodes,
                "eval_episodes": args.eval_episodes,
                "cost_limit": args.cost_limit,
                "device": args.device,
                "output_dir": str(output_dir),
                "skip_existing": bool(args.skip_existing),
                "cpo_critic_iters": args.cpo_critic_iters,
                "baseline": None if baseline is None else baseline,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    plot_grouped_metric(
        summary_df,
        metric="collision_rate",
        ylabel="Collision Rate (%)",
        save_path=output_dir / "collision_rate_by_scenario.png",
    )
    plot_grouped_metric(
        summary_df,
        metric="avg_reward",
        ylabel="Average Reward",
        save_path=output_dir / "reward_by_scenario.png",
    )
    plot_reward_safety(summary_df, output_dir / "reward_vs_safety.png")

    write_report(
        output_dir / "ROADMAP_REPORT.md",
        args,
        raw_df,
        summary_df,
        overall_df,
        significance_df,
    )
    print(f"Saved roadmap artifacts to {output_dir}")


if __name__ == "__main__":
    main()
