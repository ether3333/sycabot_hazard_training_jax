"""Create task-rescue-rate box plots from raw episode CSV files.

This script does not import JAX or the SycaBot environment, so it is independent
of the current NUM_ROBOTS / NUM_TASKS values in sycabot_env_jax.py.

Examples
--------
    python3 plot_task_rescue_boxplot.py \
        --ppo test_results/r1_t1_ep100_task_rescue_rate.csv \
        --conventional conventional_r1_t1.csv \
        --episodes 8 \
        --robots 1 \
        --tasks 1

    python3 plot_task_rescue_boxplot.py --ppo ppo.csv --conventional conv.csv \
        --episodes 30 --robots 1 --tasks 1

    python3 plot_task_rescue_boxplot.py --ppo ppo.csv --conventional conv.csv \
        --episodes all --robots 1 --tasks 1
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot task rescue rate box plots from raw episode CSV data."
    )
    parser.add_argument("--ppo", type=str, required=True,
                        help="CSV with PPO episode-level task rescue data")
    parser.add_argument("--conventional", type=str, default=None,
                        help="CSV with conventional-method episode-level task rescue data")
    parser.add_argument("--episodes", type=str, choices=["8", "30", "all"], required=True,
                        help="Number of random episodes to use, or all")
    parser.add_argument("--robots", type=int, required=True,
                        help="Number of robots for the plot title")
    parser.add_argument("--tasks", type=int, required=True,
                        help="Number of tasks for the plot title and percent conversion")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for episode subsampling")
    parser.add_argument("--out-dir", type=str, default="boxplot_results",
                        help="Directory for saved plot and selected raw data")
    return parser.parse_args()


def read_episode_csv(path: str, method: str, num_tasks: int) -> pd.DataFrame:
    """Read CSV and return columns: method, episode, tasks_rescued_pct."""
    df = pd.read_csv(path)
    cols = set(df.columns)

    if "tasks_rescued_pct" in cols:
        pct = df["tasks_rescued_pct"].astype(float)
    elif "task_rescue_rate_percent" in cols:
        pct = df["task_rescue_rate_percent"].astype(float)
    elif "max_delivered" in cols:
        pct = df["max_delivered"].astype(float) / float(num_tasks) * 100.0
    elif "delivered" in cols:
        if "episode" not in cols:
            raise ValueError(f"{path} has 'delivered' but no 'episode' column.")
        grouped = df.groupby("episode", as_index=False)["delivered"].max()
        grouped["tasks_rescued_pct"] = grouped["delivered"] / float(num_tasks) * 100.0
        grouped["method"] = method
        return grouped[["method", "episode", "tasks_rescued_pct"]]
    else:
        raise ValueError(
            f"{path} must contain one of: tasks_rescued_pct, "
            "task_rescue_rate_percent, max_delivered, delivered."
        )

    out = pd.DataFrame({
        "method": method,
        "episode": df["episode"] if "episode" in cols else np.arange(1, len(df) + 1),
        "tasks_rescued_pct": pct,
    })
    return out


def sample_episodes(raw_df: pd.DataFrame, episodes: str, seed: int) -> pd.DataFrame:
    """Sample the same requested count independently for each method."""
    if episodes == "all":
        return raw_df.copy()

    sample_size = int(episodes)
    sampled = []
    rng = np.random.default_rng(seed)
    for method, group in raw_df.groupby("method", sort=False):
        if len(group) < sample_size:
            raise ValueError(
                f"{method} has only {len(group)} episodes, but --episodes {sample_size} "
                "was requested."
            )
        chosen = rng.choice(group.index.to_numpy(), size=sample_size, replace=False)
        sampled.append(group.loc[np.sort(chosen)])
    return pd.concat(sampled, ignore_index=True)


def plot_tasks_rescued_boxplot(
    raw_df: pd.DataFrame,
    group_col: str,
    title: str,
    xlabel: str,
    save_path: str,
    tasks_rescued_col: str = "tasks_rescued_pct",
    figsize: tuple[int, int] = (8, 5),
):
    """Plot a boxplot for tasks rescued per episode, grouped by x-axis value."""
    unique_groups = list(raw_df[group_col].dropna().unique())
    data = [
        raw_df.loc[raw_df[group_col] == value, tasks_rescued_col].tolist()
        for value in unique_groups
    ]
    medians = [pd.Series(group_data).median() for group_data in data]
    xpos = range(1, len(unique_groups) + 1)

    fig, ax = plt.subplots(figsize=figsize)
    ax.boxplot(data, labels=unique_groups)
    ax.plot(xpos, medians, marker="o", color="red", alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Tasks rescued (%)")
    ax.set_ylim(-5, 105)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig, ax


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    frames = [read_episode_csv(args.ppo, "PPO", args.tasks)]
    if args.conventional is not None:
        frames.append(read_episode_csv(args.conventional, "Conventional", args.tasks))

    raw_df = pd.concat(frames, ignore_index=True)
    sampled_df = sample_episodes(raw_df, args.episodes, args.seed)
    ep_label = str(len(sampled_df)) if args.conventional is None else args.episodes
    if args.episodes == "all":
        per_method_counts = sampled_df.groupby("method").size().astype(str).tolist()
        ep_label = "all-" + "-".join(per_method_counts)

    title = f"r({args.robots})_t({args.tasks})_ep({ep_label})_task_rescue_rate"
    filename_stub = (
        f"r{args.robots}_t{args.tasks}_ep{ep_label}_task_rescue_rate"
        .replace("all-", "all_")
        .replace("/", "_")
    )
    plot_path = os.path.join(args.out_dir, f"{filename_stub}.png")
    csv_path = os.path.join(args.out_dir, f"{filename_stub}.csv")

    fig, _ = plot_tasks_rescued_boxplot(
        raw_df=sampled_df,
        group_col="method",
        title=title,
        xlabel="Method",
        save_path=plot_path,
    )
    plt.close(fig)
    sampled_df.to_csv(csv_path, index=False)

    print(f"Saved box plot -> {plot_path}")
    print(f"Saved selected data -> {csv_path}")


if __name__ == "__main__":
    main()
