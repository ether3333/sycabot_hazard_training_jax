"""Draw a task-rescue-rate box plot from saved PPO delivered data.

The script looks for the newest saved rollout CSV, then computes one value per
episode:

    max(delivered) / num_tasks * 100

Expected input is either:
    delivered_timeseries.csv  with columns episode, delivered, num_tasks
    episode_metrics.csv       with columns episode, max_delivered, num_tasks

Usage:
    python3 plot_boxplot.py
    python3 plot_boxplot.py --episodes 30
    python3 plot_boxplot.py --episodes all
"""

from __future__ import annotations

import argparse
import csv
import glob
import os

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Draw a box plot from saved PPO delivered-task data."
    )
    parser.add_argument("--data", type=str, default=None,
                        help="CSV path. Default: newest delivered/episode metrics CSV.")
    
    # ⬇️ THE FIX IS HERE ⬇️
    # We removed choices=["8", "30", "all"] so it accepts any string/number
    parser.add_argument("--episodes", type=str, default="8",
                        help="Number of random episodes to include (e.g., '5', '8', 'all').")
    
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for episode sampling.")
    parser.add_argument("--out-dir", type=str, default="boxplot_results",
                        help="Directory for the plot and selected CSV.")
    parser.add_argument("--tasks", type=int, default=None,
                        help="Task count if the CSV does not contain num_tasks.")
    parser.add_argument("--robots", type=int, default=None,
                        help="Robot count if the CSV does not contain num_robots.")
    return parser.parse_args()


def find_newest_data() -> str:
    patterns = [
        "**/delivered_timeseries.csv",
        "**/episode_metrics.csv",
    ]
    candidates = []
    for pattern in patterns:
        for path in glob.glob(pattern, recursive=True):
            parts = path.split(os.sep)
            if ".venv" in parts or "boxplot_results" in parts:
                continue
            if os.path.isfile(path):
                candidates.append(path)

    if not candidates:
        raise FileNotFoundError(
            "No delivered data CSV found. Expected a delivered_timeseries.csv "
            "or episode_metrics.csv from test_and_visualize.py."
        )
    return max(candidates, key=os.path.getmtime)


def _first_int(row: dict, key: str, fallback: int | None) -> int | None:
    if fallback is not None:
        return fallback
    value = row.get(key, "")
    if value == "":
        return None
    return int(float(value))


def load_rescue_rates(path: str, tasks_arg: int | None, robots_arg: int | None):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError(f"{path} is empty.")

    first = rows[0]
    num_tasks = _first_int(first, "num_tasks", tasks_arg)
    num_robots = _first_int(first, "num_robots", robots_arg)
    if num_tasks is None:
        raise ValueError(
            f"{path} does not contain num_tasks. Run again with --tasks N."
        )

    by_episode = {}
    if "delivered" in first:
        for row in rows:
            ep = int(float(row["episode"]))
            delivered = float(row["delivered"])
            by_episode[ep] = max(by_episode.get(ep, 0.0), delivered)
    elif "max_delivered" in first:
        for row in rows:
            ep = int(float(row["episode"]))
            by_episode[ep] = float(row["max_delivered"])
    else:
        raise ValueError(
            f"{path} must contain either delivered or max_delivered."
        )

    episode_rows = []
    for ep in sorted(by_episode):
        max_delivered = by_episode[ep]
        episode_rows.append({
            "episode": ep,
            "max_delivered": max_delivered,
            "tasks_rescued_pct": max_delivered / float(num_tasks) * 100.0,
        })

    return episode_rows, num_robots, num_tasks


def sample_rows(rows: list[dict], episodes: str, seed: int) -> list[dict]:
    if episodes == "all":
        return rows

    sample_size = int(episodes)
    if len(rows) < sample_size:
        raise ValueError(
            f"Requested {sample_size} episodes, but the data only has {len(rows)}."
        )

    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(rows), size=sample_size, replace=False))
    return [rows[int(i)] for i in idx]


def plot_boxplot(rows: list[dict], num_robots: int | None, num_tasks: int, out_dir: str):
    rates = [row["tasks_rescued_pct"] for row in rows]
    ep_count = len(rows)
    robot_label = num_robots if num_robots is not None else "unknown"
    title = f"r({robot_label})_t({num_tasks})_ep({ep_count})_task_rescue_rate"
    filename = f"r{robot_label}_t{num_tasks}_ep{ep_count}_task_rescue_rate"

    os.makedirs(out_dir, exist_ok=True)
    plot_path = os.path.join(out_dir, f"{filename}.png")
    csv_path = os.path.join(out_dir, f"{filename}.csv")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot([rates], labels=[f"{ep_count} episodes"])
    ax.scatter(np.ones(len(rates)), rates, color="purple", alpha=0.65, s=28)
    ax.plot([1], [np.median(rates)], marker="o", color="red", alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel("PPO")
    ax.set_ylabel("Tasks rescued (%)")
    ax.set_ylim(-5, 105)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["episode", "max_delivered", "tasks_rescued_pct"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved box plot -> {plot_path}")
    print(f"Saved selected data -> {csv_path}")


def main():
    args = parse_args()
    data_path = args.data or find_newest_data()
    print(f"Using data: {data_path}")

    rows, num_robots, num_tasks = load_rescue_rates(
        data_path, tasks_arg=args.tasks, robots_arg=args.robots)
    selected = sample_rows(rows, args.episodes, args.seed)
    plot_boxplot(selected, num_robots, num_tasks, args.out_dir)


if __name__ == "__main__":
    main()
