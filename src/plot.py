"""Utilities for plotting experiment results."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

from src.env import CliffWalkingEnv, Position


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """Compute a simple moving average for smoother curves."""
    if window <= 1:
        return values.copy()

    averaged = np.zeros_like(values, dtype=np.float64)
    for index in range(len(values)):
        start = max(0, index - window + 1)
        averaged[index] = np.mean(values[start : index + 1])
    return averaged


def rolling_std(values: np.ndarray, window: int) -> np.ndarray:
    """Compute a rolling standard deviation for one reward sequence."""
    if window <= 1:
        return np.zeros_like(values, dtype=np.float64)

    std_values = np.zeros_like(values, dtype=np.float64)
    for index in range(len(values)):
        start = max(0, index - window + 1)
        std_values[index] = np.std(values[start : index + 1])
    return std_values


def plot_reward_curves(
    reward_by_agent: Dict[str, np.ndarray],
    output_path: Path,
    moving_average_window: int = 10,
) -> None:
    """Plot raw reward curves and moving averages for multiple agents."""
    plt.figure(figsize=(10, 6))

    colors = {
        "Q-learning": "tab:blue",
        "SARSA": "tab:orange",
    }

    for agent_name, values in reward_by_agent.items():
        color = colors.get(agent_name)
        smoothed = moving_average(values, moving_average_window)
        episodes = np.arange(1, len(values) + 1)

        plt.plot(
            episodes,
            values,
            label=f"{agent_name} reward",
            linewidth=1.0,
            alpha=0.35,
            color=color,
        )
        plt.plot(
            episodes,
            smoothed,
            label=f"{agent_name} moving average",
            linewidth=2.2,
            color=color,
        )

    plt.title("Q-learning vs SARSA Reward Curve")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_path_map(
    env: CliffWalkingEnv,
    path: list[Position],
    title: str,
    output_path: Path,
) -> None:
    """Render one final greedy path as a simple grid image.

    Visual interpretation:
    - cells near the cliff on the bottom row are riskier
    - a path that stays one row above the cliff longer is usually more
      conservative, while a path that hugs the cliff is more aggressive
    """
    grid = np.zeros((env.rows, env.cols), dtype=np.int32)

    for row in range(env.rows):
        for col in range(env.cols):
            position = (row, col)
            if env.is_cliff(position):
                grid[row, col] = 1

    grid[env.start_pos] = 2
    grid[env.goal_pos] = 3

    plt.figure(figsize=(12, 4))
    plt.imshow(grid, cmap="Pastel1", vmin=0, vmax=3)

    path_rows = [position[0] for position in path]
    path_cols = [position[1] for position in path]
    plt.plot(path_cols, path_rows, color="black", linewidth=2.5, marker="o", markersize=5)

    for row in range(env.rows):
        for col in range(env.cols):
            position = (row, col)
            if position == env.start_pos:
                label = "S"
            elif position == env.goal_pos:
                label = "G"
            elif env.is_cliff(position):
                label = "C"
            elif position in path:
                label = "*"
            else:
                label = "."
            plt.text(col, row, label, ha="center", va="center", fontsize=11, color="black")

    plt.title(title)
    plt.xlabel("Column")
    plt.ylabel("Row")
    plt.xticks(np.arange(env.cols))
    plt.yticks(np.arange(env.rows))
    plt.grid(color="gray", linestyle="-", linewidth=0.8, alpha=0.5)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_stability_analysis(
    reward_matrix_by_agent: Dict[str, np.ndarray],
    output_path: Path,
    moving_average_window: int = 10,
    rolling_std_window: int = 10,
) -> None:
    """Plot average reward and reward volatility across repeated runs.

    The upper subplot shows the mean reward curve and its moving average across
    runs. The lower subplot shows rolling standard deviation, which is a simple
    measure of how unstable or noisy the reward sequence is during training.
    """
    fig, axes = plt.subplots(2, 1, figsize=(11, 9), sharex=True)
    colors = {
        "Q-learning": "tab:blue",
        "SARSA": "tab:orange",
    }

    for agent_name, reward_matrix in reward_matrix_by_agent.items():
        color = colors.get(agent_name)
        mean_rewards = np.mean(reward_matrix, axis=0)
        moving_avg = moving_average(mean_rewards, moving_average_window)
        per_run_rolling_std = np.vstack(
            [rolling_std(run_rewards, rolling_std_window) for run_rewards in reward_matrix]
        )
        mean_rolling_std = np.mean(per_run_rolling_std, axis=0)
        episodes = np.arange(1, len(mean_rewards) + 1)

        axes[0].plot(
            episodes,
            mean_rewards,
            label=f"{agent_name} mean reward",
            linewidth=1.0,
            alpha=0.35,
            color=color,
        )
        axes[0].plot(
            episodes,
            moving_avg,
            label=f"{agent_name} moving average",
            linewidth=2.2,
            color=color,
        )
        axes[1].plot(
            episodes,
            mean_rolling_std,
            label=f"{agent_name} rolling std",
            linewidth=2.0,
            color=color,
        )

    axes[0].set_title("Q-learning vs SARSA Stability Analysis")
    axes[0].set_ylabel("Mean Total Reward")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.4)

    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Rolling Reward Std")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_exploration_analysis(
    reward_matrix_by_epsilon: Dict[float, np.ndarray],
    agent_name: str,
    output_path: Path,
    moving_average_window: int = 10,
) -> None:
    """Plot epsilon sensitivity for one agent across repeated runs.

    Each epsilon is shown with a mean reward curve and its moving average so the
    effect of exploration rate on convergence can be compared directly.
    """
    plt.figure(figsize=(11, 6.5))
    cmap = plt.get_cmap("viridis")
    epsilon_values = sorted(reward_matrix_by_epsilon)

    for index, epsilon in enumerate(epsilon_values):
        reward_matrix = reward_matrix_by_epsilon[epsilon]
        mean_rewards = np.mean(reward_matrix, axis=0)
        moving_avg = moving_average(mean_rewards, moving_average_window)
        episodes = np.arange(1, len(mean_rewards) + 1)
        color = cmap(index / max(1, len(epsilon_values) - 1))

        plt.plot(
            episodes,
            mean_rewards,
            color=color,
            linewidth=1.0,
            alpha=0.3,
            label=f"epsilon={epsilon:.2f} mean reward",
        )
        plt.plot(
            episodes,
            moving_avg,
            color=color,
            linewidth=2.2,
            label=f"epsilon={epsilon:.2f} moving average",
        )

    plt.title(f"{agent_name} Exploration Sensitivity")
    plt.xlabel("Episode")
    plt.ylabel("Mean Total Reward")
    plt.legend(ncol=2, fontsize=9)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_exploration_summary(
    epsilon_values: list[float],
    q_learning_final_rewards: list[float],
    sarsa_final_rewards: list[float],
    q_learning_convergence: list[int],
    sarsa_convergence: list[int],
    output_path: Path,
) -> None:
    """Plot summary comparison of final reward and convergence speed."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    x = np.arange(len(epsilon_values))
    width = 0.36
    labels = [f"{epsilon:.2f}" for epsilon in epsilon_values]

    axes[0].bar(x - width / 2, q_learning_final_rewards, width, label="Q-learning", color="tab:blue")
    axes[0].bar(x + width / 2, sarsa_final_rewards, width, label="SARSA", color="tab:orange")
    axes[0].set_title("Final Mean Reward by Epsilon")
    axes[0].set_xlabel("Epsilon")
    axes[0].set_ylabel("Final Mean Reward")
    axes[0].set_xticks(x, labels)
    axes[0].legend()
    axes[0].grid(True, axis="y", linestyle="--", alpha=0.4)

    axes[1].bar(x - width / 2, q_learning_convergence, width, label="Q-learning", color="tab:blue")
    axes[1].bar(x + width / 2, sarsa_convergence, width, label="SARSA", color="tab:orange")
    axes[1].set_title("Convergence Episode by Epsilon")
    axes[1].set_xlabel("Epsilon")
    axes[1].set_ylabel("Episode")
    axes[1].set_xticks(x, labels)
    axes[1].legend()
    axes[1].grid(True, axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)
