"""Training entry point for comparing Q-learning and SARSA on Cliff Walking.

Example:
    python -m src.train
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np

from src.agents import AgentConfig, QLearningAgent, SARSAAgent
from src.env import CliffWalkingEnv, Position
from src.plot import (
    moving_average,
    plot_exploration_analysis,
    plot_exploration_summary,
    plot_path_map,
    plot_reward_curves,
    plot_stability_analysis,
    rolling_std,
)


RESULTS_DIR = Path("results")


@dataclass
class ExperimentConfig:
    """Configuration used by both Q-learning and SARSA experiments."""

    rows: int = 4
    cols: int = 12
    episodes: int = 500
    max_steps_per_episode: int = 1000
    alpha: float = 0.1
    gamma: float = 0.9
    epsilon: float = 0.1
    seed: int = 42
    moving_average_window: int = 10
    rolling_std_window: int = 10
    analysis_runs: int = 10
    exploration_epsilons: tuple[float, ...] = (0.01, 0.1, 0.2)


@dataclass
class TrainingResult:
    """Artifacts produced by training one agent."""

    rewards: List[float]
    q_table: np.ndarray
    greedy_path: List[Position]
    greedy_path_text: str
    greedy_policy_text: str
    greedy_total_reward: float
    path_risk_note: str
    path_risk_level: str


@dataclass
class StabilityAnalysisResult:
    """Aggregated stability statistics across repeated runs."""

    reward_matrix: np.ndarray
    mean_rewards: np.ndarray
    moving_average_rewards: np.ndarray
    mean_rolling_std: np.ndarray
    final_mean_reward: float
    final_mean_rolling_std: float
    convergence_episode: int


@dataclass
class ExplorationSettingResult:
    """Summary of one epsilon setting for one algorithm."""

    epsilon: float
    representative_result: TrainingResult
    stability: StabilityAnalysisResult


@dataclass
class ExplorationExperimentResult:
    """All epsilon settings for one algorithm."""

    agent_name: str
    by_epsilon: Dict[float, ExplorationSettingResult]


def run_q_learning_episode(
    env: CliffWalkingEnv,
    agent: QLearningAgent,
    max_steps: int,
) -> float:
    """Run one training episode for Q-learning and return total reward."""
    state = env.reset()
    total_reward = 0.0

    for _ in range(max_steps):
        action = agent.select_action(state)
        result = env.step(action)

        agent.update(
            state=state,
            action=action,
            reward=result.reward,
            next_state=result.next_state,
            done=result.done,
        )

        total_reward += result.reward
        state = result.next_state
        if result.done:
            break

    return total_reward


def run_sarsa_episode(
    env: CliffWalkingEnv,
    agent: SARSAAgent,
    max_steps: int,
) -> float:
    """Run one training episode for SARSA and return total reward."""
    state = env.reset()
    action = agent.select_action(state)
    total_reward = 0.0

    for _ in range(max_steps):
        result = env.step(action)
        total_reward += result.reward

        next_action = None if result.done else agent.select_action(result.next_state)
        agent.update(
            state=state,
            action=action,
            reward=result.reward,
            next_state=result.next_state,
            done=result.done,
            next_action=next_action,
        )

        state = result.next_state
        if result.done:
            break

        action = next_action

    return total_reward


def evaluate_greedy_path(
    env: CliffWalkingEnv,
    q_table: np.ndarray,
    max_steps: int = 100,
) -> Dict[str, object]:
    """Evaluate the final greedy behavior implied by a Q-table.

    Returns:
    - `path`: visited grid positions from start to finish
    - `path_text`: text visualization of the final path
    - `policy_text`: text visualization of the greedy policy over the grid
    - `total_reward`: reward collected along the greedy rollout
    """
    path = env.rollout_greedy_path(q_table, max_steps=max_steps)
    path_text = env.render_path(path)
    policy_text = env.render_policy(q_table)

    eval_env = CliffWalkingEnv(rows=env.rows, cols=env.cols)
    state = eval_env.reset()
    total_reward = 0.0

    for _ in range(max_steps):
        action = int(np.argmax(q_table[state]))
        result = eval_env.step(action)
        total_reward += result.reward
        state = result.next_state
        if result.done:
            break

    return {
        "path": path,
        "path_text": path_text,
        "policy_text": policy_text,
        "total_reward": total_reward,
    }


def classify_path_risk(env: CliffWalkingEnv, path: List[Position]) -> str:
    """Classify whether the final path looks aggressive or conservative.

    In Cliff Walking, a policy is usually considered more aggressive when it
    travels close to the cliff to minimize step count. A policy is more
    conservative when it stays higher above the cliff, accepting extra steps to
    reduce the chance of a costly cliff fall during exploratory training.
    """
    unique_positions = list(dict.fromkeys(path))
    intermediate_positions = [
        position
        for position in unique_positions
        if position != env.start_pos and position != env.goal_pos
    ]
    min_distance_to_cliff = (
        min(abs(position[0] - (env.rows - 1)) for position in intermediate_positions)
        if intermediate_positions
        else env.rows
    )

    cells_above_cliff = [
        position
        for position in unique_positions
        if position[0] == env.rows - 2 and 1 <= position[1] <= env.cols - 2
    ]

    if len(cells_above_cliff) >= max(1, env.cols // 3):
        return "aggressive"

    if min_distance_to_cliff >= 2:
        return "conservative"

    return "moderate"


def analyze_path_risk(env: CliffWalkingEnv, path: List[Position]) -> str:
    """Explain whether the final path looks aggressive or conservative."""
    risk_level = classify_path_risk(env, path)
    if risk_level == "aggressive":
        return (
            "This path is relatively aggressive: it spends many steps directly "
            "above the cliff, which is shorter but riskier during exploration."
        )
    if risk_level == "conservative":
        return (
            "This path is relatively conservative: it stays farther away from "
            "the cliff, which is safer but usually requires more steps."
        )
    return (
        "This path shows a moderate trade-off: it approaches the cliff in some "
        "segments but does not hug the cliff for the entire route."
    )


def build_path_report(agent_name: str, result: TrainingResult) -> str:
    """Build a readable text report for one final greedy path."""
    path_sequence = " -> ".join(f"({row},{col})" for row, col in result.greedy_path)
    return (
        f"{agent_name} final greedy path\n"
        f"{'=' * (len(agent_name) + 18)}\n\n"
        "Map legend:\n"
        "S = Start\n"
        "G = Goal\n"
        "C = Cliff\n"
        "* = Final greedy path\n\n"
        "Path map:\n"
        f"{result.greedy_path_text}\n\n"
        "Greedy policy:\n"
        f"{result.greedy_policy_text}\n\n"
        "Path sequence:\n"
        f"{path_sequence}\n\n"
        "Interpretation:\n"
        f"{result.path_risk_note}\n"
    )


def _build_agent_config(env: CliffWalkingEnv, config: ExperimentConfig) -> AgentConfig:
    """Create a shared agent configuration from experiment settings."""
    return AgentConfig(
        n_states=env.n_states,
        n_actions=env.n_actions,
        alpha=config.alpha,
        gamma=config.gamma,
        epsilon=config.epsilon,
        seed=config.seed,
    )


def _config_with_seed(config: ExperimentConfig, seed: int) -> ExperimentConfig:
    """Return a copy of the experiment configuration with a specific seed."""
    return replace(config, seed=seed)


def train_q_learning(config: ExperimentConfig) -> TrainingResult:
    """Train a Q-learning agent and return reward history and final artifacts."""
    env = CliffWalkingEnv(rows=config.rows, cols=config.cols)
    agent = QLearningAgent(_build_agent_config(env, config))
    rewards: List[float] = []

    for _ in range(config.episodes):
        episode_reward = run_q_learning_episode(env, agent, config.max_steps_per_episode)
        rewards.append(episode_reward)

    evaluation = evaluate_greedy_path(env, agent.q_table)
    return TrainingResult(
        rewards=rewards,
        q_table=agent.q_table.copy(),
        greedy_path=evaluation["path"],
        greedy_path_text=evaluation["path_text"],
        greedy_policy_text=evaluation["policy_text"],
        greedy_total_reward=float(evaluation["total_reward"]),
        path_risk_note=analyze_path_risk(env, evaluation["path"]),
        path_risk_level=classify_path_risk(env, evaluation["path"]),
    )


def train_sarsa(config: ExperimentConfig) -> TrainingResult:
    """Train a SARSA agent and return reward history and final artifacts."""
    env = CliffWalkingEnv(rows=config.rows, cols=config.cols)
    agent_config = _build_agent_config(env, config)
    agent = SARSAAgent(agent_config)
    rewards: List[float] = []

    for _ in range(config.episodes):
        episode_reward = run_sarsa_episode(env, agent, config.max_steps_per_episode)
        rewards.append(episode_reward)

    evaluation = evaluate_greedy_path(env, agent.q_table)
    return TrainingResult(
        rewards=rewards,
        q_table=agent.q_table.copy(),
        greedy_path=evaluation["path"],
        greedy_path_text=evaluation["path_text"],
        greedy_policy_text=evaluation["policy_text"],
        greedy_total_reward=float(evaluation["total_reward"]),
        path_risk_note=analyze_path_risk(env, evaluation["path"]),
        path_risk_level=classify_path_risk(env, evaluation["path"]),
    )


def estimate_convergence_episode(
    moving_average_rewards: np.ndarray,
    tolerance_ratio: float = 0.05,
) -> int:
    """Estimate when training reaches near-final performance.

    The convergence episode is defined as the first episode where the moving
    average reward reaches within `tolerance_ratio` of the final moving average.
    Reward is higher when it is less negative, so the threshold is built around
    the final value while accounting for scale.
    """
    final_value = float(moving_average_rewards[-1])
    tolerance = tolerance_ratio * max(1.0, abs(final_value))
    threshold = final_value - tolerance

    for episode_index, value in enumerate(moving_average_rewards, start=1):
        if value >= threshold:
            return episode_index
    return len(moving_average_rewards)


def run_stability_analysis(
    train_fn: Callable[[ExperimentConfig], TrainingResult],
    config: ExperimentConfig,
    seed_offset: int = 0,
) -> StabilityAnalysisResult:
    """Repeat training multiple times and aggregate reward stability metrics."""
    reward_runs = []
    for run_index in range(config.analysis_runs):
        run_seed = config.seed + seed_offset + run_index
        run_config = _config_with_seed(config, run_seed)
        training_result = train_fn(run_config)
        reward_runs.append(np.array(training_result.rewards, dtype=np.float64))

    reward_matrix = np.vstack(reward_runs)
    mean_rewards = np.mean(reward_matrix, axis=0)
    moving_average_rewards = np.array(
        [
            np.mean(
                mean_rewards[max(0, index - config.moving_average_window + 1) : index + 1]
            )
            for index in range(len(mean_rewards))
        ],
        dtype=np.float64,
    )
    rolling_std_runs = np.vstack(
        [rolling_std(run_rewards, config.rolling_std_window) for run_rewards in reward_matrix]
    )
    mean_rolling_std = np.mean(rolling_std_runs, axis=0)

    return StabilityAnalysisResult(
        reward_matrix=reward_matrix,
        mean_rewards=mean_rewards,
        moving_average_rewards=moving_average_rewards,
        mean_rolling_std=mean_rolling_std,
        final_mean_reward=float(mean_rewards[-1]),
        final_mean_rolling_std=float(mean_rolling_std[-1]),
        convergence_episode=estimate_convergence_episode(moving_average_rewards),
    )


def run_exploration_experiment(
    train_fn: Callable[[ExperimentConfig], TrainingResult],
    agent_name: str,
    config: ExperimentConfig,
    epsilon_values: tuple[float, ...],
    seed_offset: int = 0,
) -> ExplorationExperimentResult:
    """Repeat experiments for several epsilon values under fixed settings."""
    by_epsilon: Dict[float, ExplorationSettingResult] = {}

    for epsilon_index, epsilon in enumerate(epsilon_values):
        epsilon_config = replace(config, epsilon=epsilon, seed=config.seed + epsilon_index * 100)
        representative_result = train_fn(epsilon_config)
        stability = run_stability_analysis(
            train_fn,
            epsilon_config,
            seed_offset=seed_offset + epsilon_index * 1000,
        )
        by_epsilon[epsilon] = ExplorationSettingResult(
            epsilon=epsilon,
            representative_result=representative_result,
            stability=stability,
        )

    return ExplorationExperimentResult(agent_name=agent_name, by_epsilon=by_epsilon)


def save_rewards_csv(rewards: List[float], agent_name: str, output_path: Path) -> None:
    """Save one agent's reward sequence to CSV."""
    reward_array = np.array(rewards, dtype=np.float64)
    episodes = np.arange(1, len(reward_array) + 1, dtype=np.int32)
    stacked = np.column_stack((episodes, reward_array))
    np.savetxt(
        output_path,
        stacked,
        delimiter=",",
        fmt=["%d", "%.6f"],
        header=f"episode,{agent_name}_reward",
        comments="",
    )


def save_q_table(q_table: np.ndarray, output_path: Path) -> None:
    """Save a Q-table to CSV where each row is one state."""
    np.savetxt(output_path, q_table, delimiter=",", fmt="%.6f")


def save_text(text: str, output_path: Path) -> None:
    """Write a text artifact to disk."""
    output_path.write_text(text, encoding="utf-8")


def save_reward_matrix_csv(
    reward_matrix: np.ndarray,
    agent_name: str,
    output_path: Path,
) -> None:
    """Save repeated-run reward data with one column per run."""
    episodes = np.arange(1, reward_matrix.shape[1] + 1, dtype=np.int32).reshape(-1, 1)
    stacked = np.hstack((episodes, reward_matrix.T))
    run_headers = [f"{agent_name}_run_{run_index + 1}" for run_index in range(reward_matrix.shape[0])]
    header = ",".join(["episode", *run_headers])
    fmt = ["%d"] + ["%.6f"] * reward_matrix.shape[0]
    np.savetxt(output_path, stacked, delimiter=",", fmt=fmt, header=header, comments="")


def save_stability_metrics_csv(
    q_learning_analysis: StabilityAnalysisResult,
    sarsa_analysis: StabilityAnalysisResult,
    output_path: Path,
) -> None:
    """Save mean reward and rolling std comparisons to one CSV file."""
    episodes = np.arange(1, len(q_learning_analysis.mean_rewards) + 1, dtype=np.int32)
    stacked = np.column_stack(
        (
            episodes,
            q_learning_analysis.mean_rewards,
            q_learning_analysis.moving_average_rewards,
            q_learning_analysis.mean_rolling_std,
            sarsa_analysis.mean_rewards,
            sarsa_analysis.moving_average_rewards,
            sarsa_analysis.mean_rolling_std,
        )
    )
    np.savetxt(
        output_path,
        stacked,
        delimiter=",",
        fmt=["%d", "%.6f", "%.6f", "%.6f", "%.6f", "%.6f", "%.6f"],
        header=(
            "episode,"
            "q_learning_mean_reward,q_learning_moving_average,q_learning_rolling_std,"
            "sarsa_mean_reward,sarsa_moving_average,sarsa_rolling_std"
        ),
        comments="",
    )


def save_exploration_summary_csv(
    q_learning_exploration: ExplorationExperimentResult,
    sarsa_exploration: ExplorationExperimentResult,
    output_path: Path,
) -> None:
    """Save epsilon sensitivity summary metrics for both algorithms."""
    rows = []
    for experiment in (q_learning_exploration, sarsa_exploration):
        for epsilon in sorted(experiment.by_epsilon):
            setting = experiment.by_epsilon[epsilon]
            rows.append(
                [
                    experiment.agent_name,
                    epsilon,
                    setting.stability.convergence_episode,
                    setting.stability.final_mean_reward,
                    setting.stability.final_mean_rolling_std,
                    setting.representative_result.greedy_total_reward,
                    setting.representative_result.path_risk_level,
                ]
            )

    np.savetxt(
        output_path,
        np.array(rows, dtype=object),
        delimiter=",",
        fmt=["%s", "%.2f", "%d", "%.6f", "%.6f", "%.6f", "%s"],
        header=(
            "agent,epsilon,convergence_episode,final_mean_reward,"
            "final_mean_rolling_std,greedy_total_reward,path_risk_level"
        ),
        comments="",
    )


def build_stability_report(
    q_learning_analysis: StabilityAnalysisResult,
    sarsa_analysis: StabilityAnalysisResult,
) -> str:
    """Summarize which method looks more stable and why."""
    if q_learning_analysis.final_mean_rolling_std < sarsa_analysis.final_mean_rolling_std:
        more_stable = "Q-learning"
    elif sarsa_analysis.final_mean_rolling_std < q_learning_analysis.final_mean_rolling_std:
        more_stable = "SARSA"
    else:
        more_stable = "Both methods appear similarly stable"

    return (
        "Stability analysis summary\n"
        "==========================\n\n"
        f"More stable method near the end of training: {more_stable}\n\n"
        "How stability is measured:\n"
        "- Mean reward across repeated runs reflects typical learning progress.\n"
        "- Rolling reward standard deviation reflects how strongly rewards fluctuate during training.\n\n"
        f"Q-learning final mean reward: {q_learning_analysis.final_mean_reward:.3f}\n"
        f"Q-learning final rolling std: {q_learning_analysis.final_mean_rolling_std:.3f}\n"
        f"SARSA final mean reward: {sarsa_analysis.final_mean_reward:.3f}\n"
        f"SARSA final rolling std: {sarsa_analysis.final_mean_rolling_std:.3f}\n\n"
        "Interpretation:\n"
        "SARSA usually reflects exploration more directly because its update target "
        "uses the actual next epsilon-greedy action. If exploration might step near "
        "the cliff and incur large penalties, SARSA learns to account for that risk "
        "during training, so it often adopts a more conservative route. Q-learning "
        "updates toward the greedy max next action instead, so it can look more "
        "optimistic about risky states and may prefer a shorter path closer to the cliff.\n"
    )


def build_exploration_report(
    q_learning_exploration: ExplorationExperimentResult,
    sarsa_exploration: ExplorationExperimentResult,
) -> str:
    """Create a report-ready draft about epsilon's effect on both algorithms."""
    lines = [
        "Exploration analysis draft",
        "==========================",
        "",
        "This experiment compares epsilon = 0.01, 0.10, and 0.20 under the same environment and learning-rate settings.",
        "A larger epsilon means the agent explores more often, so it sees more risky transitions near the cliff but also learns from a broader range of states.",
        "",
        "Key observations by algorithm:",
    ]

    for experiment in (q_learning_exploration, sarsa_exploration):
        lines.append(f"- {experiment.agent_name}:")
        for epsilon in sorted(experiment.by_epsilon):
            setting = experiment.by_epsilon[epsilon]
            lines.append(
                "  "
                f"epsilon={epsilon:.2f}, convergence episode={setting.stability.convergence_episode}, "
                f"final mean reward={setting.stability.final_mean_reward:.3f}, "
                f"path={setting.representative_result.path_risk_level}"
            )

    lines.extend(
        [
            "",
            "Report draft:",
            (
                "Increasing epsilon generally increases exploration, which can slow apparent convergence because the agent keeps taking non-greedy actions for longer. "
                "However, additional exploration can also prevent the agent from committing too early to a poor local behavior."
            ),
            (
                "For Q-learning, a higher epsilon mainly changes the data it collects, while the update target still assumes the best next action through max_a' Q(s', a'). "
                "Because of this optimistic target, Q-learning often prefers shorter routes close to the cliff even when exploration is still active."
            ),
            (
                "For SARSA, a higher epsilon has a more direct effect because the update target uses the actual next epsilon-greedy action. "
                "If exploratory moves can step into the cliff, SARSA learns that the nearby states are effectively more dangerous, so it often shifts toward a safer and more conservative final path."
            ),
            (
                "In practice, epsilon therefore affects not only final reward but also policy character: small epsilon values tend to favor faster exploitation, "
                "while larger epsilon values usually produce slower convergence and can encourage safer behavior, especially for SARSA."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def save_summary(
    config: ExperimentConfig,
    q_learning_result: TrainingResult,
    sarsa_result: TrainingResult,
    q_learning_analysis: StabilityAnalysisResult,
    sarsa_analysis: StabilityAnalysisResult,
) -> None:
    """Save a compact JSON summary for the final report."""
    summary = {
        "config": asdict(config),
        "q_learning": {
            "final_episode_reward": q_learning_result.rewards[-1],
            "greedy_total_reward": q_learning_result.greedy_total_reward,
            "greedy_path": [list(position) for position in q_learning_result.greedy_path],
            "path_risk_note": q_learning_result.path_risk_note,
            "stability_final_mean_reward": q_learning_analysis.final_mean_reward,
            "stability_final_rolling_std": q_learning_analysis.final_mean_rolling_std,
        },
        "sarsa": {
            "final_episode_reward": sarsa_result.rewards[-1],
            "greedy_total_reward": sarsa_result.greedy_total_reward,
            "greedy_path": [list(position) for position in sarsa_result.greedy_path],
            "path_risk_note": sarsa_result.path_risk_note,
            "stability_final_mean_reward": sarsa_analysis.final_mean_reward,
            "stability_final_rolling_std": sarsa_analysis.final_mean_rolling_std,
        },
    }
    (RESULTS_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )


def parse_args() -> ExperimentConfig:
    """Parse command-line arguments into an experiment configuration."""
    parser = argparse.ArgumentParser(description="Compare Q-learning and SARSA.")
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--cols", type=int, default=12)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--max-steps-per-episode", type=int, default=1000)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--moving-average-window", type=int, default=10)
    parser.add_argument("--rolling-std-window", type=int, default=10)
    parser.add_argument("--analysis-runs", type=int, default=10)
    args = parser.parse_args()
    return ExperimentConfig(
        rows=args.rows,
        cols=args.cols,
        episodes=args.episodes,
        max_steps_per_episode=args.max_steps_per_episode,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon=args.epsilon,
        seed=args.seed,
        moving_average_window=args.moving_average_window,
        rolling_std_window=args.rolling_std_window,
        analysis_runs=args.analysis_runs,
    )


def main() -> None:
    """Run the full comparison experiment and save outputs to results/."""
    config = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    q_learning_result = train_q_learning(config)
    sarsa_result = train_sarsa(config)
    q_learning_analysis = run_stability_analysis(train_q_learning, config, seed_offset=0)
    # Use the same seed schedule for both algorithms so the comparison is
    # paired under matching random conditions.
    sarsa_analysis = run_stability_analysis(train_sarsa, config, seed_offset=0)
    q_learning_exploration = run_exploration_experiment(
        train_fn=train_q_learning,
        agent_name="Q-learning",
        config=config,
        epsilon_values=config.exploration_epsilons,
        seed_offset=2000,
    )
    sarsa_exploration = run_exploration_experiment(
        train_fn=train_sarsa,
        agent_name="SARSA",
        config=config,
        epsilon_values=config.exploration_epsilons,
        seed_offset=2000,
    )

    save_rewards_csv(
        rewards=q_learning_result.rewards,
        agent_name="q_learning",
        output_path=RESULTS_DIR / "q_learning_rewards.csv",
    )
    save_rewards_csv(
        rewards=sarsa_result.rewards,
        agent_name="sarsa",
        output_path=RESULTS_DIR / "sarsa_rewards.csv",
    )
    env = CliffWalkingEnv(rows=config.rows, cols=config.cols)
    save_q_table(q_learning_result.q_table, RESULTS_DIR / "q_learning_q_table.csv")
    save_q_table(sarsa_result.q_table, RESULTS_DIR / "sarsa_q_table.csv")
    save_text(
        build_path_report("Q-learning", q_learning_result),
        RESULTS_DIR / "q_learning_path.txt",
    )
    save_text(
        build_path_report("SARSA", sarsa_result),
        RESULTS_DIR / "sarsa_path.txt",
    )
    save_text(q_learning_result.greedy_policy_text, RESULTS_DIR / "q_learning_policy.txt")
    save_text(sarsa_result.greedy_policy_text, RESULTS_DIR / "sarsa_policy.txt")
    save_reward_matrix_csv(
        reward_matrix=q_learning_analysis.reward_matrix,
        agent_name="q_learning",
        output_path=RESULTS_DIR / "q_learning_reward_runs.csv",
    )
    save_reward_matrix_csv(
        reward_matrix=sarsa_analysis.reward_matrix,
        agent_name="sarsa",
        output_path=RESULTS_DIR / "sarsa_reward_runs.csv",
    )
    save_stability_metrics_csv(
        q_learning_analysis=q_learning_analysis,
        sarsa_analysis=sarsa_analysis,
        output_path=RESULTS_DIR / "stability_metrics.csv",
    )
    save_text(
        build_stability_report(q_learning_analysis, sarsa_analysis),
        RESULTS_DIR / "stability_analysis.txt",
    )
    save_exploration_summary_csv(
        q_learning_exploration=q_learning_exploration,
        sarsa_exploration=sarsa_exploration,
        output_path=RESULTS_DIR / "exploration_summary.csv",
    )
    save_text(
        build_exploration_report(q_learning_exploration, sarsa_exploration),
        RESULTS_DIR / "exploration_analysis_draft.txt",
    )
    save_summary(
        config,
        q_learning_result,
        sarsa_result,
        q_learning_analysis,
        sarsa_analysis,
    )
    plot_path_map(
        env=env,
        path=q_learning_result.greedy_path,
        title="Q-learning Final Greedy Path",
        output_path=RESULTS_DIR / "q_learning_path.png",
    )
    plot_path_map(
        env=env,
        path=sarsa_result.greedy_path,
        title="SARSA Final Greedy Path",
        output_path=RESULTS_DIR / "sarsa_path.png",
    )

    plot_reward_curves(
        reward_by_agent={
            "Q-learning": np.array(q_learning_result.rewards, dtype=np.float64),
            "SARSA": np.array(sarsa_result.rewards, dtype=np.float64),
        },
        output_path=RESULTS_DIR / "reward_curve.png",
        moving_average_window=config.moving_average_window,
    )
    plot_stability_analysis(
        reward_matrix_by_agent={
            "Q-learning": q_learning_analysis.reward_matrix,
            "SARSA": sarsa_analysis.reward_matrix,
        },
        output_path=RESULTS_DIR / "stability_analysis.png",
        moving_average_window=config.moving_average_window,
        rolling_std_window=config.rolling_std_window,
    )
    plot_exploration_analysis(
        reward_matrix_by_epsilon={
            epsilon: setting.stability.reward_matrix
            for epsilon, setting in q_learning_exploration.by_epsilon.items()
        },
        agent_name="Q-learning",
        output_path=RESULTS_DIR / "q_learning_exploration.png",
        moving_average_window=config.moving_average_window,
    )
    plot_exploration_analysis(
        reward_matrix_by_epsilon={
            epsilon: setting.stability.reward_matrix
            for epsilon, setting in sarsa_exploration.by_epsilon.items()
        },
        agent_name="SARSA",
        output_path=RESULTS_DIR / "sarsa_exploration.png",
        moving_average_window=config.moving_average_window,
    )
    plot_exploration_summary(
        epsilon_values=sorted(config.exploration_epsilons),
        q_learning_final_rewards=[
            q_learning_exploration.by_epsilon[epsilon].stability.final_mean_reward
            for epsilon in sorted(config.exploration_epsilons)
        ],
        sarsa_final_rewards=[
            sarsa_exploration.by_epsilon[epsilon].stability.final_mean_reward
            for epsilon in sorted(config.exploration_epsilons)
        ],
        q_learning_convergence=[
            q_learning_exploration.by_epsilon[epsilon].stability.convergence_episode
            for epsilon in sorted(config.exploration_epsilons)
        ],
        sarsa_convergence=[
            sarsa_exploration.by_epsilon[epsilon].stability.convergence_episode
            for epsilon in sorted(config.exploration_epsilons)
        ],
        output_path=RESULTS_DIR / "exploration_summary.png",
    )

    print(f"Results saved to: {RESULTS_DIR.resolve()}")


if __name__ == "__main__":
    main()
