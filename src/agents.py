"""Tabular RL agents for Cliff Walking experiments.

This module keeps the shared tabular logic in one base class and lets each
algorithm override only its TD target. That makes the difference between
Q-learning and SARSA easy to explain in a homework report:

- Q-learning is off-policy: it updates toward the best possible next action.
- SARSA is on-policy: it updates toward the next action actually selected by
  the current epsilon-greedy policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class AgentConfig:
    """Hyperparameters shared by both tabular agents."""

    n_states: int
    n_actions: int
    alpha: float = 0.5
    gamma: float = 1.0
    epsilon: float = 0.1
    seed: Optional[int] = None


class BaseAgent:
    """Base class for tabular control agents.

    Shared responsibilities:
    - store the Q-table
    - implement epsilon-greedy action selection
    - provide common TD-error update mechanics
    """

    def __init__(self, config: AgentConfig) -> None:
        self.n_states = config.n_states
        self.n_actions = config.n_actions
        self.alpha = config.alpha
        self.gamma = config.gamma
        self.epsilon = config.epsilon
        self.rng = np.random.default_rng(config.seed)

        # Q-table[state, action] stores the current action-value estimate.
        self.q_table = np.zeros((self.n_states, self.n_actions), dtype=np.float64)

    def select_action(self, state: int) -> int:
        """Select an action using epsilon-greedy exploration."""
        self._validate_state(state)
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))

        # Random tie-breaking avoids a fixed directional bias at initialization.
        return self.greedy_action(state)

    def greedy_action(self, state: int) -> int:
        """Select the greedy action without exploration."""
        self._validate_state(state)
        best_value = np.max(self.q_table[state])
        best_actions = np.flatnonzero(np.isclose(self.q_table[state], best_value))
        return int(self.rng.choice(best_actions))

    def _apply_td_update(
        self,
        state: int,
        action: int,
        td_target: float,
    ) -> None:
        """Apply the standard one-step TD update to Q(s, a)."""
        self._validate_state(state)
        self._validate_action(action)
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.alpha * td_error

    def _validate_state(self, state: int) -> None:
        """Check whether a state index is valid for this Q-table."""
        if not 0 <= state < self.n_states:
            raise ValueError(f"State index out of range: {state}")

    def _validate_action(self, action: int) -> None:
        """Check whether an action index is valid for this agent."""
        if not 0 <= action < self.n_actions:
            raise ValueError(f"Action index out of range: {action}")

    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool,
        next_action: Optional[int] = None,
    ) -> None:
        """Update the Q-table. Implemented by subclasses."""
        raise NotImplementedError


class QLearningAgent(BaseAgent):
    """Off-policy TD control using the Q-learning update rule.

    Q-learning uses the maximum next-state action value as its TD target:
        r + gamma * max_a' Q(s', a')

    This is called off-policy because the update does not depend on the actual
    action the behavior policy will take next.
    """

    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool,
        next_action: Optional[int] = None,
    ) -> None:
        best_next_q = 0.0 if done else np.max(self.q_table[next_state])
        td_target = reward + self.gamma * best_next_q
        self._apply_td_update(state, action, td_target)


class SARSAAgent(BaseAgent):
    """On-policy TD control using the SARSA update rule.

    SARSA uses the next action chosen by the current policy as its TD target:
        r + gamma * Q(s', a')

    This is called on-policy because the update follows the same epsilon-greedy
    policy that is being used to interact with the environment.
    """

    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool,
        next_action: Optional[int] = None,
    ) -> None:
        if not done and next_action is None:
            raise ValueError("SARSA update requires next_action when done is False.")
        if next_action is not None:
            self._validate_action(next_action)

        next_q = 0.0 if done else self.q_table[next_state, next_action]
        td_target = reward + self.gamma * next_q
        self._apply_td_update(state, action, td_target)


# Backward-compatible alias so existing imports continue to work.
SarsaAgent = SARSAAgent
