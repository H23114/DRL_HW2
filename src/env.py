"""Environment definitions for tabular reinforcement learning experiments.

This module implements a tabular Cliff Walking gridworld that is intentionally
simple and explicit. The code is written for homework use, so the environment
logic, state representation, and visualization helpers are easy to inspect.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np


Action = int
State = int
Position = Tuple[int, int]


@dataclass(frozen=True)
class StepResult:
    """Container for a single environment transition."""

    next_state: State
    reward: float
    done: bool


class CliffWalkingEnv:
    """A small gridworld with a cliff between start and goal.

    Grid layout:
    - Start is at bottom-left.
    - Goal is at bottom-right.
    - Cliff cells are the bottom-row cells between start and goal.

    Rewards:
    - Normal step: -1
    - Falling off cliff: -100 and reset to start
    - Reaching goal: -1 and episode terminates
    """

    ACTION_UP = 0
    ACTION_RIGHT = 1
    ACTION_DOWN = 2
    ACTION_LEFT = 3

    def __init__(self, rows: int = 4, cols: int = 12) -> None:
        self.rows = rows
        self.cols = cols
        self.start_pos: Position = (rows - 1, 0)
        self.goal_pos: Position = (rows - 1, cols - 1)
        self.agent_pos = self.start_pos

        # Action encoding:
        # 0 = up, 1 = right, 2 = down, 3 = left.
        self.action_map: Dict[Action, Tuple[int, int]] = {
            self.ACTION_UP: (-1, 0),
            self.ACTION_RIGHT: (0, 1),
            self.ACTION_DOWN: (1, 0),
            self.ACTION_LEFT: (0, -1),
        }

        self.n_actions = len(self.action_map)
        self.n_states = self.rows * self.cols

    def reset(self) -> State:
        """Reset the environment and return the initial state."""
        self.agent_pos = self.start_pos
        return self.state_to_index(self.agent_pos)

    def step(self, action: Action) -> StepResult:
        """Apply one action and return the transition result."""
        if action not in self.action_map:
            raise ValueError(f"Invalid action: {action}")

        delta_row, delta_col = self.action_map[action]
        row, col = self.agent_pos

        next_row = np.clip(row + delta_row, 0, self.rows - 1)
        next_col = np.clip(col + delta_col, 0, self.cols - 1)
        next_pos = (int(next_row), int(next_col))

        # Falling into the cliff gives a large negative reward and returns the
        # agent to the start state without ending the episode.
        if self._is_cliff(next_pos):
            self.agent_pos = self.start_pos
            return StepResult(
                next_state=self.state_to_index(self.start_pos),
                reward=-100.0,
                done=False,
            )

        self.agent_pos = next_pos
        done = self.is_terminal(next_pos)
        reward = -1.0
        return StepResult(
            next_state=self.state_to_index(next_pos),
            reward=reward,
            done=done,
        )

    def state_to_index(self, position: Position) -> State:
        """Convert a grid position into a tabular state index."""
        row, col = position
        if not self._is_inside_grid(position):
            raise ValueError(f"Position out of range: {position}")
        return row * self.cols + col

    def index_to_state(self, index: State) -> Position:
        """Convert a tabular state index back into a grid position."""
        if not 0 <= index < self.n_states:
            raise ValueError(f"State index out of range: {index}")
        return divmod(index, self.cols)

    def is_terminal(self, position: Position) -> bool:
        """Return True only when the position is the goal state."""
        return position == self.goal_pos

    def is_cliff(self, position: Position) -> bool:
        """Public helper for checking whether a cell belongs to the cliff."""
        return self._is_cliff(position)

    def render_policy(self, q_table: np.ndarray) -> str:
        """Return a text visualization of the greedy policy."""
        arrows = {
            self.ACTION_UP: "^",
            self.ACTION_RIGHT: ">",
            self.ACTION_DOWN: "v",
            self.ACTION_LEFT: "<",
        }
        rows = []

        for row in range(self.rows):
            symbols = []
            for col in range(self.cols):
                pos = (row, col)
                if pos == self.start_pos:
                    symbols.append("S")
                elif pos == self.goal_pos:
                    symbols.append("G")
                elif self._is_cliff(pos):
                    symbols.append("C")
                else:
                    state = self.state_to_index(pos)
                    best_action = int(np.argmax(q_table[state]))
                    symbols.append(arrows[best_action])
            rows.append(" ".join(symbols))

        return "\n".join(rows)

    def render_path(
        self,
        path: Iterable[Position | State],
        mark_path: str = "*",
    ) -> str:
        """Return a text grid showing one episode path.

        The path can contain either state indices or (row, col) positions.
        Start, goal, and cliff cells keep their own symbols to remain readable.
        """
        normalized_path: List[Position] = []
        for item in path:
            if isinstance(item, tuple):
                position = item
            else:
                position = self.index_to_state(int(item))
            normalized_path.append(position)

        path_cells = set(normalized_path)
        rows = []
        for row in range(self.rows):
            symbols = []
            for col in range(self.cols):
                pos = (row, col)
                if pos == self.start_pos:
                    symbols.append("S")
                elif pos == self.goal_pos:
                    symbols.append("G")
                elif self._is_cliff(pos):
                    symbols.append("C")
                elif pos in path_cells:
                    symbols.append(mark_path)
                else:
                    symbols.append(".")
            rows.append(" ".join(symbols))
        return "\n".join(rows)

    def rollout_greedy_path(
        self,
        q_table: np.ndarray,
        max_steps: int = 100,
    ) -> List[Position]:
        """Generate a greedy path from the start state using a learned Q-table."""
        state = self.reset()
        path = [self.index_to_state(state)]

        for _ in range(max_steps):
            action = int(np.argmax(q_table[state]))
            result = self.step(action)
            state = result.next_state
            path.append(self.index_to_state(state))
            if result.done:
                break

        return path

    def _is_cliff(self, position: Position) -> bool:
        """Check whether a grid cell belongs to the cliff."""
        row, col = position
        return row == self.rows - 1 and 0 < col < self.cols - 1

    def _is_inside_grid(self, position: Position) -> bool:
        """Check whether a position is inside the map boundary."""
        row, col = position
        return 0 <= row < self.rows and 0 <= col < self.cols
