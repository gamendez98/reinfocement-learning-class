from collections import defaultdict
from enum import Enum
from typing import Tuple, List, Set

import numpy as np
import pandas as pd


# %%

class Action(str, Enum):
    UP = 'up'
    RIGHT = 'right'
    DOWN = 'down'
    LEFT = 'left'


# %%

State = Tuple[int, int]


class EnvironmentWorld:
    ACTIONS = [Action(action) for action in Action]

    def __init__(self, board: List[List[str]], terminal_states: List[State] = None, action_noise=None):
        self.action_noise = action_noise or defaultdict(float)
        self.board = pd.DataFrame(board)
        self.num_rows = len(board)
        self.num_cols = len(board[0])
        self.initial_state = (0, 0)
        self.terminal_states = terminal_states or []
        for i, row in self.board.iterrows():
            for j, cell in enumerate(row):
                if cell == 'S':
                    self.initial_state = (j, i)
        self.current_state = self.initial_state

    def __repr__(self):
        board = self.board.copy()
        x, y = self.current_state
        board.loc[y, x] += 'C'
        return board.__repr__()

    def __str__(self):
        return self.__repr__()

    def get_current_state(self) -> State:
        return self.current_state

    def get_state_possible_actions(self, state: State) -> List[Action]:
        if self.state_is_terminal(state):
            return []
        return self.ACTIONS

    def get_possible_actions(self) -> List[Action]:
        return self.get_state_possible_actions(self.current_state)

    def get_reward(self, state: State) -> float:
        x, y = state
        board_value = self.board.loc[y, x]
        if board_value == '#':
            return np.nan
        try:
            return float(board_value)
        except ValueError:
            return 0

    def get_next_state(self, state: State, action: Action) -> State:
        x, y = state
        x_, y_ = x, y
        if action == Action.UP:
            y_ = max(0, y - 1)
        elif action == Action.DOWN:
            y_ = min(self.num_rows - 1, y + 1)
        elif action == Action.LEFT:
            x_ = max(0, x - 1)
        elif action == Action.RIGHT:
            x_ = min(self.num_cols - 1, x + 1)
        if self.board.loc[y_, x_] != '#':
            return x_, y_
        return x, y

    def do_action(self, action: Action) -> Tuple[float, State]:
        noise = self.action_noise[action]
        if np.random.rand() < noise:
            action = np.random.choice(self.get_possible_actions())
        self.current_state = self.get_next_state(self.current_state, action)
        return self.get_reward(self.current_state), self.current_state

    def reset(self):
        self.current_state = self.initial_state
        return self.current_state

    def is_terminal(self) -> bool:
        return self.state_is_terminal(self.current_state)

    def state_is_terminal(self, state: State) -> bool:
        return state in self.terminal_states

    def get_possible_next_state(self, state: State) -> Set[State]:
        possibilities = {self.get_next_state(state, Action(action)) for action in Action}
        possibilities.add(state)
        return possibilities

    def probability(self, current_state: State, next_state: State, action: Action) -> float:
        intended_next_state = self.get_next_state(current_state, action)
        unintended_next_states = [self.get_next_state(current_state, noise_action) for noise_action in
                                  self.get_state_possible_actions(current_state)]
        unintended_probability = sum(
            0.25 * (unintended_state == next_state) for unintended_state in unintended_next_states) * self.noise
        probs = (1 - self.noise) * (intended_next_state == next_state) + unintended_probability
        return probs
