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
    ACTIONS = [action for action in Action]

    def __init__(self, board: List[List[str]], noise=0.25):
        self.board = pd.DataFrame(board)
        self.num_rows = len(board)
        self.num_cols = len(board[0])
        self.initial_state = (0, 0)
        self.noise = noise
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

    @classmethod
    def get_possible_actions(cls, _state: Tuple[int, int]) -> List[Action]:
        return cls.ACTIONS

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
        if np.random.rand() < self.noise:
            action = np.random.choice(self.ACTIONS)
        self.current_state = self.get_next_state(self.current_state, action)
        return self.get_reward(self.current_state), self.current_state

    def reset(self):
        self.current_state = self.initial_state
        return self.current_state

    def is_terminal(self) -> bool:
        return self.get_reward(self.current_state) != 0

    def get_possible_next_state(self, state: State)-> Set[State]:
        possibilities = {self.get_next_state(state, action) for action in Action}
        possibilities.add(state)
        return possibilities
