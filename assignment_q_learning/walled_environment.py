from abc import ABC
from collections import defaultdict
from enum import Enum
from typing import Tuple, List, Set

import numpy as np
import pandas as pd
from numpy.random import random, randint


# %%

class Action(str, Enum):
    UP = 'up'
    RIGHT = 'right'
    DOWN = 'down'
    LEFT = 'left'


# %%

State = Tuple[int, int]

class WalledEnvironmentWorld:
    ACTIONS = [Action(action) for action in Action]

    def __init__(self, board: List[List[str]], walls: List[Tuple[State, State]], terminal_states: List[State] = None,
                 action_noise=None,
                 initial_state=None):
        self.action_noise = action_noise or defaultdict(float)
        self.walls = walls
        self.board = pd.DataFrame(board)
        self.num_rows = len(board)
        self.num_cols = len(board[0])
        self.initial_state = initial_state
        self.terminal_states = terminal_states or []
        self.current_state = None
        self.reset()

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
        next_state = (x_, y_)
        if (state, next_state) in self.walls or (next_state, state) in self.walls:
            next_state = state
        return next_state

    def do_action(self, action: Action) -> Tuple[float, State]:
        noise = self.action_noise[action]
        if np.random.rand() < noise:
            action = np.random.choice(self.get_possible_actions())
        self.current_state = self.get_next_state(self.current_state, action)
        return self.get_reward(self.current_state), self.current_state

    def reset(self):
        if self.initial_state is None:
            self.current_state = (randint(0, self.num_cols - 1), randint(0, self.num_rows - 1))
        else:
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
