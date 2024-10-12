

from collections import defaultdict
from random import choice
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from assignment_td_sarsa.environment_world import EnvironmentWorld, State, Action


class QLearningAgent:

    def __init__(self, world: EnvironmentWorld, learning_rate: float, discount_factor: float, epsilon: float,
                 learning_rate_decay: float = 1.0):
        self.world = world
        self.initial_learning_rate = learning_rate
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.Q = defaultdict(lambda: defaultdict(float))  # Q[State][Action]

    def get_action(self, state: State) -> Optional[Action]:
        if np.random.rand() < self.epsilon:
            possible_actions = self.world.get_state_possible_actions(state)
            if not possible_actions:
                return None
            return choice(possible_actions)
        return self.get_opt_action(state)

    def get_opt_action(self, state: State) -> Optional[Action]:
        possible_actions = self.world.get_state_possible_actions(state)
        return max(
            possible_actions,
            key=lambda action: self.Q[state][action],
            default=None
        )

    def iterate_learning(self, num_steps: int, reset_learning_rate: bool = True):
        self.world.reset()
        if reset_learning_rate:
            self.learning_rate = self.initial_learning_rate
        for _ in tqdm(range(num_steps)):
            self.run_step()
            if self.world.is_terminal():
                self.world.reset()
            self.learning_rate *= self.learning_rate_decay

    def run_step(self):
        state = self.world.current_state
        action = self.get_action(state)
        reward, new_state = self.world.do_action(action)
        self.update_q(state, action, new_state, reward)

    def update_q(self, state: State, action: Action, next_state: State, reward: float):
        next_action = self.get_opt_action(next_state)
        td_error = reward + self.discount_factor * self.Q[next_state][next_action] - self.Q[state][action]
        self.Q[state][action] += self.learning_rate * td_error

    def print_policy(self):
        policy_matrix = [[None for y_ in range(self.world.num_rows)] for x in range(self.world.num_cols)]
        for y in range(self.world.num_rows):
            for x in range(self.world.num_cols):
                action = self.get_opt_action((x, y))
                if action is not None:
                    policy_matrix[x][y] = action.value

        print(pd.DataFrame(policy_matrix).transpose())
