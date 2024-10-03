from collections import defaultdict
from typing import Callable

import pandas as pd
from tqdm import tqdm

from assignment_td_sarsa.environment_world import EnvironmentWorld, State, Action


class TDLearning:

    def __init__(self, world: EnvironmentWorld, policy:Callable[[State], Action], learning_rate: float, discount_factor: float):
        self.world = world
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.policy = policy # policy(state) -> a
        self.V = defaultdict(float)  # Value[State]

    def iterate_learning(self, num_steps: int):
        for _ in tqdm(range(num_steps)):
            self.run_step()
            if self.world.is_terminal():
                self.world.reset()

    def run_step(self):
        state = self.world.current_state
        action = self.policy(state)
        reward, new_state = self.world.do_action(action)
        self.update(state, new_state, reward)

    def update(self, state: State, new_state: State, reward: float):
        td_error = reward + self.discount_factor * self.V[new_state] - self.V[state]
        self.V[state] += self.learning_rate * td_error

    def print_values(self):
        value_matrix = [[None for y_ in range(self.world.num_rows)] for x in range(self.world.num_cols)]
        for y in range(self.world.num_rows):
            for x in range(self.world.num_cols):
                value = self.V[(x, y)]
                if value is not None:
                    value_matrix[x][y] = value

        print(pd.DataFrame(value_matrix).transpose())
