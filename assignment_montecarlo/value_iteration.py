from typing import Tuple, Optional

import numpy as np
import pandas as pd

from assignment_montecarlo.environment_world import EnvironmentWorld, State, Action


class ValueIteration:
    def __init__(self, mdp: EnvironmentWorld, discount: float = 0.9, iterations: int = 100):
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = [[mdp.get_reward((x, y)) for y in range(mdp.num_rows)] for x in range(mdp.num_cols)]

    def __repr__(self):
        values = pd.DataFrame(self.values)
        return values.T.__repr__()

    def __str__(self):
        return self.__repr__()

    def run_value_iteration(self, iterations=None):
        iterations = iterations or self.iterations
        for i in range(iterations):
            new_values = [[0 for _ in range(self.mdp.num_rows)] for __ in range(self.mdp.num_cols)]
            for x in range(self.mdp.num_cols):
                for y in range(self.mdp.num_rows):
                    new_values[x][y] = self.compute_action_from_values((x, y))[1]
            self.values = new_values


    def get_value(self, state: State) -> float:
        x, y = state
        return self.values[x][y]

    def compute_qvalue_from_values(self, state: State, action: Action) -> float:
        return sum(
            self.mdp.probability(state, next_state, action) * (
                    self.mdp.get_reward(next_state) + self.discount * self.get_value(next_state)
            )
            for next_state in self.mdp.get_possible_next_state(state)
        )

    def compute_action_from_values(self, state: State) -> Tuple[Action, float]:
        action, value = max((
            (action, self.compute_qvalue_from_values(state, Action(action)))
            for action in self.mdp.get_state_possible_actions(state)
        ), key=lambda pair: pair[1], default=(None, 0))
        return action, value

    def get_action(self, state: State) -> Action:
        return self.compute_action_from_values(state)[0]

    def get_qvalue(self, state: State, action: Action) -> float:
        return self.compute_qvalue_from_values(state, action)

    def get_policy(self, state: State) -> Optional[Action]:
        if np.isnan(self.get_value(state)):
            return None
        return self.get_action(state)

    def get_full_policy(self):
        return pd.DataFrame([
            [self.get_policy((x, y)) for x in range(self.mdp.num_cols)]
            for y in range(self.mdp.num_rows)
        ])
