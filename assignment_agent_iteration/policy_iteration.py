from typing import Tuple, Optional

import pandas as pd

from assignment_agent_iteration.grid_world import EnvironmentWorld, State, Action


class PolicyIteration:
    def __init__(self, mdp: EnvironmentWorld, discount: float = 0.9, iteration_limit: int = 100):
        self.mdp = mdp
        self.discount = discount
        self.iteration_limit = iteration_limit
        self.values = [[mdp.get_reward((x, y)) for y in range(mdp.num_rows)] for x in range(mdp.num_cols)]
        self.policy = [[self.get_action((x, y)) for y in range(mdp.num_rows)] for x in range(mdp.num_cols)]
        self.iterations_to_converge = None

    def __repr__(self):
        values = pd.DataFrame(self.values)
        return values.T.__repr__()

    def __str__(self):
        return self.__repr__()

    def run_policy_iteration(self) -> int:
        for i in range(self.iteration_limit):
            self.policy_evaluation()
            if not self.policy_iteration():
                self.iterations_to_converge = i + 1
                return i + 1

    def policy_evaluation(self):
        new_values = [[0 for _ in range(self.mdp.num_rows)] for __ in range(self.mdp.num_cols)]
        for x in range(self.mdp.num_cols):
            for y in range(self.mdp.num_rows):
                new_values[x][y] = self.compute_qvalue_from_values((x, y), self.get_policy((x, y)))
        self.values = new_values

    def policy_iteration(self) -> bool:
        new_policy = [[0 for _ in range(self.mdp.num_rows)] for __ in range(self.mdp.num_cols)]
        for x in range(self.mdp.num_cols):
            for y in range(self.mdp.num_rows):
                new_policy[x][y] = self.get_action((x, y))
        policy_changed = self.policy != new_policy
        self.policy = new_policy
        return policy_changed

    def get_value(self, state: State) -> float:
        x, y = state
        return self.values[x][y]

    def compute_qvalue_from_values(self, state: State, action) -> float:
        return sum(
            self.mdp.probability(state, next_state, action) * (
                    self.mdp.get_reward(next_state) + self.discount * self.get_value(next_state)
            )
            for next_state in self.mdp.get_possible_next_state(state)
        )

    def compute_action_from_values(self, state: State) -> Tuple[Action, float]:
        return max((
            (action, self.compute_qvalue_from_values(state, action))
            for action in Action
        ), key=lambda pair: pair[1])

    def get_action(self, state: State) -> Action:
        return self.compute_action_from_values(state)[0]

    def get_qvalue(self, state: State, action: Action) -> float:
        return self.compute_qvalue_from_values(state, action)

    def get_policy(self, state: State) -> Optional[Action]:
        return self.policy[state[0]][state[1]]

    def get_full_policy(self):
        return pd.DataFrame(self.policy)
