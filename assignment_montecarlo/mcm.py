import time
from random import choice
from typing import List

import numpy as np
from collections import defaultdict

import pandas as pd
from tqdm import tqdm

from assignment_montecarlo.environment_world import EnvironmentWorld, Action, State

Episode = List[tuple[State, Action, float]]


class MonteCarloAgent:
    def __init__(self, world: EnvironmentWorld, discount_factor: float = 1.0, initial_epsilon: float = 0.1,
                 epsilon_decay: float = 0.999):
        self.world = world
        self.discount_factor = discount_factor
        self.initial_epsilon = initial_epsilon
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.Q = defaultdict(lambda: defaultdict(float))  # Q[state][action]
        self.episode_count = defaultdict(lambda: defaultdict(int))  # number of observations for [state][action]
        self.policy = defaultdict(lambda: defaultdict(float))  # Policy[state][action]

    def epsilon_greedy_policy(self, state: State) -> Action:
        actions = self.world.get_state_possible_actions(state)
        if np.random.rand() < self.epsilon:
            return choice(actions)
        else:
            q_values = [self.Q[state][a] for a in actions]
            best_action = actions[np.argmax(q_values)]
            return best_action

    def generate_episode(self) -> Episode:
        episode = []
        current_state = self.world.reset()  # Start with an initial state
        done = False

        while not done:
            action = self.epsilon_greedy_policy(current_state)
            reward, next_state = self.world.do_action(action)
            episode.append((current_state, action, reward))
            current_state = next_state
            done = self.world.is_terminal()

        return episode

    def update_q(self, episode: Episode):
        step_reward = 0
        visited_state_actions = set()

        for i, (state, action, reward) in enumerate(reversed(episode)):
            step_reward = reward + self.discount_factor * step_reward

            self.Q[state][action] = (self.Q[state][action] * self.episode_count[state][action] + step_reward) / (
                    self.episode_count[state][action] + 1)
            self.episode_count[state][action] += 1
            visited_state_actions.add((state, action))

    def learn(self, max_episodes: int, convergence_check_frequency: int = 100, convergence_patience: int = 3):
        self.epsilon = self.initial_epsilon
        old_policy = self.policy.copy()
        for i in tqdm(range(max_episodes), desc='Episodes'):
            check_for_convergence = i % convergence_check_frequency == 0 and i
            episode = self.generate_episode()
            self.update_q(episode)
            self.improve_policy()
            if check_for_convergence and old_policy == self.policy:
                convergence_patience -= 1
                if convergence_patience <= 0:
                    print('Early stop due to convergence.')
                    return
                old_policy = self.policy.copy()
            self.epsilon *= self.epsilon_decay

    def improve_policy(self):
        for state in self.Q:
            actions = list(self.Q[state].keys())
            best_action = max(actions, key=lambda a: self.Q[state][a])

            self.policy[state] = defaultdict(
                int,
                {action: 1.0 if action == best_action else 0.0 for action in actions}
            )

    def print_policy(self):
        policy_matrix = [[None for y in range(self.world.num_rows)] for x in range(self.world.num_cols)]
        for y in range(self.world.num_rows):
            for x in range(self.world.num_cols):
                for action in self.world.ACTIONS:
                    if self.policy[(x, y)][action]:
                        policy_matrix[x][y] = action.value
                        break

        print(pd.DataFrame(policy_matrix).transpose())