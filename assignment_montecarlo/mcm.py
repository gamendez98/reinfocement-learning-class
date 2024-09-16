from random import choice
from typing import List

import numpy as np
from collections import defaultdict

from assignment_montecarlo.environment_world import EnvironmentWorld, Action, State

Episode = List[(State, Action, float)]

class MonteCarloAgent:
    def __init__(self, world: EnvironmentWorld, discount_factor:float=1.0, epsilon:float=0.1):
        self.world = world
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.Q = defaultdict(lambda: defaultdict(float))  # Q[state][action]
        self.returns = defaultdict(lambda: defaultdict(list))  # Returns[state][action]
        self.policy = defaultdict(lambda: defaultdict(float))  # Policy[state][action]

    def epsilon_greedy_policy(self, state: State) -> Action:
        actions = self.world.get_possible_actions(state)
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

        for state, action, reward in reversed(episode):
            step_reward = reward + self.discount_factor * step_reward

            self.returns[state][action].append(step_reward)
            self.Q[state][action] = np.mean(self.returns[state][action]) # type: ignore[assignment]
            visited_state_actions.add((state, action))

    def learn(self, num_episodes: int):
        for episode_num in range(num_episodes):
            episode = self.generate_episode()
            self.update_q(episode)
            self.improve_policy()

    def improve_policy(self):
        for state in self.Q:
            actions = list(self.Q[state].keys())
            best_action = max(actions, key=lambda a: self.Q[state][a])

            self.policy[state] = defaultdict(
                int,
                {action: 1.0 if action == best_action else 0.0 for action in actions}
            )
