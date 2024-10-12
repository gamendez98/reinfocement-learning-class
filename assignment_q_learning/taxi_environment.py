from collections import defaultdict
from enum import Enum
from random import choice
from typing import Tuple, List, Set, NamedTuple, Optional

import numpy as np
import pandas as pd
from numpy.random import randint


# %%

class TaxiAction(str, Enum):
    UP = 'up'
    RIGHT = 'right'
    DOWN = 'down'
    LEFT = 'left'
    PICKUP = 'pickup'
    DROPOFF = 'dropoff'


# %%
Position = Tuple[int, int]


class Station(NamedTuple):
    position: Position
    name: str


class TaxiState(NamedTuple):
    position: Position
    passenger_station: Optional[Station]
    on_passenger: bool = False


class TaxiEnvironmentWorld:
    ACTIONS = [TaxiAction(action) for action in TaxiAction]
    PASSENGER_ACTIONS = [TaxiAction.PICKUP, TaxiAction.DROPOFF]

    def __init__(self, board: List[List[str]], walls: List[Tuple[Position, Position]], stations: List[Station],
                 terminal_states: List[TaxiState] = None, action_noise=None, initial_state=None,
                 wrong_action_cost: float = -10, drop_off_reward: float = 5, pick_up_reward: float = 1):
        self.action_noise = action_noise or defaultdict(float)
        self.stations = stations
        self.wrong_action_cost = wrong_action_cost
        self.drop_off_reward = drop_off_reward
        self.pick_up_reward = pick_up_reward
        self.walls = walls
        self.passenger: Optional[TaxiState] = None  # position and destiny
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

    def get_current_state(self) -> TaxiState:
        return self.current_state

    def get_state_possible_actions(self, state: TaxiState) -> List[TaxiAction]:
        if self.state_is_terminal(state):
            return []
        return self.ACTIONS

    def get_possible_actions(self) -> List[TaxiAction]:
        return self.get_state_possible_actions(self.current_state)

    def get_reward(self, state: TaxiState, next_state: TaxiState, action: TaxiAction) -> float:
        passenger_station = state.passenger_station
        position_, passenger_station_ = next_state.position, next_state.passenger_station
        x_, y_ = position_
        if action in self.PASSENGER_ACTIONS:
            if passenger_station and not passenger_station_:  # drop off a passenger
                return self.drop_off_reward
            if passenger_station_ and not passenger_station:  # pick up a passenger
                return self.pick_up_reward
            return self.wrong_action_cost  # passenger action failed
        board_value = self.board.loc[y_, x_]
        try:
            return float(board_value)
        except ValueError:
            return 0

    def get_next_state(self, state: TaxiState, action: TaxiAction) -> TaxiState:

        if action == TaxiAction.PICKUP:
            return self.map_pick_up_action(state)
        if action == TaxiAction.DROPOFF:
            return self.map_drop_off_action(state)

        x, y = state.position
        x_, y_ = x, y

        if action == TaxiAction.UP:
            y_ = max(0, y - 1)
        elif action == TaxiAction.DOWN:
            y_ = min(self.num_rows - 1, y + 1)
        elif action == TaxiAction.LEFT:
            x_ = max(0, x - 1)
        elif action == TaxiAction.RIGHT:
            x_ = min(self.num_cols - 1, x + 1)
        next_position = (x_, y_)
        if (state.position, next_position) in self.walls or (next_position, state.position) in self.walls:
            next_position = (x, y)
        return TaxiState(next_position, state.passenger_station, on_passenger=self.on_passenger(next_position))

    def map_pick_up_action(self, state: TaxiState) -> TaxiState:
        if state.passenger_station:  # already has a passenger
            return state
        if not self.passenger:
            return state
        if state.position != self.passenger.position:  # wrong position
            return state
        return TaxiState(state.position, self.passenger.passenger_station)

    @staticmethod
    def map_drop_off_action(state: TaxiState) -> TaxiState:
        if not state.passenger_station:  # no passenger
            return state
        if not state.passenger_station:
            return state
        if state.position != state.passenger_station.position:  # wrong drop off place
            return state
        return TaxiState(state.position, None)

    def on_passenger(self, position: Position) -> bool:
        if not self.passenger:
            return False
        return position == self.passenger.position

    def do_action(self, action: TaxiAction) -> Tuple[float, TaxiState]:
        noise = self.action_noise[action]
        if np.random.rand() < noise:
            action = np.random.choice(self.get_possible_actions())
        next_state = self.get_next_state(self.current_state, action)
        reward = self.get_reward(self.current_state, next_state, action)
        if action == TaxiAction.DROPOFF and next_state != self.current_state:  # successful drop off
            self.reset_passenger()
        if action == TaxiAction.PICKUP and next_state != self.current_state:  # pick up
            self.passenger = None
        self.current_state = next_state
        return reward, self.current_state

    def random_position(self) -> Position:
        return randint(0, self.num_cols - 1), randint(0, self.num_rows - 1)

    def random_station(self) -> Station:
        return choice(self.stations)

    def reset_passenger(self):
        origin_station = self.random_station()
        self.passenger = TaxiState(origin_station.position, self.random_station())

    def reset(self):
        self.current_state = TaxiState(self.random_position(), None)
        self.reset_passenger()
        return self.current_state

    def is_terminal(self) -> bool:
        return self.state_is_terminal(self.current_state)

    def state_is_terminal(self, state: TaxiState) -> bool:
        return False

    def get_possible_next_state(self, state: TaxiState) -> Set[TaxiState]:
        possibilities = {self.get_next_state(state, TaxiAction(action)) for action in TaxiAction}
        possibilities.add(state)
        return possibilities
