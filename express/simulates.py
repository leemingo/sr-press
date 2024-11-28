"""Pressing Simulation Module."""

from abc import ABC, abstractmethod
from functools import reduce
from typing import Callable, List, Optional, Dict, Tuple
import copy
import itertools

import numpy as np
import pandas as pd
from tqdm import tqdm

import express.config as config
from express.features import gamestates
from express.utils import add_names, play_left_to_right

class SimulatedInstance:
    def __init__(
        self,
        state: pd.DataFrame,
        features: pd.DataFrame,
        is_raw: bool,
        simulated_pos: Optional[Dict[int, Tuple[float, float]]],
    ):
        """
        Represents a simulated game state instance with associated features and metadata.

        Parameters
        ----------
        state : pd.DataFrame
            The game state data for this instance.
        features : pd.DataFrame
            The computed features for this instance.
        is_raw : bool
            Indicates whether this instance is the original game state (True) or a simulated one (False).
        simulated_pos : Dict[int, Tuple[float, float]] or None
            A dictionary mapping player IDs to their simulated (x, y) positions (None if this is the raw instance).
        """
        self.state = state 
        self.features = features 
        self.is_raw = is_raw 
        self.simulated_pos = simulated_pos
        self.prob = None  

class exPressComponent(ABC):
    """Base class for all components."""

    component_name = "default"

    def __init__(self, 
                 db,
                 component, 
                 xfns: List[Callable],
                 nb_prev_actions: int = 1,
                 x_bins: int = config.field_length,
                 y_bins: int = config.field_width):
        
        self.db = db
        self.component = component
        self.model = component.model

        self.xfns = xfns
        self.nb_prev_actions = nb_prev_actions
        self.x_bins = x_bins
        self.y_bins = y_bins

    def simulate_features(
        self,
        game_id: int,
        action_id: int,
        pos_by_players: Optional[Dict[int, List[Tuple[float, float]]]] = None,
    ):
        """Apply a list of feature generators."""
        actions = add_names(self.db.actions(game_id)).reset_index()
        idx = actions[(actions["game_id"] == game_id) & (actions["action_id"] == action_id)].index

        if len(idx) != 1:
            raise ValueError(f"idx 중복 또는 누락 발생: {idx}")
        else:
            idx = idx[0]

        home_team_id, _ = self.db.get_home_away_team_id(game_id)
        states = play_left_to_right(gamestates(actions, self.nb_prev_actions), home_team_id)
        states = [state.loc[[idx]].copy() for state in states]

        simulated_instances = []

        # Compute fixed features (raw features)
        df_fixed_features = reduce(
            lambda left, right: pd.merge(left, right, how="outer", left_index=True, right_index=True),
            (fn(states) for fn in self.xfns),
        )
        simulated_instances.append(
            SimulatedInstance(
                state=states[0],
                features=df_fixed_features,
                is_raw=True,
                simulated_pos=None,
            )
        )

        player_ids = list(pos_by_players.keys())
        position_lists = [pos_by_players[id] for id in player_ids]
        position_combinations = list(itertools.product(*position_lists))

        for positions in tqdm(position_combinations, desc="Generating player's position simulations"):
            copy_states = copy.deepcopy(states)
            copy_freeze_frame = pd.DataFrame(copy_states[0].at[idx, "freeze_frame_360"])

            simulated_pos = {}
            for id, (x, y) in zip(player_ids, positions):
                copy_freeze_frame.loc[id, ['x', 'y']] = (x, y)
                simulated_pos[id] = (x, y)

                if copy_freeze_frame.at[id, "actor"]:
                    copy_states[0].loc[idx, ["start_x", "start_y"]] = (x, y)
                    copy_states[0].loc[idx, ["end_x", "end_y"]] = (x, y)

            copy_states[0].at[idx, "freeze_frame_360"] = copy_freeze_frame.to_dict(orient="records")

            df_simulated_features = reduce(
                lambda left, right: pd.merge(left, right, how="outer", left_index=True, right_index=True),
                (fn(copy_states) for fn in self.xfns),
            )

            simulated_instances.append(
                SimulatedInstance(
                    state=copy_states[0],
                    features=df_simulated_features,
                    is_raw=False,
                    simulated_pos=simulated_pos,
                )
            )

        return simulated_instances

    @abstractmethod
    def simulate(self, 
                 game_id: int, 
                 action_id: int, 
                 pos_by_players: Optional[Dict[int, List[Tuple[float, float]]]] = None
        ):
            """Abstract method to be implemented in child classes."""
            pass


class expressXGBoostComponent(exPressComponent):
    """XGBoost component."""

    component_name = "xgboost"
    def __init__(self, 
                 db,
                 component, 
                 xfns: List[Callable],
                 nb_prev_actions: int = 3,
                 x_bins: int = config.field_length,
                 y_bins: int = config.field_width
        ):
            super().__init__(db, component, xfns, nb_prev_actions, x_bins, y_bins)

    def simulate(
        self,
        game_id: int,
        action_id: int,
        pos_by_players: Optional[Dict[int, List[Tuple[float, float]]]] = None,
    ):
        simulated_instances = self.simulate_features(game_id, action_id, pos_by_players)

        for instance in tqdm(simulated_instances, desc="Simulating with xgboost model"):
            features = instance.features
            prob = self.model.predict_proba(features)[:, 1]
            instance.prob = prob[0]  # Convert to scalar

        return simulated_instances


class exPressPytorchComponent(exPressComponent):
    """PyTorch component."""

    component_name = "pytorch"

    def __init__(self, 
                 db,
                 component, 
                 xfns: List[Callable],
                 labels: pd.DataFrame = None,
                 nb_prev_actions: int = 3,
                 x_bins: int = config.field_length,
                 y_bins: int = config.field_width
        ): 
            super().__init__(db, component, xfns, nb_prev_actions, x_bins, y_bins)
            self.labels = labels
            self.transform = component.transform

    def simulate(
        self,
        game_id: int,
        action_id: int,
        pos_by_players: Optional[Dict[int, List[Tuple[float, float]]]] = None,
    ):
        simulated_instances = self.simulate_features(game_id, action_id, pos_by_players)

        for instance in tqdm(simulated_instances, desc="Simulating with PyTorch model"):
            sample_features = instance.features.iloc[0]
            sample_target = self.labels.loc[(game_id, action_id)].to_dict()

            sample = {
                "game_id": game_id,
                "action_id": action_id,
                **sample_features,
                **sample_target,
            }

            sample = self.transform(sample)  # ToSoccerMapTensor
            x, y = sample
            prob = self.model(x)  # self.forward(x)
            instance.prob = prob.item()  # Convert tensor to scalar

        return simulated_instances