"""Model architectures."""
import pickle
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import torch
from xgboost import XGBClassifier, XGBRegressor
from gplearn.genetic import SymbolicClassifier
from rich.progress import track
from tqdm import tqdm
from sklearn.model_selection import cross_val_score, train_test_split
from torch.utils.data import DataLoader, Subset, random_split
from express.utils import add_names, play_left_to_right
from express.datasets import PressingDataset
from express.features import gamestates
from itertools import product
import express.config as config
from functools import partial, reduce, wraps
import copy

class SimulatedInstance:
    def __init__(self, raw_data, features, raw, freeze_frame_idx, simulated_x, simulated_y):
        self.data = raw_data
        self.features = features
        self.raw = raw
        self.freeze_frame_idx = freeze_frame_idx
        self.simulated_x = simulated_x
        self.simulated_y = simulated_y
        self.prob = None

class expressXGBoostComponent():
    def __init__(self, 
                 component, 
                 db,
                 xfns: List[Callable],
                 nb_prev_actions: int = 3,
                 x_bins: int = 105,
                 y_bins: int = 68
        ):
            self.component = component
            self.model = component.model
            self.db = db

            self.xfns = xfns
            self.nb_prev_actions = nb_prev_actions
            self.x_bins, self.y_bins = x_bins, y_bins

    def simulate_features(
        self,
        game_id: int,
        action_id: int,
        freeze_frame_idx: int = 0,
        xy: Optional[List[pd.DataFrame]] = None,
    ):
        """Apply a list of feature generators.

        Parameters
        ----------
        game_id : int
            The ID of the game for which features should be computed.
        action_id : int
            The ID of the action for which features should be computed.
        xfns : List[Callable], optional
            The feature generators.
        nb_prev_actions : int, optional
            The number of previous actions to be included in a game state.
        xy: list(pd.DataFrame), optional
            The x and y coordinates of simulated start location.
        x_bins : int, optional
            The number of bins to simulated for the start location along the x-axis.
        y_bins : int, optional
            The number of bins to simulated for the start location along the y-axis.


        Returns
        -------
        pd.DataFrame
            A dataframe with the features.
        """
        # retrieve actions from database
        actions = add_names(self.db.actions(game_id)).reset_index()
        idx = actions[(actions["game_id"] == game_id) & (actions["action_id"] == action_id)].index # only an event

        if len(idx) != 1:
            raise ValueError(f"idx 중복 또는 누락 발생: {idx}")
        else:
            idx = idx[0]

        # convert actions to gamestates
        home_team_id, _ = self.db.get_home_away_team_id(game_id)
        states = play_left_to_right(gamestates(actions, self.nb_prev_actions), home_team_id)
        states = [state.loc[[idx]].copy() for state in states]
        
        # simulate start location
        if xy is None:
            # create bin centers
            yy, xx = np.ogrid[0.5:self.y_bins, 0.5:self.x_bins]
            # map to spadl coordinates
            x_coo = np.clip(xx / self.x_bins * config.field_length, 0, config.field_length)
            y_coo = np.clip(yy / self.y_bins * config.field_width, 0, config.field_width)

        simulated_instances = []
        # compute fixed features(raw features)
        df_fixed_features = reduce(
            lambda left, right: pd.merge(left, right, how="outer", left_index=True, right_index=True),
            (fn(states) for fn in self.xfns),
        )
        simulated_instances.append(SimulatedInstance(states[0], df_fixed_features, raw=True,
                                                     freeze_frame_idx= freeze_frame_idx, simulated_x=None, simulated_y=None))

        if xy is None:
            for x, y in tqdm(np.array(np.meshgrid(x_coo, y_coo)).T.reshape(-1, 2)):
                copy_states = copy.deepcopy(states) # deep copy
                copy_freeze_frame = pd.DataFrame(copy_states[0].at[idx, "freeze_frame_360"])
                copy_freeze_frame.loc[freeze_frame_idx, ["x", "y"]] = (x, y) 

                # actor: (start_x, start_y) and freeze_frame_360'actor coordinate가 변경되야함
                if copy_freeze_frame.at[freeze_frame_idx, "actor"]:
                    copy_states[0].loc[idx, ["start_x", "start_y"]] = (x, y)
                    copy_states[0].loc[idx, ["end_x", "end_y"]] = (x, y) 
                    
                copy_states[0].at[idx, "freeze_frame_360"] = copy_freeze_frame.to_dict(orient="records")

                df_simulated_features = reduce(
                        lambda left, right: pd.merge(left, right, how="outer", left_index=True, right_index=True),
                        (fn(copy_states) for fn in self.xfns),
                    )
                simulated_instances.append(SimulatedInstance(copy_states[0], df_simulated_features, raw=False,
                                                             freeze_frame_idx= freeze_frame_idx, simulated_x= x, simulated_y= y))
        else:
            for x, y in tqdm(xy):
                copy_states = copy.deepcopy(states) # deep copy
                copy_freeze_frame = pd.DataFrame(copy_states[0].at[idx, "freeze_frame_360"])
                copy_freeze_frame.loc[freeze_frame_idx, ["x", "y"]] = (x, y) 

                # actor: (start_x, start_y) and freeze_frame_360'actor coordinate가 변경되야함
                if copy_freeze_frame.at[freeze_frame_idx, "actor"]:
                    copy_states[0].loc[idx, ["start_x", "start_y"]] = (x, y) 
                    copy_states[0].loc[idx, ["end_x", "end_y"]] = (x, y) 

                copy_states[0].at[idx, "freeze_frame_360"] = copy_freeze_frame.to_dict(orient="records")

                df_simulated_features = reduce(
                        lambda left, right: pd.merge(left, right, how="outer", left_index=True, right_index=True),
                        (fn(copy_states) for fn in self.xfns),
                    )
                simulated_instances.append(SimulatedInstance(copy_states[0], df_simulated_features, raw=False,
                                                             freeze_frame_idx= freeze_frame_idx, simulated_x= x, simulated_y= y))

        return simulated_instances

    def simulate(        
        self,
        game_id: int,
        action_id: int,
        freeze_frame_idx: int = 0,
        xy: Optional[List[pd.DataFrame]] = None,
    ):
        simulated_instances = self.simulate_features(game_id, action_id, freeze_frame_idx, xy)

        for instance in simulated_instances:
            features = instance.features
            prob = self.model.predict_proba(features)[:, 1]
            instance.prob = prob[0] # convert list to scalar

        return simulated_instances
     