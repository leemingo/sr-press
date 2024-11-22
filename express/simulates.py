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

from express.datasets import PressingDataset

from itertools import product

class expressXGBoostComponent():
    def __init__(self, component, dataset):
        self.component = component
        self.model = component.model
        self.dataset = dataset

    def adjust_player_positions(self, idx, target=None, adjustments=None):
        if adjustments is None:
            x_changes = range(-5, 6)
            y_changes = range(-5, 6)

            # player_id: actor, teammate_1, opponent_1, teammate_2, opponent_2,.....teammate_11, opponent_11
            player = "actor" if target is None else target
            adjustments_list = [{'player_id': player, 'dx': dx, 'dy': dy} for dx, dy in product(x_changes, y_changes)]

        results = []
        for adj in adjustments_list:
            simulated_sample = self.dataset.features.loc[idx].copy()

            if adj["player_id"] == "actor":
                x_col, y_col = ["start_x_a0"], ["start_y_a0"]
                player = "press"
            else:
                x_col = [col for col in simulated_sample.columns if col.startswith(adj["player_id"]) and "x" in col]
                y_col = [col for col in simulated_sample.columns if col.startswith(adj["player_id"]) and "y" in col]

                if len(x_col) != 1:
                    raise ValueError(f"x_col 중복 또는 누락 발생: {x_col}")
                if len(y_col) != 1:
                    raise ValueError(f"y_col 중복 또는 누락 발생: {y_col}")
            
            simulated_sample[x_col[0]] += adj["dx"]
            simulated_sample[y_col[0]] += adj["dy"] 

            probability = self.model.predict_proba(simulated_sample.to_frame().T)[:, 1]
            results.append({"dx": adj["dx"], "dy": adj["dy"], 
                            "simulated_x": simulated_sample[x_col[0]], "simulated_y": simulated_sample[y_col[0]],
                            "action": simulated_sample,
                            'probability': probability[0]}) # probability[0]: convert list to scalar

        return results
    
    def predict(self, idx):
        sample_features = self.dataset.features.loc[idx]
        prob = self.model.predict_proba(sample_features.to_frame().T)[:, 1]
        
        return prob[0] # convert list to scalar
        
class exPressPytorchComponent():
    def __init__(self, component, dataset):
        self.component = component
        self.model = component.model
        self.dataset = dataset

    def adjust_player_positions(self, idx, adjustments=None):
        if adjustments is None:
            x_changes = range(-5, 6)
            y_changes = range(-5, 6)
            adjustments_list = [{'player_id': 0, 'x': dx, 'y': dy} for dx, dy in product(x_changes, y_changes)]
        
        sample_features = self.dataset.features.loc[idx].to_dict()
        game_id, action_id = self.dataset.features.loc[idx].name
        sample_target = self.dataset.labels.loc[idx].to_dict()

        print(sample_features)
        print(sample_target)
        ss
        results = []
        for adj in adjustments_list:
            
            simulated_data = adjust_player_positions(original_data, {adj['player_id']: (adj['x'], adj['y'])})
            transformed_data = self.transform(simulated_data)
            probability = model.predict(transformed_data)
            results.append({'adjustment': adj, 'probability': probability})

        sample = {
            "game_id": game_id,
            "action_id": action_id,
            **sample_features,
            **sample_target,
        }

        for player_id, new_position in adjustments.items():
            adjusted_data.loc[adjusted_data['player_id'] == player_id, ['x', 'y']] = new_position
        return adjusted_data