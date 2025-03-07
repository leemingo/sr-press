"""Provides some utilities widely used by other modules."""
import os
import sys
base_path = os.path.abspath(os.path.join(os.getcwd()))
sys.path.append(base_path)

from typing import Dict, List, Sequence, Union
import numpy as np
import pandas as pd

import express.config as config
from express.databases import SQLiteDatabase

def add_names(actions: pd.DataFrame) -> pd.DataFrame:
    """Add the type name(include Pressing), result name and bodypart name to a SPADL(include Pressing) dataframe.

    Parameters
    ----------
    actions : pd.DataFrame
        A SPADL dataframe.

    Returns
    -------
    pd.DataFrame
        The original dataframe with a 'type_name', 'result_name' and
        'bodypart_name' appended.
    """

    return actions.drop(columns=["type_name", "result_name", "bodypart_name"], errors="ignore") \
        .merge(config.actiontypes_df(), how="left") \
        .merge(config.results_df(), how="left") \
        .merge(config.bodyparts_df(), how="left") \
        .set_index(actions.index)

def play_left_to_right(gamestates: List[pd.DataFrame], home_team_id: int) -> List[pd.DataFrame]:
    """Perform all action in the same playing direction.

    This changes the start and end location of each action and the freeze
    frame, such that all actions are performed as if the team plays from left
    to right.

    Parameters
    ----------
    gamestates : GameStates
        The game states of a game.
    home_team_id : int
        The ID of the home team.

    Returns
    -------
    GameStates
        The game states with all actions performed left to right.
    """
    a0 = gamestates[0]
    away_idx = a0.team_id != home_team_id
    for actions in gamestates:
        for col in ["start_x", "end_x"]:
            actions.loc[away_idx, col] = config.field_length - actions[away_idx][col].values
        for col in ["start_y", "end_y"]:
            actions.loc[away_idx, col] = config.field_width - actions[away_idx][col].values

        for idx, action in actions.loc[away_idx].iterrows():
            freeze_frame = action["freeze_frame_360"]

            if freeze_frame is not None:
                freezedf = pd.DataFrame(freeze_frame).fillna(
                    {"teammate": False, "actor": False, "keeper": False}
                )
                freezedf["x"] = config.field_length - freezedf["x"].values
                freezedf["y"] = config.field_width - freezedf["y"].values
                actions.at[idx, "freeze_frame_360"] = freezedf.to_dict(orient="records")

    return gamestates

def load_actions(game_id):
    try:
        TRAIN_DB_PATH = os.path.join(base_path, "stores/train_database.sqlite")
        train_db = SQLiteDatabase(TRAIN_DB_PATH)
        return add_names(train_db.actions(game_id))
    except:
        TEST_DB_PATH = os.path.join(base_path, "stores/test_database.sqlite")
        test_db = SQLiteDatabase(TEST_DB_PATH)
        return add_names(test_db.actions(game_id))

def pressed_prev_actions(action: pd.Series, actions: pd.DataFrame, nb_prev_actions):
    game_id, action_id = action.game_id, action.action_id

    pressing_action = actions.loc[(game_id, action_id)]

    prev_action_range = (
        (actions.index.get_level_values("action_id") > action_id - nb_prev_actions) &
        (actions.index.get_level_values("action_id") < action_id)
    )

    overlap = (
        prev_action_range & 
        (actions["team_id"] != pressing_action["team_id"]) &
        ((actions["time_seconds"] + actions["duration"]) >= pressing_action["time_seconds"]) &
        ((actions["time_seconds"] + actions["duration"]) <= (pressing_action["time_seconds"] + pressing_action["duration"]))
    )

    return not actions[overlap].empty
