"""Provides some utilities widely used by other modules."""
from typing import Dict, List, Sequence, Union
import pandas as pd
import express.config as config

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
        .merge(config.results_df, how="left") \
        .merge(config.bodyparts_df, how="left") \
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