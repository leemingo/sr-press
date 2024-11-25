"""Implements the labels used in each compoment."""
import os
import sys
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(base_path)

from typing import Callable, List, Optional
import pandas as pd
from express.utils import add_names
from express.databases import Database
import express.config as config

# P(G-|St,pt), if the (defending) team concedes a shot
def concede_shots(actions: pd.DataFrame, nr_actions: int = 10) -> pd.DataFrame:
    """Determine if the defending team concedes a shot by the attacking team within the next x actions.

    Parameters
    ----------
    actions : pd.DataFrame
        The actions of a game.
    nr_actions : int, default=10  # noqa: DAR103
        Number of actions after the current action to consider.

    Returns
    -------
    pd.DataFrame
        A DataFrame with a column 'shots' where each row is True if the opposing team takes a shot within
        the next x actions; otherwise False.
    """

    shots = actions["type_name"] == "shot"
    y = pd.concat([shots, actions["team_id"]], axis=1)
    y.columns = ["shot", "team_id"]

    # adding future results
    for i in range(1, nr_actions):
        for c in ["team_id", "shot"]:
            shifted = y[c].shift(-i) # Shifts to the near future
            shifted[-i:] = y[c].iloc[len(y) - 1] # Fill missing values at the end
            y["%s+%d" % (c, i)] = shifted

    res = y["shot"]
    for i in range(1, nr_actions):
        gi = y["shot+%d" % i] & (y["team_id+%d" % i] != y["team_id"]) # presser과 다른 팀의 선수가 슛을 성공하면
        res = res | gi

    return pd.DataFrame(res, columns=["concede_shots"])

# P(G+|St,pt) if the defending team successfully recovers the ball
def counterpress(actions: pd.DataFrame) -> pd.DataFrame:
    """Determine if a counterpressing action occurs within a specified time limit following a turnover.

    Parameters
    ----------
    actions : pd.DataFrame
        The actions of a game.
    time_limit : int, default=5
        The time limit in seconds to consider for counterpressing actions after a turnover.

    Returns
    -------
    pd.DataFrame
        A DataFrame with a column 'counterpress' where each row is True if a counterpress action occurs within the
        specified time limit after an open play turnover; otherwise False.
    """
    counter_press = (actions["result_id"] == config.results.index("success"))
    return pd.DataFrame(counter_press.values, columns=["counterpress"])

def posession_change(actions: pd.DataFrame, n_seconds: int = 5) -> pd.DataFrame:
    """
    Determine if possession changes within n (5) seconds after each 'pressing' action in the game_actions DataFrame.

    Parameters:
    actions : pd.DataFrame
        The actions of a game.
    n_seconds : int, default=5
        The time limit in seconds to consider for posession changed actions after a pressing.


    Returns
    -------
    pd.DataFrame
        A DataFrame with a column 'change_posessions' where each row is True if a posession changed occurs within the
        specified time limit after a pressing; otherwise False.
    """
    pressing = actions["type_name"] == "pressing"
    pressing_idx_list = list(pressing[pressing==True].index)

    for pressing_idx in pressing_idx_list:
        pressing_row = actions.loc[pressing_idx]
        seq_idx_list = list(actions[(actions['period_id'] == pressing_row['period_id']) & (actions['time_seconds'] >= pressing_row['time_seconds']) & (actions['time_seconds'] <= pressing_row['time_seconds'] + n_seconds)].index) # Indexing events within n seconds

        if actions.loc[seq_idx_list[-1]]['type_name'] != 'shot' and seq_idx_list[-1] != actions['action_id'].max(): # In cases such as throw-ins or fouls where possession changes, the next event is assigned after 5 seconds, so one additional event is included.
            seq_idx_list.append(seq_idx_list[-1] + 1)
        
        seq_df = actions.loc[seq_idx_list]
        seq_df = seq_df[seq_df['original_event_id'].notna()]

        pressing_team_id = pressing_row['team_id']
        if pressing_team_id in seq_df['possession_team_id'].values:
            pressing[pressing_idx] = True
        else:
            pressing[pressing_idx] = False
    return pd.DataFrame(pressing.values, columns=["posession_change"])



all_labels = [concede_shots, counterpress, posession_change] 

def get_labels(
    db: Database,
    game_id: int,
    yfns: List[Callable] = all_labels,
    actionfilter: Optional[Callable] = None,
):
    """Apply a list of label generators.

    Parameters
    ----------
    db : Database
        The database with raw data.
    game_id : int
        The ID of the game for which features should be computed.
    yfns : List[Callable], optional
        The label generators.
    actionfilter : Callable, optional
        A function that filters the actions to be used.
    nb_prev_actions : int, optional
        The number of previous actions to be included in a game state.

    Returns
    -------
    pd.DataFrame
        A dataframe with the labels.
    """
    game_actions = add_names(db.actions(game_id)).reset_index()
    if actionfilter is None:
        idx = pd.Series([True] * len(game_actions), index=game_actions.index)
    else:
        idx = actionfilter(game_actions)

    # check if we have to return an empty dataframe
    if idx.sum() < 1:
        return pd.DataFrame(columns=yfns)
    else:
        pressure_index = game_actions.loc[idx, ["game_id", "action_id"]].set_index(['game_id', 'action_id']).index

    df_labels = pd.concat(
        # TODO: move .set_index to socceraction label generators
        [fn(game_actions).loc[idx] for fn in yfns],
        axis=1,
    )

    return df_labels.set_index(pressure_index)
