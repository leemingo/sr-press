"""Implements the labels used in each compoment."""
import os
import sys
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(base_path)

from typing import Callable, List, Optional
from functools import partial, reduce, wraps
import numpy as np
import pandas as pd
from express.utils import add_names
from express.databases import Database
import express.config as config

attracking_actiontypes: list[str] = [
    "pass",
    "cross",
    "throw_in",
    "freekick_crossed",
    "freekick_short",
    "corner_crossed",
    "corner_short",
    "take_on",
    # "foul",
    # "tackle",
    # "interception",
    "shot",
    "shot_penalty",
    "shot_freekick",
    "keeper_save",
    "keeper_claim",
    "keeper_punch",
    "keeper_pick_up",
    # "clearance",
    # "bad_touch",
    # "non_action",
    "dribble",
    "goalkick",
]

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
    y = pd.concat([shots, actions[["team_id", "period_id"]]], axis=1)
    y.columns = ["shot", "team_id", "period_id"]

    # adding future results
    for i in range(1, nr_actions):
        for c in ["team_id", "shot", "period_id"]:
            shifted = y[c].shift(-i) # Shifts to the near future
            shifted[-i:] = y[c].iloc[len(y) - 1] # Fill missing values at the end
            y["%s+%d" % (c, i)] = shifted

    res = y["shot"]
    for i in range(1, nr_actions):
        gi = (y["period_id+%d" % i] == y["period_id"]) & (y["shot+%d" % i]) & (y["team_id+%d" % i] != y["team_id"]) # presser과 다른 팀의 선수가 슛을 성공하면
        res = res | gi

    return pd.DataFrame(res, columns=["concede_shots"])

# def possession_change_by_seconds(actions: pd.DataFrame, n_seconds: int = 5) -> pd.DataFrame:
#     """
#     Determine if possession changes within n (5) seconds after each 'pressing' action in the game_actions DataFrame.

#     Parameters:
#     actions : pd.DataFrame
#         The actions of a game.
#     n_seconds : int, default=5
#         The time limit in seconds to consider for posession changed actions after a pressing.


#     Returns
#     -------
#     pd.DataFrame
#         A DataFrame with a column 'change_posessions' where each row is True if a posession changed occurs within the
#         specified time limit after a pressing; otherwise False.
#     """

#     actions["possession_change_by_seconds"] = False
#     pressing_actions = actions[actions["type_name"] == "pressing"]

#     for pressing_idx, pressing_row in pressing_actions.iterrows():
#         # Filter events within the same period and within the time window
#         window_events = actions[
#             (actions["period_id"] == pressing_row["period_id"]) &
#             (actions["time_seconds"] >= pressing_row["time_seconds"]) &
#             (actions["time_seconds"] <= pressing_row["time_seconds"] + n_seconds) &
#             (actions["type_name"].isin(attracking_actiontypes))
#         ]

#         if not window_events.empty:
#             if pressing_row["team_id"] in window_events["team_id"].values:
#                 actions.at[pressing_idx, "possession_change_by_seconds"] = True

#     return actions[["possession_change_by_seconds"]]

def possession_change_by_seconds(actions: pd.DataFrame, n_seconds: int = 5) -> pd.DataFrame:
    """
    Determine if possession changes within n (5) seconds after each 'pressing' action in the game_actions DataFrame.

    Parameters:
    actions : pd.DataFrame
        The actions of a game.
    n_seconds : int, default=5
        The time limit in seconds to consider for possession changed actions after a pressing.


    Returns
    -------
    pd.DataFrame
        A DataFrame with a column 'change_possessions' where each row is True if a possession changed occurs within the
        specified time limit after a pressing; otherwise False.
    """
    pressing = actions["type_name"] == "pressing"
    pressing_idx_list = list(pressing[pressing==True].index)

    for pressing_idx in pressing_idx_list:
        pressing_row = actions.loc[pressing_idx]
        seq_idx_list = list(actions[(actions['period_id'] == pressing_row['period_id']) & (actions['time_seconds'] >= pressing_row['time_seconds']) & (actions['time_seconds'] <= pressing_row['time_seconds'] + n_seconds)].index) # Indexing events within n seconds

        delayed_actions = ["throw_in", "freekick_short", "freekick_crossed", "corner_crossed", "corner_short"]
        if seq_idx_list[-1] != actions['action_id'].max():
            if actions.loc[seq_idx_list[-1]+1]['type_name'] in delayed_actions:
                seq_idx_list.append(seq_idx_list[-1] + 1)
        seq_df = actions.loc[seq_idx_list]
        seq_df = seq_df[seq_df['original_event_id'].notna()]

        possession_actions = ["pass", "dribble", "cross", "shot"] + delayed_actions
        pressing_team_id = pressing_row['team_id']
        if not seq_df[(seq_df['team_id'] == pressing_team_id) & (seq_df['type_name'].isin(possession_actions))].empty:
            pressing[pressing_idx] = True
        else:
            pressing[pressing_idx] = False
    return pd.DataFrame(pressing.values, columns=[f"possession_change_by_{n_seconds}_seconds"])

def possession_change_by_seconds_and_distance(actions: pd.DataFrame, n_seconds: int = 5, distance: int = 5) -> pd.DataFrame:
    """
    Determine if possession changes within n (5) seconds after each 'pressing' action in the game_actions DataFrame.

    Parameters:
    actions : pd.DataFrame
        The actions of a game.
    n_seconds : int, default=5
        The time limit in seconds to consider for possession changed actions after a pressing.


    Returns
    -------
    pd.DataFrame
        A DataFrame with a column 'change_possessions' where each row is True if a possession changed occurs within the
        specified time limit after a pressing; otherwise False.
    """
    pressing = actions["type_name"] == "pressing"
    pressing_idx_list = list(pressing[pressing==True].index)

    for pressing_idx in pressing_idx_list:
        pressing_row = actions.loc[pressing_idx]
        dist = np.hypot(actions["start_x"] - pressing_row["start_x"], actions["start_y"] - pressing_row["start_y"])
        seq_idx_list = list(actions[(actions['period_id'] == pressing_row['period_id']) & 
                                    (actions['time_seconds'] >= pressing_row['time_seconds']) & 
                                    (actions['time_seconds'] <= pressing_row['time_seconds'] + n_seconds) &
                                    (dist <= distance)
                                    ].index) # Indexing events within n seconds

        delayed_actions = ["throw_in", "freekick_short", "freekick_crossed", "corner_crossed", "corner_short"]
        if seq_idx_list[-1] != actions['action_id'].max():
            if actions.loc[seq_idx_list[-1]+1]['type_name'] in delayed_actions:
                seq_idx_list.append(seq_idx_list[-1] + 1)
        seq_df = actions.loc[seq_idx_list]
        seq_df = seq_df[seq_df['original_event_id'].notna()]

        possession_actions = ["pass", "dribble", "cross", "shot"] + delayed_actions
        pressing_team_id = pressing_row['team_id']
        if not seq_df[(seq_df['team_id'] == pressing_team_id) & (seq_df['type_name'].isin(possession_actions))].empty:
            pressing[pressing_idx] = True
        else:
            pressing[pressing_idx] = False
    return pd.DataFrame(pressing.values, columns=[f"possession_change_by_{n_seconds}_seconds_and_{distance}m_distance"])

# P(G-|St,pt), if the (defending) team concedes a shot
def possession_change_by_actions(actions: pd.DataFrame, nr_actions: int = 10) -> pd.DataFrame:
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

    attracking_actions = actions["type_name"].isin(attracking_actiontypes)
    y = pd.concat([attracking_actions, actions[["team_id", "period_id"]]], axis=1)
    y.columns = ["attacking", "team_id", "period_id"]

    # adding future results
    for i in range(1, nr_actions):
        for c in ["attacking", "team_id", "period_id"]:
            shifted = y[c].shift(-i) # Shifts to the near future
            shifted[-i:] = y[c].iloc[len(y) - 1] # Fill missing values at the end
            y["%s+%d" % (c, i)] = shifted

    res = pd.Series([False] * len(actions))
    for i in range(1, nr_actions):
        pressing_indicator = (y["period_id+%d" % i] == y["period_id"]) & (y["team_id+%d" % i] == y["team_id"]) & (y["attacking+%d" % i]) 

        res = res | pressing_indicator

    return pd.DataFrame(res, columns=[f"possession_change_by_{nr_actions}_actions"])

# P(G-|St,pt), if the (defending) team concedes a shot
def possession_change_by_actions_and_distance(actions: pd.DataFrame, nr_actions: int = 10, distance: int = 5) -> pd.DataFrame:
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

    attracking_actions = actions["type_name"].isin(attracking_actiontypes)
    y = pd.concat([attracking_actions, actions[["team_id", "start_x", "start_y", "period_id"]]], axis=1)
    y.columns = ["attacking", "team_id", "start_x", "start_y", "period_id"]

    # adding future results
    for i in range(1, nr_actions):
        for c in ["attacking", "team_id", "start_x", "start_y", "period_id"]:
            shifted = y[c].shift(-i) # Shifts to the near future
            shifted[-i:] = y[c].iloc[len(y) - 1] # Fill missing values at the end
            y["%s+%d" % (c, i)] = shifted

    res = pd.Series([False] * len(actions))
    for i in range(1, nr_actions):
        dist = np.hypot(y["start_x+%d" % i] - y["start_x"], y["start_y+%d" % i] - y["start_y"])
        pressing_indicator = (y["period_id+%d" % i] == y["period_id"]) & (y["team_id+%d" % i] == y["team_id"]) & (y["attacking+%d" % i]) & (dist <= distance)

        res = res | pressing_indicator

    return pd.DataFrame(res, columns=[f"possession_change_by_{nr_actions}_actions_and_{distance}m_distance"])

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

# Dynamically create variables for possession change functions for actions 1 to 10
for i in range(2, 8, 2):
    globals()[f"possession_change_by_{i}_actions"] = partial(possession_change_by_actions, nr_actions=i)
    globals()[f"possession_change_by_{i}_actions"].__name__ = f"possession_change_by_{i}_actions"

# Dynamically create variables for possession change functions for actions 1 to 10
for i in range(2, 8, 2):
    for d in range(3, 11, 2):
        globals()[f"possession_change_by_{i}_actions_and_{d}m_distance"] = partial(possession_change_by_actions_and_distance, nr_actions=i, distance=d)
        globals()[f"possession_change_by_{i}_actions_and_{d}m_distance"].__name__ = f"possession_change_by_{i}_actions_and_{d}m_distance"

for i in range(3, 8):
    globals()[f"possession_change_by_{i}_seconds"] = partial(possession_change_by_seconds, n_seconds=i)
    globals()[f"possession_change_by_{i}_seconds"].__name__ = f"possession_change_by_{i}_seconds"

for i in range(3, 8):
    for d in range(3, 11, 2):
        globals()[f"possession_change_by_{i}_seconds_and_{d}m_distance"] = partial(possession_change_by_seconds_and_distance, n_seconds=i, distance=d)
        globals()[f"possession_change_by_{i}_seconds_and_{d}m_distance"].__name__ = f"possession_change_by_{i}_seconds_and_{d}m_distance"

# Dynamically fetch variables for the possession change actions
dynamic_labels = [globals()[f"possession_change_by_{i}_actions"] for i in range(2, 8, 2)]
dynamic_labels1 = [globals()[f"possession_change_by_{i}_actions_and_{d}m_distance"] for d in range(3, 11, 2) for i in range(2, 8, 2)]

dynamic_labels2 = [globals()[f"possession_change_by_{i}_seconds"] for i in range(3, 8)]
dynamic_labels3 = [globals()[f"possession_change_by_{i}_seconds_and_{d}m_distance"] for d in range(3, 11, 2) for i in range(3, 8)]
all_labels = [concede_shots, counterpress] + dynamic_labels + dynamic_labels1 + dynamic_labels2 + dynamic_labels3

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
