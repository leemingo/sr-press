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
        gi = y["shot+%d" % i] & (y["team_id+%d" % i] == y["team_id"]) # 변경: idx가 압박을 당한 선수이므로 같은 팀의 선수가 슛을 성공하면
        res = res | gi

    return pd.DataFrame(res, columns=["concede_shots"])

def possession_change_by_seconds(actions: pd.DataFrame, n_seconds: int = 5) -> pd.DataFrame:
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

    actions["possession_change_by_seconds"] = False

    for idx, row in actions.iterrows():
        # Filter events within the same period and within the time window
        window_events = actions[
            (actions["period_id"] == row["period_id"]) &
            (actions["time_seconds"] >= row["time_seconds"]) &
            (actions["time_seconds"] <= row["time_seconds"] + n_seconds) &
            (actions["team_id"] != row["team_id"]) & # presser팀
            (actions["type_name"].isin(attracking_actiontypes))
        ]

        if not window_events.empty:
            actions.at[idx, "possession_change_by_seconds"] = True

    return actions[["possession_change_by_seconds"]]

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
        pressing_indicator = (y["period_id+%d" % i] == y["period_id"]) & (y["team_id+%d" % i] != y["team_id"]) & (y["attacking+%d" % i]) # presser팀이 소유권을 되찾는 경우
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
        pressing_indicator = (y["period_id+%d" % i] == y["period_id"]) & (y["team_id+%d" % i] != y["team_id"]) & (y["attacking+%d" % i]) & (dist <= distance)

        res = res | pressing_indicator

    return pd.DataFrame(res, columns=[f"possession_change_by_{nr_actions}_actions_and_{distance}m_distance"])

# Dynamically create variables for possession change functions for actions 1 to 10
k = 1
for i in range(k, 11):
    globals()[f"possession_change_by_{i}_actions"] = partial(possession_change_by_actions, nr_actions=i)
    globals()[f"possession_change_by_{i}_actions"].__name__ = f"possession_change_by_{i}_actions"

# Dynamically create variables for possession change functions for actions 1 to 10
k = 1
distance = 5
for i in range(k, 11):
    globals()[f"possession_change_by_{i}_actions_and_{distance}m_distance"] = partial(possession_change_by_actions_and_distance, nr_actions=i, distance=distance)
    globals()[f"possession_change_by_{i}_actions_and_{distance}m_distance"].__name__ = f"possession_change_by_{i}_actions_and_{distance}m_distance"

# Dynamically fetch variables for the possession change actions
dynamic_labels = [globals()[f"possession_change_by_{i}_actions"] for i in range(k, 11)]
dynamic_labels1 = [globals()[f"possession_change_by_{i}_actions_and_{distance}m_distance"] for i in range(k, 11)]

all_labels = [concede_shots, possession_change_by_seconds] + dynamic_labels + dynamic_labels1

def get_labels(
    db: Database,
    game_id: int,
    yfns: List[Callable] = all_labels,
    actionfilter: Optional[Callable] = None,
    pressured_idx: List[int] = None
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
    actions = add_names(db.actions(game_id))
    # 압박 이벤트는 소유권 검사시 제외: 애초에 a_i이후에 발생하는 소유권 여부이기 때문임
    actions = actions[actions["type_id"] != config.actiontypes.index("pressing")]

    if actionfilter is None:
        idx = pd.Series([True] * len(actions), index=actions.index)
    else:
        idx = actionfilter(actions)

    actions = actions.loc[idx]
    # reindex: 여러 선수가 압박을 한 경우여도 어차피 a_i 이후 공의 소유권 여부를 검사하므로 동일한 레이블을 갖음. ex) {S_i, p_i1}과 {S_i, p_i2}의 레이블은 동일함
    # 이에 따라 동일한 인덱스를 부여해야함
    # idx: 압박 당하는 액션(a_i)를 기준으로 병합했기 때문에 해당 인덱스는 압박을 당하는 팀(공을 소유한 팀)의 액션임 -> 레이블링시 주의해야함!
    if pressured_idx is not None:
        idx = actions.loc[actions.index.intersection(pressured_idx)].reindex(pressured_idx).index

    try:
        df_labels = pd.concat(
            # TODO: move .set_index to socceraction label generators
            [fn(actions.reset_index()).set_index(actions.index).loc[idx] for fn in yfns],
            axis=1,
        )
    except Exception:
        df_labels = pd.concat(
            # TODO: move .set_index to socceraction label generators
            [fn(actions).loc[idx] for fn in yfns],
            axis=1,
        )

    return df_labels
