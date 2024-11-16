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

all_labels = [concede_shots, counterpress] 

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

    df_labels = pd.concat(
        # TODO: move .set_index to socceraction label generators
        [fn(game_actions).loc[idx] for fn in yfns],
        axis=1,
    )

    return df_labels
