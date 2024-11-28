"""Implements the state features used in exPressComponent."""
import os
import sys
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(base_path)

from functools import partial, reduce, wraps
from typing import Callable, Dict, List, Optional, Union, no_type_check

import numpy as np
import pandas as pd
import math

import express.config as config
from express.databases import Database
from express.databases.base import TABLE_ACTIONS
from express.utils import add_names, play_left_to_right

def required_fields(fields):
    def decorator(func):
        func.required_fields = fields
        return func

    return decorator

def gamestates(actions, nb_prev_actions: int = 3):
    r"""Convert a dataframe of actions to gamestates.

    Each gamestate is represented as the <nb_prev_actions> previous actions.

    The list of gamestates is internally represented as a list of actions
    dataframes :math:`[a_0,a_1,\ldots]` where each row in the a_i dataframe contains the
    previous action of the action in the same row in the :math:`a_{i-1}` dataframe.

    Parameters
    ----------
    actions : Actions
        A DataFrame with the actions of a game.
    nb_prev_actions : int, default=3  # noqa: DAR103
        The number of previous actions included in the game state.

    Raises
    ------
    ValueError
        If the number of actions is smaller 1.

    Returns
    -------
    GameStates
         The <nb_prev_actions> previous actions for each action.
    """

    if nb_prev_actions < 1:
        raise ValueError("The game state should include at least one preceding action.")
    states = [actions]
    for i in range(1, nb_prev_actions):
        # Issue 1: List-type columns like ['visible_area_360', 'freeze_frame_360'] raise TypeError during interpolation.
        # socceraction module update on 24.06.23 introduced an error when interpolating these columns:
        # https://github.com/ML-KULeuven/socceraction/commit/e9c398ea50219eb6eb188c44c35ad70b7fd79cc6
        
        # Issue 2: All events missing 'freeze_frame_360' are interpolated unintentionally.
        # The intended logic was to interpolate only the first event in each half, but other events without freeze_frame_360' are also filled with x.iloc[0], causing incorrect interpolations.                                                                              
        # prev_actions = actions.groupby(["game_id", "period_id"], sort=False, as_index=False).apply(
        #     lambda x: x.shift(i, fill_value=float("nan")).fillna(x.iloc[0])  # noqa: B023
        # )

        group_actions = actions.groupby(['game_id', 'period_id'], sort=False, as_index=False, group_keys=False)
        prev_actions = group_actions.apply(lambda x: x.shift(i))
        fill_indices = group_actions.head(i).index # First i rows per group get NaN, fill with original values
        prev_actions.loc[fill_indices] = actions.loc[fill_indices]

        states.append(prev_actions)  

    return states

@no_type_check
def simple(actionfn: Callable):
    """Make a function decorator to apply actionfeatures to game states.

    Parameters
    ----------
    actionfn : Callable
        A feature transformer that operates on actions.

    Returns
    -------
    FeatureTransfomer
        A feature transformer that operates on game states.
    """

    @wraps(actionfn)
    def _wrapper(gamestates) -> pd.DataFrame:
        if not isinstance(gamestates, (list,)):
            gamestates = [gamestates]
        X = []
        for i, a in enumerate(gamestates):
            Xi = actionfn(a)
            Xi.columns = [c + "_a" + str(i) for c in Xi.columns]
            X.append(Xi)
        return pd.concat(X, axis=1)

    return _wrapper

# SPADL FEATURES: For each of the actions, we define a set of categorical and 
# real-valued features based on information explicitly included in the SPADL representation. 

@required_fields(["type_id"])
@simple
def actiontype(actions):
    """Get the type of each action.

    Parameters
    ----------
    actions : Actions
        The actions of a game.

    Returns
    -------
    Features
        The 'type_name' of each action.
    """
    X = pd.DataFrame(index=actions.index)
    X["actiontype"] = pd.Categorical(
        actions["type_id"].replace(config.actiontypes_df().type_name.to_dict()),
        categories=config.actiontypes,
        ordered=False,
    )
    return X

@required_fields(["type_name"])
@simple
def actiontype_onehot(actions):
    """Get the one-hot-encoded type of each action.

    Parameters
    ----------
    actions : pd.DataFrame
        The actions of a game.

    Returns
    -------
    pd.DataFrame
        A one-hot encoding of each action's type.
    """
    X = pd.DataFrame()
    for type_name in config.actiontypes:
        col = 'type_' + type_name
        X[col] = actions['type_name'] == type_name
    return X

@required_fields(["result_id"])
@simple
def result(actions):
    """Get the result of each action.

    Parameters
    ----------
    actions : Actions
        The actions of a game.

    Returns
    -------
    Features
        The 'result_name' of each action.
    """

    X = pd.DataFrame(index=actions.index)
    X["result"] = pd.Categorical(
        actions["result_id"].replace(config.results_df().result_name.to_dict()),
        categories=config.results,
        ordered=False,
    )
    return X

@required_fields(["result_name"])
@simple
def result_onehot(actions):
    """Get the one-hot-encode result of each action.

    Parameters
    ----------
    actions : pd.DataFrame
        The actions of a game.

    Returns
    -------
    pd.DataFrame
        The one-hot encoding of each action's result.
    """
    X = pd.DataFrame()
    for result_name in config.results:
        col = 'result_' + result_name
        X[col] = actions['result_name'] == result_name
    return X

@required_fields(["bodypart_id"])
@simple
def bodypart(actions):
    """Get the body part used to perform each action.

    Parameters
    ----------
    actions : Actions
        The actions of a game.

    Returns
    -------
    Features
        The 'bodypart_name' of each action.
    """

    X = pd.DataFrame(index=actions.index)
    foot_id = config.bodyparts.index("foot")
    left_foot_id = config.bodyparts.index("foot_left")
    right_foot_id = config.bodyparts.index("foot_right")
    X["bodypart"] = pd.Categorical(
        actions["bodypart_id"]
        .replace([left_foot_id, right_foot_id], foot_id)
        .replace(config.bodyparts_df().bodypart_name.to_dict()),
        categories=["foot", "head", "other", "head/other"],
        ordered=False,
    )
    return X

@required_fields(["bodypart_name"])
@simple
def bodypart_onehot(actions):
    """Get the one-hot-encoded bodypart of each action.

    Parameters
    ----------
    actions : pd.DataFrame
        The actions of a game.

    Returns
    -------
    pd.DataFrame
        The one-hot encoding of each action's bodypart.
    """
    X = pd.DataFrame()
    for bodypart_name in config.bodyparts:
        col = 'bodypart_' + bodypart_name
        X[col] = actions['bodypart_name'] == bodypart_name
    return X

@required_fields(["period_id", "time_seconds"])
@simple
def time(actions):
    """Get the time when each action was performed.

    This generates the following features:
        :period_id:
            The ID of the period.
        :time_seconds:
            Seconds since the start of the period.
        :time_seconds_overall:
            Seconds since the start of the game. Stoppage time during previous
            periods is ignored.

    Parameters
    ----------
    actions : pd.DataFrame
        The actions of a game.

    Returns
    -------
    pd.DataFrame
        The 'period_id', 'time_seconds' and 'time_seconds_overall' when each
        action was performed.
    """
    timedf = actions[['period_id', 'time_seconds']].copy()
    timedf['time_seconds_overall'] = ((timedf.period_id - 1) * 45 * 60) + timedf.time_seconds
    return timedf

@required_fields(["start_x", "start_y"])
@simple
def startlocation(actions):
    """Get the location where each action started.

    Parameters
    ----------
    actions : pd.DataFrame
        The actions of a game.

    Returnss
    -------
    pd.DataFrame
        The 'start_x' and 'start_y' location of each action.
    """
    return actions[['start_x', 'start_y']]


@required_fields(["end_x", "end_y"])
@simple
def endlocation(actions):
    """Get the location where each action ended.

    Parameters
    ----------
    actions : pd.DataFrame
        The actions of a game.

    Returns
    -------
    pd.DataFrame
        The 'end_x' and 'end_y' location of each action.
    """
    return actions[['end_x', 'end_y']]

# Complex FEATURES: The complex features combine information within an action and across consecutive actions.

@required_fields(["start_x", "start_y"])
@simple
def startpolar(actions):
    """Get the polar coordinates of each action's start location.

    The center of the opponent's goal is used as the origin.

    Parameters
    ----------
    actions : pd.DataFrame
        The actions of a game.

    Returns
    -------
    pd.DataFrame
        The 'start_dist_to_goal' and 'start_angle_to_goal' of each action.
    """
    polardf = pd.DataFrame()
    dx = abs(config._goal_x - actions['start_x'])
    dy = abs(config._goal_y - actions['start_y'])
    polardf['start_dist_to_goal'] = np.sqrt(dx ** 2 + dy ** 2)
    with np.errstate(divide='ignore', invalid='ignore'):
        polardf['start_angle_to_goal'] = np.nan_to_num(np.arctan(dy / dx))
    return polardf


@required_fields(["end_x", "end_y"])
@simple
def endpolar(actions):
    """Get the polar coordinates of each action's end location.

    The center of the opponent's goal is used as the origin.

    Parameters
    ----------
    actions : pd.DataFrame
        The actions of a game.

    Returns
    -------
    pd.DataFrame
        The 'start_dist_to_goal' and 'start_angle_to_goal' of each action.
    """
    polardf = pd.DataFrame()
    dx = abs(config._goal_x - actions['end_x'])
    dy = abs(config._goal_y - actions['end_y'])
    polardf['end_dist_to_goal'] = np.sqrt(dx ** 2 + dy ** 2)
    with np.errstate(divide='ignore', invalid='ignore'):
        polardf['end_angle_to_goal'] = np.nan_to_num(np.arctan(dy / dx))
    return polardf

@required_fields(["start_x", "start_y", "end_x", "end_y"])
@simple
def movement(actions):
    """Get the distance covered by each action.

    Parameters
    ----------
    actions : pd.DataFrame
        The actions of a game.

    Returns
    -------
    pd.DataFrame
        The horizontal ('dx'), vertical ('dy') and total ('movement') distance
        covered by each action.
    """
    mov = pd.DataFrame()
    mov['dx'] = actions.end_x - actions.start_x
    mov['dy'] = actions.end_y - actions.start_y
    mov['movement'] = np.sqrt(mov.dx ** 2 + mov.dy ** 2)
    return mov

@required_fields(["team_id"])
def team(gamestates):
    """Check whether the possession changed during the game state.

    For each action in the game state, True if the team that performed the
    action is the same team that performed the last action of the game state;
    otherwise False.

    Parameters
    ----------
    gamestates : pd.DataFrame
        The game states of a game.

    Returns
    -------
    pd.DataFrame
        A dataframe with a column 'team_ai' for each <nb_prev_actions> indicating
        whether the team that performed action a0 is in possession.
    """
    a0 = gamestates[0]
    teamdf = pd.DataFrame()
    for i, a in enumerate(gamestates[1:]):
        teamdf['team_a' + (str(i + 1))] = a.team_id == a0.team_id
    return teamdf

@required_fields(["time_seconds"])
def time_delta(gamestates):
    """Get the number of seconds between the last and previous actions.

    Parameters
    ----------
    gamestates : pd.DataFrame
        The game states of a game.

    Returns
    -------
    pd.DataFrame
        A dataframe with a column 'time_delta_i' for each <nb_prev_actions>
        containing the number of seconds between action ai and action a0.
    """
    a0 = gamestates[0]
    dt = pd.DataFrame()
    for i, a in enumerate(gamestates[1:]):
        dt['time_delta_a' + (str(i + 1))] = a0.time_seconds - a.time_seconds
    return dt

@required_fields([["start_x", "start_y", "end_x", "end_y"]])
def space_delta(gamestates):
    """Get the distance covered between the last and previous actions.

    Parameters
    ----------
    gamestates : pd.DataFrame
        The gamestates of a game.

    Returns
    -------
    pd.DataFrame
        A dataframe with a column for the horizontal ('dx_a0i'), vertical
        ('dy_a0i') and total ('mov_a0i') distance covered between each
        <nb_prev_actions> action ai and action a0.
    """
    a0 = gamestates[0]
    spaced = pd.DataFrame()
    for i, a in enumerate(gamestates[1:]):
        dx = a.end_x - a0.start_x
        spaced['dx_a0' + (str(i + 1))] = dx
        dy = a.end_y - a0.start_y
        spaced['dy_a0' + (str(i + 1))] = dy
        spaced['mov_a0' + (str(i + 1))] = np.sqrt(dx ** 2 + dy ** 2)
    return spaced

@required_fields(["start_x", "start_y"])
@simple
def relative_startlocation(actions):
    """Get the location where each action started relative to the sideline and goalline.

    Parameters
    ----------
    actions : SPADLActions
        The actions of a game.

    Returns
    -------
    Features
        The 'start_dist_sideline' and 'start_dist_goalline' of each action.
    """
    actions["_start_dist_sideline"] = config._spadl_cfg["width"] - actions["start_y"]
    actions["start_dist_sideline"] = actions[["_start_dist_sideline", "start_y"]].min(axis=1)
    actions["start_dist_goalline"] = config._spadl_cfg["length"] - actions["start_x"]
    return actions[["start_dist_sideline", "start_dist_goalline"]]

@required_fields(["end_x", "end_y"])
@simple
def relative_endlocation(actions):
    """Get the location where each action ended relative to the sideline and goalline.

    Parameters
    ----------
    actions : SPADLActions
        The actions of a game.

    Returns
    -------
    Features
        The 'end_dist_sideline' and 'end_dist_goalline' of each action.
    """
    actions["_end_dist_sideline"] = config._spadl_cfg["width"] - actions["end_y"]
    actions["end_dist_sideline"] = actions[["_end_dist_sideline", "end_y"]].min(axis=1)
    actions["end_dist_goalline"] = config._spadl_cfg["length"] - actions["end_x"]
    return actions[["end_dist_sideline", "end_dist_goalline"]]

@required_fields(["start_x", "start_y", "end_x", "end_y"])
@simple
def angle(actions):
    """Get the angle between the start and end location of an action.

    The action's start location is used as the origin in a polar coordinate
    system with the polar axis parallell to the the goalline.

    Parameters
    ----------
    actions : SPADLActions
        The actions of a game.

    Returns
    -------
    Features
        The 'angle' of each action.
    """
    polardf = pd.DataFrame(index=actions.index)
    dx = (actions["end_x"] - actions["start_x"]).values
    dy = (actions["end_y"] - actions["start_y"]).values
    with np.errstate(divide="ignore", invalid="ignore"):
        polardf["angle"] = np.nan_to_num(np.arctan(dy / dx))
    polardf.loc[actions["end_x"] < actions["start_x"], "angle"] += np.pi
    return polardf


@required_fields(["start_x", "start_y", "end_x", "end_y", "time_seconds"])
def speed(gamestates):
    """Get the speed at which the ball moved during the previous actions.

    Parameters
    ----------
    gamestates : GameStates
        The game states of a game.

    Returns
    -------
    Features
        A dataframe with columns 'speedx_a0i', 'speedy_a0i', 'speed_a0i'
        for each <nb_prev_actions> containing the ball speed in m/s  between
        action ai and action a0.
    """
    a0 = gamestates[0]
    speed = pd.DataFrame(index=a0.index)
    for i, a in enumerate(gamestates[1:]):
        dx = a.end_x - a0.start_x
        dy = a.end_y - a0.start_y
        dt = a0.time_seconds - a.time_seconds
        dt[dt <= 0] = 1e-6
        speed["speedx_a0" + (str(i + 1))] = dx.abs() / dt
        speed["speedy_a0" + (str(i + 1))] = dy.abs() / dt
        speed["speed_a0" + (str(i + 1))] = np.sqrt(dx**2 + dy**2) / dt
    return speed

@required_fields(["period_id", "time_seconds", "player_id", "type_name"])
@simple
def player_possession_time(actions):
    """Get the time (sec) a player was in ball possession before attempting the action.

    We only look at the dribble preceding the action and reset the possession
    time after a defensive interception attempt or a take-on.

    Parameters
    ----------
    actions : SPADLActions
        The actions of a game.

    Returns
    -------
    Features
        The 'player_possession_time' of each action.
    """
    cur_action = actions[["period_id", "time_seconds", "player_id", "type_name"]]
    prev_action = actions.shift(1)[["period_id", "time_seconds", "player_id", "type_name"]]
    df = cur_action.join(prev_action, rsuffix="_prev")
    same_player = df.player_id == df.player_id_prev
    same_period = df.period_id == df.period_id_prev
    prev_dribble = df.type_name_prev == "dribble"
    mask = same_period & same_player & prev_dribble
    df.loc[mask, "player_possession_time"] = (
        df.loc[mask, "time_seconds"] - df.loc[mask, "time_seconds_prev"]
    )
    return df[["player_possession_time"]].fillna(0)

# CONTEXT FEATURES: The game context features
@required_fields([["team_id", "type_name", "result_id"]])
def goalscore(gamestates):
    """Get the number of goals scored by each team after the action.

    Parameters
    ----------
    gamestates : pd.DataFrame
        The gamestates of a game.

    Returns
    -------
    pd.DataFrame
        The number of goals scored by the team performing the last action of the
        game state ('goalscore_team'), by the opponent ('goalscore_opponent'),
        and the goal difference between both teams ('goalscore_diff').
    """
    actions = gamestates[0]
    teamA = actions['team_id'].values[0]
    goals = actions['type_name'].str.contains('shot') & (
        actions['result_id'] == config.results.index('success')
    )
    owngoals = (actions['result_id'] == config.results.index('owngoal')) # owngoal'type_name is bad_touch
    
    teamisA = actions['team_id'] == teamA
    teamisB = ~teamisA
    goalsteamA = (goals & teamisA) | (owngoals & teamisB)
    goalsteamB = (goals & teamisB) | (owngoals & teamisA)
    goalscoreteamA = goalsteamA.cumsum() - goalsteamA
    goalscoreteamB = goalsteamB.cumsum() - goalsteamB

    scoredf = pd.DataFrame()
    scoredf['goalscore_team'] = (goalscoreteamA * teamisA) + (goalscoreteamB * teamisB)
    scoredf['goalscore_opponent'] = (goalscoreteamB * teamisA) + (goalscoreteamA * teamisB)
    scoredf['goalscore_diff'] = scoredf['goalscore_team'] - scoredf['goalscore_opponent']
    return scoredf

# 360 FEATURES
def _opponents_in_radius(actions, radius: int = 1):
    """Get the number of opponent(pressing당하는 선수: 공격수) in a radius around the actions start and end location.

    Parameters
    ----------
    actions : SPADLActions
        The actions of a game.
    radius : int
        The radius (in meters) of the circle in which defenders are counted.

    Returns
    -------
    Features
        The number of defenders is a radius around the actions start and end location.
    """
    defenders_in_radius = np.full((len(actions), 2), np.nan, dtype=float) 
    for i, (_, action) in enumerate(actions.iterrows()):
        if not action["freeze_frame_360"]:
            continue

        opponents_coo = np.array(
            [(o["x"], o["y"]) for o in action["freeze_frame_360"] if not o["teammate"]]
        )
        if len(opponents_coo) == 0:
            continue

        start_coo = np.array([action["start_x"], action["start_y"]])
        end_coo = np.array([action["end_x"], action["end_y"]])

        # Distance to start location
        dist_defender_start = np.sqrt(
            (opponents_coo[:, 0] - start_coo[0]) ** 2 + (opponents_coo[:, 1] - start_coo[1]) ** 2
        )
        defenders_in_radius[i, 0] = np.sum(dist_defender_start <= radius)

        # Distance to end location
        dist_defender_end = np.sqrt(
            (opponents_coo[:, 0] - end_coo[0]) ** 2 + (opponents_coo[:, 1] - end_coo[1]) ** 2
        )
        defenders_in_radius[i, 1] = np.sum(dist_defender_end <= radius)

    return pd.DataFrame(
        defenders_in_radius,
        index=actions.index,
        columns=[f"nb_opponents_start_{radius}m", f"nb_opponents_end_{radius}m"],
    )

@required_fields(["freeze_frame_360", "start_x", "start_y", "end_x", "end_y"])
@simple
def dist_opponent(actions):
    """Get the distance to the nearest defender.

    Parameters
    ----------
    actions : SPADLActions
        The actions of a game.

    Returns
    -------
    Features
        The distance to the nearest defender at the start and end location of the action,
        and along the action's path.
    """

    dist = np.full((len(actions), 3), np.nan, dtype=float) 
    for i, (_, action) in enumerate(actions.iterrows()):
        if not action["freeze_frame_360"]:
            continue

        opponents_coo = np.array([(o["x"], o["y"]) for o in action["freeze_frame_360"] if not o["teammate"]])
        if len(opponents_coo) == 0:
            continue

        start_coo = np.array([action["start_x"], action["start_y"]])
        end_coo = np.array([action["end_x"], action["end_y"]])

        # Distance to start location
        dist[i, 0] = np.amin(
            np.sqrt(
                (opponents_coo[:, 0] - start_coo[0]) ** 2
                + (opponents_coo[:, 1] - start_coo[1]) ** 2
            )
        )

        # Distance to end location
        dist[i, 1] = np.amin(
            np.sqrt(
                (opponents_coo[:, 0] - end_coo[0]) ** 2 + (opponents_coo[:, 1] - end_coo[1]) ** 2
            )
        )

        # Distance to action path
        # normalized tangent vector
        if (start_coo == end_coo).all():
            dist[i, 2] = dist[i, 0]
        else:
            d = np.divide(end_coo - start_coo, np.linalg.norm(end_coo - start_coo))
            # signed parallel distance components
            s = np.dot(start_coo - opponents_coo, d)
            t = np.dot(opponents_coo - end_coo, d)
            # clamped parallel distance
            h = np.maximum.reduce([s, t, np.zeros(len(opponents_coo))])
            # perpendicular distance component, as before
            # note that for the 3D case these will be vectors
            c = np.cross(opponents_coo - start_coo, d)
            # use hypot for Pythagoras to improve accuracy
            dist[i, 2] = np.amin(np.hypot(h, c))

    return pd.DataFrame(
        dist,
        index=actions.index,
        columns=["dist_opponent_start", "dist_opponent_end", "dist_opponent_action"],
    )

def closest_players(actions, num_players=3):
    """For each action, find the n closest teammates and n closest opponents to the action's location.

    Parameters
    ----------
    actions : SPADLActions
        The actions of a game.
    num_players (int): Number of closest teammates and opponents to find for each action.

    Returns:
    - DataFrame: Contains x, y, and distance for each of the closest teammates and opponents.
    """
    # Get the n closest teammates and opponents
    num_teammates = num_players if num_players < 11 else 10 # Exclude event player
    num_opponents = num_players
    cloeset_teammates_xy = np.full((len(actions), num_teammates*3), np.nan, dtype=float) # x, y, distance for each teammate player
    cloeset_opponents_xy = np.full((len(actions), num_opponents*3), np.nan, dtype=float) # x, y, distance for each opponent player
    
    for i, (_, action) in enumerate(actions.iterrows()):
        if not action["freeze_frame_360"]:
            continue
        
        freeze_frame = pd.DataFrame.from_records(action["freeze_frame_360"])
        start_x, start_y = action.start_x, action.start_y
        teammate_locs = freeze_frame[freeze_frame.teammate & ~freeze_frame.actor].copy() # Exclude event player
        opponent_locs = freeze_frame[~freeze_frame.teammate].copy()

        teammate_locs['distance'] = np.hypot(teammate_locs.x - start_x, teammate_locs.y - start_y)
        opponent_locs['distance'] = np.hypot(opponent_locs.x - start_x, opponent_locs.y - start_y)

        closest_teammates_df = teammate_locs.nsmallest(num_teammates, 'distance')[['x', 'y', 'distance']]
        closest_opponents_df = opponent_locs.nsmallest(num_opponents, 'distance')[['x', 'y', 'distance']]

        for j in range(len(closest_teammates_df)):
            cloeset_teammates_xy[i, j * 3] = closest_teammates_df.iloc[j]['x']
            cloeset_teammates_xy[i, j * 3 + 1] = closest_teammates_df.iloc[j]['y']
            cloeset_teammates_xy[i, j * 3 + 2] = closest_teammates_df.iloc[j]['distance']

        for j in range(len(closest_opponents_df)):
            cloeset_opponents_xy[i, j * 3] = closest_opponents_df.iloc[j]['x']
            cloeset_opponents_xy[i, j * 3 + 1] = closest_opponents_df.iloc[j]['y']
            cloeset_opponents_xy[i, j * 3 + 2] = closest_opponents_df.iloc[j]['distance']

    # Combine teammate and opponent data
    cloeset_xy = np.hstack((cloeset_teammates_xy, cloeset_opponents_xy))

    # Generate column names dynamically
    columns = []
    for k in range(1, num_teammates + 1):
        columns.extend([f"teammate_{k}_x", f"teammate_{k}_y", f"teammate_{k}_distance"])
    for k in range(1, num_opponents + 1):
        columns.extend([f"opponent_{k}_x", f"opponent_{k}_y", f"opponent_{k}_distance"])

    return pd.DataFrame(cloeset_xy, index=actions.index, columns=columns)

_opponents_in_3m_radius = required_fields(["start_x", "start_y", "end_x", "end_y", "freeze_frame_360"]
)(simple(partial(_opponents_in_radius, radius=3))) # parameter: radius=3
_opponents_in_3m_radius.__name__ = "_opponents_in_3m_radius"

closest_3_players = required_fields(["freeze_frame_360", "start_x", "start_y"]
)(simple(partial(closest_players, num_players=3)))  # parameter: num_players=3
closest_3_players.__name__ = "closest_3_players"

closest_11_players = required_fields(["freeze_frame_360", "start_x", "start_y"]
)(simple(partial(closest_players, num_players=11)))  # parameter: num_players=11
closest_11_players.__name__ = "closest_11_players"

all_features = [
    # SPADL FEATURES
    actiontype,
    # actiontype_onehot,
    result,
    # result_onehot,
    bodypart,
    # bodypart_onehot,
    time,
    startlocation,
    endlocation,

    # COMPLEXT FEATURES
    startpolar,
    endpolar,
    movement,
    team,
    time_delta,
    space_delta,
    relative_startlocation,
    relative_endlocation,
    angle,
    speed,
    player_possession_time,

    # CONTEXT FEATURES
    goalscore,

    # 360 FEATURES
    _opponents_in_3m_radius,
    dist_opponent,
    closest_11_players,
]

def feature_column_names(fs: List[Callable], nb_prev_actions: int = 3) -> List[str]:
    """Return the names of the features generated by a list of transformers.

    Parameters
    ----------
    fs : list(callable)
        A list of feature transformers.
    nb_prev_actions : int, default=3
        The number of previous actions included in the game state.

    Returns
    -------
    list(str)
        The name of each generated feature.
    """

    cols = TABLE_ACTIONS + ["type_name", "result_name", "bodypart_name"]
    dummy_actions = pd.DataFrame(np.zeros((10, len(cols))), columns=cols).set_index(
        ["game_id", "action_id"]
    )

    for c in cols:
        if "name" in c:
            dummy_actions[c] = dummy_actions[c].astype(str)
    
    gs = gamestates(dummy_actions, nb_prev_actions)  # S_i = {a_i-2, a_i-1, a_i}

    return list(pd.concat([f(gs) for f in fs], axis=1).columns.values)

def get_features(
    db: Database,
    game_id: int,
    xfns: List[Callable] = all_features,
    actionfilter: Optional[Callable] = None,
    nb_prev_actions: int = 3,
):
    """Apply a list of state feature generators.

    Parameters
    ----------
    db : Database
        The database with raw data.
    game_id : int
        The ID of the game for which features should be computed.
    xfns : List[Callable], optional
        The feature generators.
    actionfilter : Callable, optional
        A function that filters the actions to be used.
    nb_prev_actions : int, optional
        The number of previous actions to be included in a game state.

    Returns
    -------
    pd.DataFrame
        A dataframe with the features.
    """
    # retrieve actions from database
    
    actions = add_names(db.actions(game_id))

    # 압박을 제외한 이벤트로 SPALD features, Complex Features, Game context Features
    actions = actions[actions["type_id"] != config.actiontypes.index("pressing")]

    # filter pressing actions of interest
    if actionfilter is None:
        idx = pd.Series([True] * len(actions), index=actions.index)
    else:
        idx = actionfilter(actions)

    # check if we have to return an empty dataframe
    if idx.sum() < 1:
        raise ValueError("No data left after actionfilter; returning an empty DataFrame with specified columns.")
    
    # convert actions to gamestates
    home_team_id, _ = db.get_home_away_team_id(game_id)

    # 기존 위치 좌표: _fix_direction_of_play로 인해 홈팀은 left->right, 원정팀은 left<-right로 진행되는 상태
    # play_left_to_right: 모든 팀의 공격 방향이 left->right로 수정합니다.(원정팀만 flip진행하면 됨)
    states = play_left_to_right(gamestates(actions, nb_prev_actions), home_team_id) # {S_i}
    states = [state.loc[idx].copy() for state in states] # 압박을 당한 액션(a_i)에 한정하여 feature engineering수행: faster!
    
    # compute features: STATE를 활용한 feature engineering
    if not xfns:
        df_features = pd.DataFrame(index=states[0].index)
    else:
        # reduce: iterable object내 각 요소(dataframe)를 연산한 뒤 이전 연산 결과들과 누적해서 반환해 주는 함수
        df_features = reduce(
            lambda left, right: pd.merge(left, right, how="outer", left_index=True, right_index=True),
            (fn(states) for fn in xfns),
        )

    return df_features