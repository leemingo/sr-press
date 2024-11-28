"""Implements the pressure features used in exPressComponent."""
import os
import sys
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(base_path)

from functools import partial, reduce, wraps
from typing import Callable, Dict, List, Optional, Union, no_type_check

import numpy as np
import pandas as pd

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
            Xi.columns = [c + "_p" + str(i) for c in Xi.columns] # rename '_a' to '_p'
            X.append(Xi)
        return pd.concat(X, axis=1)

    return _wrapper

# SPADL FEATURES

@required_fields(["period_id", "time_seconds"])
@simple
def time(actions):
    """Get the time when a pressing action was performed.

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
        A pressing action of a game.

    Returns
    -------
    pd.DataFrame
        The 'period_id', 'time_seconds' and 'time_seconds_overall' when each
        action was performed.
    """
    timedf = actions[['period_id', 'time_seconds']].copy() # p_i
    timedf['time_seconds_overall'] = ((timedf.period_id - 1) * 45 * 60) + timedf.time_seconds
    return timedf

@required_fields(["start_x", "start_y"])
@simple
def startlocation(actions):
    """Get the location where a pressing action started.

    Parameters
    ----------
    actions : pd.DataFrame
        A pressing action of a game.

    Returnss
    -------
    pd.DataFrame
        The 'start_x' and 'start_y' location of each action.
    """
    return actions[['start_x', 'start_y']] # p_i

# Complex FEATURES: The complex features combine information within an action and across consecutive actions.

@required_fields(["start_x", "start_y"])
@simple
def startpolar(actions):
    """Get the polar coordinates of a pressing action's start location.

    The center of the opponent's goal is used as the origin.

    Parameters
    ----------
    actions : pd.DataFrame
        A pressing action of a game.

    Returns
    -------
    pd.DataFrame
        The 'start_dist_to_goal' and 'start_angle_to_goal' of each action.
    """
    polardf = pd.DataFrame() # p_i
    dx = abs(config._goal_x - actions['start_x'])
    dy = abs(config._goal_y - actions['start_y'])
    polardf['start_dist_to_goal'] = np.sqrt(dx ** 2 + dy ** 2)
    with np.errstate(divide='ignore', invalid='ignore'):
        polardf['start_angle_to_goal'] = np.nan_to_num(np.arctan(dy / dx))
    return polardf

@required_fields(["time_seconds"])
def time_delta(gamestates):
    """Get the number of seconds between each action and the pressing action
    Parameters
    ----------
    gamestates : pd.DataFrame
        The game states of a game.

    Returns
    -------
    pd.DataFrame
        A dataframe with a column 'time_delta_pi' for each <nb_prev_actions>
        containing the number of seconds between action ai and a pressing action p0.
    """
    p0 = gamestates[0] # p_i
    dt = pd.DataFrame()
    for i, a in enumerate(gamestates[1:]): # gamestates[1:] = {a_i}
        dt['time_delta_p' + (str(i))] = p0.time_seconds - a.time_seconds
    return dt

@required_fields([["start_x", "start_y", "end_x", "end_y"]])
def space_delta(gamestates):
    """Get the distance covered between each action and the pressing action

    Parameters
    ----------
    gamestates : pd.DataFrame
        The gamestates of a game.

    Returns
    -------
    pd.DataFrame
        A dataframe with a column for the horizontal ('dx_p0i'), vertical
        ('dy_a0i') and total ('mov_p0i') distance covered between each
        <nb_prev_actions> action ai and a pressing action p0.
    """
    p0 = gamestates[0] # p_i
    spaced = pd.DataFrame()
    for i, a in enumerate(gamestates[1:]): # gamestates[1:] = {a_i}
        dx = a.end_x - p0.start_x
        spaced['dx_p' + (str(i))] = dx
        dy = a.end_y - p0.start_y
        spaced['dy_p' + (str(i))] = dy
        spaced['mov_p' + (str(i))] = np.sqrt(dx ** 2 + dy ** 2)
    return spaced

@required_fields(["start_x", "start_y"])
@simple
def relative_startlocation(actions):
    """Get the location where each pressure action started relative to the sideline and goalline.

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


@required_fields(["start_x", "start_y"])
def relative_defender_angle(gamestates):
    """Get the angle between the defender’s goal, the ball and the defender.

    Parameters
    ----------
    actions : SPADLActions
        The actions of a game.

    Returns
    -------
    Features
        The 'start_dist_sideline' and 'start_dist_goalline' of each action.
    """
    p0 = gamestates[0] # {p_i}
    polardf = pd.DataFrame()
    for i, a in enumerate(gamestates[1:]): # gamestates[1:] = {a_i}
        defender = np.stack([p0.start_x.values, p0.start_y.values], axis=1)  
        ball_coo = np.stack([a.start_x.values, a.start_y.values], axis=1)  
        defender_goal = np.array([0, config._goal_y])

        a = defender - defender_goal
        b = ball_coo - defender_goal

        angle = np.arccos(
            np.clip(
                np.sum(a * b, axis=1) / (np.linalg.norm(a) * np.linalg.norm(b, axis=1)), -1, 1
            )
        )

        polardf['start_angle_goalline_p' + (str(i))] = angle

    return polardf

# 360 FEATURES
@required_fields(["start_x", "start_y", "end_x", "end_y", "freeze_frame_360"])
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
        columns=[f"nb_defenders_start_{radius}m", f"nb_defenders_end_{radius}m"],
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
        columns=["dist_defender_start", "dist_defender_end", "dist_defender_action"],
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

@required_fields(["freeze_frame_360", "start_x", "start_y"])
def expected_receiver_and_presser_by_distance(actions, min_players=3):    
    distances = np.full((len(actions), min_players*2), np.nan, dtype=float)
    angles = np.full((len(actions), min_players*2), np.nan, dtype=float)

    for i, action in enumerate(actions.itertuples(index=False)):
        if not action.freeze_frame_360:
            continue
        
        freeze_frame = pd.DataFrame.from_records(action.freeze_frame_360)
        start_x, start_y = action.start_x, action.start_y
        teammate_locs = freeze_frame[freeze_frame.teammate & ~freeze_frame.actor & ~freeze_frame.keeper].reset_index(drop=True) # Exclude event player
        opponent_locs = freeze_frame[~freeze_frame.teammate & ~freeze_frame.keeper].reset_index(drop=True)

        if (len(teammate_locs) < min_players) or (len(opponent_locs) < min_players):
            continue
        
        # Calculate the closest opponent to the start location(presser)
        # np.hypot: Calculate Euclidean distance
        dist_presser_to_target = np.hypot(opponent_locs.x - start_x, opponent_locs.y - start_y)
        target_idx = np.argmin(dist_presser_to_target)
        target_x, target_y = opponent_locs.loc[target_idx, ["x", "y"]].values
       
        # expected-receiver
        dists_to_target = np.hypot(opponent_locs.x - target_x, opponent_locs.y - target_y)
        expected_receiver_idxs = np.argsort(dists_to_target)[:min_players] # Include the target itself
        expected_receivers_locs = opponent_locs.loc[expected_receiver_idxs, ["x", "y"]].values

        # expected-presser
        expected_pressers_idx = []
        for expected_receiver_idx in expected_receiver_idxs:
            receiver_x, receiver_y = opponent_locs.loc[expected_receiver_idx, ["x", "y"]].values

            # Note: Multiple pressers may target the same receiver.
            dists_to_receiver = np.hypot(teammate_locs.x - receiver_x, teammate_locs.y - receiver_y)

            presser_idx = np.argmin(dists_to_receiver)
            expected_pressers_idx.append(presser_idx) 
        expected_presser_locs = teammate_locs.iloc[expected_pressers_idx][["x", "y"]].values

        # Compute distances and angles between presser and expected receivers
        distances[i, :min_players] = np.hypot(expected_receivers_locs[:, 0] - start_x, expected_receivers_locs[:, 1] - start_y)
        angles[i, :min_players] = np.arctan2(expected_receivers_locs[:, 0] - start_x, expected_receivers_locs[:, 1] - start_y)

        # Compute distances and angles between presser and expected presser
        distances[i, min_players:] = np.hypot(expected_presser_locs[:, 0] - start_x, expected_presser_locs[:, 1] - start_y)
        angles[i, min_players:] = np.arctan2(expected_presser_locs[:, 0] - start_x, expected_presser_locs[:, 1] - start_y)

    distance_columns = (
        [f"distance_to_expected_receiver{idx}" for idx in range(1, min_players+1)]
        + [f"distance_to_expected_presser{idx}" for idx in range(1, min_players+1)]
    )
    angle_columns = (
        [f"angle_to_expected_receiver{idx}" for idx in range(1, min_players+1)]
        + [f"angle_to_expected_presser{idx}" for idx in range(1, min_players+1)]
    )

    # Create DataFrames for distances and angles
    distances_df = pd.DataFrame(distances, index=actions.index, columns=distance_columns)
    angles_df = pd.DataFrame(angles, index=actions.index, columns=angle_columns)

    return pd.concat([distances_df, angles_df], axis=1)

@required_fields(["freeze_frame_360", "start_x", "start_y"])
def expected_receiver_and_presser_by_distance_with_context(gamestates, min_players=3):    
    p0 = gamestates[0] # p_i
    state = gamestates[1:] # S_i = {a_i-2, a_i-1, a_i}

    # dist_angle_df = pd.DataFrame(index=p0.index)
    coord_df = pd.DataFrame(index=p0.index)
    for i, actions in enumerate(state): # S_i = {a_i-2, a_i-1, a_i}
        # distances = np.full((len(p0), min_players*2), np.nan, dtype=float)
        # angles = np.full((len(p0), min_players*2), np.nan, dtype=float)
        expected_presser_coord = np.full((len(p0), min_players*2), np.nan, dtype=float)
        expected_receiver_coord = np.full((len(p0), min_players*2), np.nan, dtype=float)

        for j, action in enumerate(actions.itertuples(index=True)):
            freeze_frame_360 = p0.loc[action.Index, "freeze_frame_360"]

            if not isinstance(freeze_frame_360, list) or not freeze_frame_360:
                continue

            freeze_frame = pd.DataFrame.from_records(freeze_frame_360)
            teammate_locs = freeze_frame[freeze_frame.teammate & ~freeze_frame.actor & ~freeze_frame.keeper].reset_index(drop=True) # Exclude event player
            opponent_locs = freeze_frame[~freeze_frame.teammate & ~freeze_frame.keeper].reset_index(drop=True)

            start_coo = np.array([action.start_x, action.start_y])
            end_coo = np.array([action.end_x, action.end_y])

            if (len(teammate_locs) < min_players) or (len(opponent_locs) < min_players):
                continue
            
            # Calculate the closest opponent(event player과 동일한 팀) to the start location(event player)
            # np.hypot: Calculate Euclidean distance
            dist_teammatess = np.hypot(teammate_locs.x - end_coo[0], teammate_locs.y - end_coo[1])
            dist_opponents = np.hypot(opponent_locs.x - end_coo[0], opponent_locs.y - end_coo[1])

            # compute the angle between each potential receiver and the passing line
            player_vec = end_coo - start_coo
            player_to_teammatess_vec = teammate_locs[["x", "y"]].values - start_coo
            player_to_opponents_vec = opponent_locs[["x", "y"]].values - start_coo
            angle_teammates = np.arccos(
                np.clip(
                    np.sum(player_vec * player_to_teammatess_vec, axis=1) /
                    (np.linalg.norm(player_vec) * np.linalg.norm(player_to_teammatess_vec, axis=1)),
                    -1, 1
                )
            )
            angle_opponents = np.arccos(
                np.clip(
                    np.sum(player_vec * player_to_opponents_vec, axis=1) /
                    (np.linalg.norm(player_vec) * np.linalg.norm(player_to_opponents_vec, axis=1)),
                    -1, 1
                )
            )

            expected_presser_idxs = np.argsort((np.amin(dist_teammatess) / dist_teammatess) * (np.amin(angle_teammates) / angle_teammates))[:min_players]
            expected_receiver_idxs = np.argsort((np.amin(dist_opponents) / dist_opponents) * (np.amin(angle_opponents) / angle_opponents))[:min_players]

            expected_presser_coord[j, :]  = teammate_locs.loc[expected_presser_idxs.tolist(),["x", "y"]].values
            expected_receiver_coord[j, :] = opponent_locs.loc[expected_receiver_idxs.tolist(),["x", "y"]].values

        expected_presser_columns = [f"expected_presser_p{str(i)}_{axis}" for i in range(1, min_players+1) for axis in ["x", "y"]]
        expected_receiver_columns = [f"expected_receiver_p{str(i)}_{axis}" for i in range(1, min_players+1) for axis in ["x", "y"]]

        # Create DataFrames for distances and angles
        expected_presser_coord = pd.DataFrame(expected_presser_coord, index=actions.index, columns=expected_presser_columns)
        expected_receiver_coord = pd.DataFrame(expected_receiver_coord, index=actions.index, columns=expected_receiver_columns)

        coord_df = pd.concat([coord_df, expected_presser_coord, expected_receiver_coord], axis=1)

    return coord_df

_opponents_in_3m_radius = required_fields(["start_x", "start_y", "end_x", "end_y", "freeze_frame_360"])(
    simple(partial(_opponents_in_radius, radius=3))) # parameter: radius=3
_opponents_in_3m_radius.__name__ = "_opponents_in_3m_radius"

closest_11_players = required_fields(["freeze_frame_360", "start_x", "start_y"]
)(simple(partial(closest_players, num_players=11)))  # parameter: num_players=11
closest_11_players.__name__ = "closest_11_players"

expected_3_receiver_and_presser_by_distance = required_fields(["freeze_frame_360", "start_x", "start_y"])(
   simple(partial(expected_receiver_and_presser_by_distance, min_players=3)))  # parameter: num_players=3
expected_3_receiver_and_presser_by_distance.__name__ = "expected_3_receiver_and_presser_by_distance"

expected_3_receiver_and_presser_by_distance_with_context = required_fields(["freeze_frame_360", "start_x", "start_y"])(
   partial(expected_receiver_and_presser_by_distance_with_context, min_players=3))  # parameter: num_players=3
expected_3_receiver_and_presser_by_distance_with_context.__name__ = "expected_3_receiver_and_presser_by_distance_with_context"

pressure_state_features = [
    # SPADL FEATURES
    # time,
    # startlocation,

    # COMPLEX FEATURES
    # startpolar,
    # relative_startlocation,

    # 360 FEATURES
    _opponents_in_3m_radius,
    dist_opponent,
    closest_11_players,

]

pressure_with_context_features = [
    # (1) the distance between the defender and the ball,
    space_delta, 

    # (2) the angle between the defender’s goal, the ball and the defender 
    relative_defender_angle,

    # (3) the time delay between the start of the action and the start of the pressure.
    time_delta,
]

def feature_column_names_pressure_state(fs: List[Callable], nb_prev_actions: int = 3) -> List[str]:
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
    
    gs = gamestates(dummy_actions, nb_prev_actions) # p_i

    return list(pd.concat([f(gs) for f in fs], axis=1).columns.values)

def feature_column_names_pressure_context(fs: List[Callable], nb_prev_actions: int = 3) -> List[str]:
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
    
    ps = gamestates(dummy_actions, 1) # p_i
    gs = gamestates(dummy_actions, nb_prev_actions) # a_i
    combined_states = ps + gs # {p_i, a_i}

    return list(pd.concat([f(combined_states) for f in fs], axis=1).columns.values)

def get_features(
    db: Database,
    game_id: int,
    xfns: Dict[Union[str, Callable], Optional[List]],
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
    
    actions = add_names(db.actions(game_id)).reset_index()

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

    states = []
    pressure_states = []
    for (game_id, period_id), group in actions.groupby(["game_id", "period_id"], sort=False, group_keys=False):
        # iterrows(): converts all data types to 'object' when iterating over rows, which leads to loss of original data types(convert int64 to object)
        # Note: using 'itertuples' is faster and maintains data types
        for action in group.itertuples(index=False):
            # 압박 당하는 액션(a_i)를 기준으로 병합
            if (action.type_name == "pressing"): 
                continue

            # a_i액션을 수행한 선수를 압박하는 액션(p_i)을 조사하여 a_i를 압박하는 이벤트인 p_i = {p_i1, p_i2..}과 같이 존재하면
            # {a_i, p_{i1} }, {a_i, p_{i2} }과 같이 분해한다.
            start_time = action.time_seconds
            end_time = start_time + action.duration
            team_id = action.team_id

            # overlap1: if a pressure event appears before a pass, and the pressure’s timestamp plus its duration encompasses the pass’s timestamp, that pass is said to have been made under pressure
            # overlap2: if a pressure event occurs after a pass, but before the end of the pass (as calculated by using its duration), that pass is said to have been received under pressure.
            overlap_cond1 = (actions['time_seconds'] <= start_time) & ((actions['time_seconds'] + actions['duration']) >= start_time)
            overlap_cond2 = (actions['time_seconds'] >= start_time) & ((actions['time_seconds'] + actions['duration']) <= end_time)
            overlap = (
                (actions["game_id"] == game_id) 
                & (actions["period_id"] == period_id) 
                & (actions["team_id"] != team_id) # presser'team != 압박당하는 선수'team
                & overlap_cond1
            )

            pressing_actions = actions[overlap].loc[idx]

            for pressing_action in pressing_actions.itertuples(index=False):       
                states.append(action) 
                pressure_states.append(pressing_action)

    # states: 압박을 당하는 액션(a_i)는 여러 선수가 압박(p_i={p_i1, p_i2...})를 할 수 있기 때문에 동일한 states가 존재할 수 있음
    # ex) {S_i, p_i1}, {S_i, p_i2}
    states = pd.DataFrame.from_records([s._asdict() for s in states]).reset_index(drop=True)
    pressure_states = pd.DataFrame.from_records([p._asdict() for p in pressure_states]).reset_index(drop=True)

    # 기존 위치 좌표: _fix_direction_of_play로 인해 홈팀은 left->right, 원정팀은 left<-right로 진행되는 상태
    # play_left_to_right: 모든 팀의 공격 방향이 left->right로 수정합니다.(원정팀만 flip진행하면 됨)
    states = play_left_to_right(gamestates(states, nb_prev_actions), home_team_id)     # a_i
    pressure_states = play_left_to_right(gamestates(pressure_states, 1), home_team_id) # p_i
    combined_states = pressure_states + states # {p_i, S_i} = {p_i, a_i}
    
    # compute features: Pressure STATE를 활용한 feature engineering
    if not xfns["pressure_state_xfns"]:
        pressure_states_features = pd.DataFrame(index=states[0].index)
    else:
        # reduce: iterable object내 각 요소(dataframe)를 연산한 뒤 이전 연산 결과들과 누적해서 반환해 주는 함수
        pressure_states_features = reduce(
            lambda left, right: pd.merge(left, right, how="outer", left_index=True, right_index=True),
            (fn(pressure_states) for fn in xfns["pressure_state_xfns"]),
        )

    # compute features: STATE과 Pressure STATE를 활용한 feature engineering
    if not xfns["pressure_with_context_xfns"]:
        combined_states_features = pd.DataFrame(index=states[0].index)
    else:
        # reduce: iterable object내 각 요소(dataframe)를 연산한 뒤 이전 연산 결과들과 누적해서 반환해 주는 함수
        combined_states_features = reduce(
            lambda left, right: pd.merge(left, right, how="outer", left_index=True, right_index=True),
            (fn(combined_states) for fn in xfns["pressure_with_context_xfns"]),
        )

    df_features = pd.concat([pressure_states_features, combined_states_features], axis=1, ignore_index=False)
    # 중복 인덱스도 부여 가능: 여러 선수가 압박을 한 상황인 {a_i, p_i1}, {a_i, p_i2}, {a_i, p_i3}..모두 동일한 index(game_id, action)를 가짐
    # S_i = {a_i-2, a_i-1, a_i}에 대한 feautre는 동일하나, p_i에 대한 feature는 각기 다름
    df_features.index = states[0].set_index(["game_id", "action_id"]).index 

    return df_features