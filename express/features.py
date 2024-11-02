"""Implements the features used in exPressComponent."""
import os
import sys
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(base_path)

import math
from functools import partial, reduce
from typing import Callable, List, Optional

import numpy as np
import pandas as pd
import socceraction.vaep.features as fs
from socceraction.vaep.features import gamestates as to_gamestates

import express.config as config
from express.databases import Database
from express.databases.base import TABLE_ACTIONS
from express.utils import add_names, play_left_to_right

_spadl_cfg = {
    "length": 105,
    "width": 68,
    "penalty_box_length": 16.5,
    "penalty_box_width": 40.3,
    "six_yard_box_length": 5.5,
    "six_yard_box_width": 18.3,
    "goal_width": 7.32,
    "penalty_spot_distance": 11,
    "goal_length": 2,
    "origin_x": 0,
    "origin_y": 0,
    "circle_radius": 9.15,
}

_goal_x: float = _spadl_cfg["length"]
_goal_y: float = _spadl_cfg["width"] / 2

_pass_like = [
    "pass",
    "cross",
    "throw_in",
    "freekick_crossed",
    "freekick_short",
    "corner_crossed",
    "corner_short",
    "goalkick",
]

def required_fields(fields):
    def decorator(func):
        func.required_fields = fields
        return func

    return decorator

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

    gs = to_gamestates(dummy_actions, nb_prev_actions)  # type: ignore

    return list(pd.concat([f(gs) for f in fs], axis=1).columns.values)

@required_fields(["type_name"])
@fs.simple
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

@required_fields(["result_name"])
@fs.simple
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

@required_fields(["bodypart_name"])
@fs.simple
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
@fs.simple
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
@fs.simple
def startlocation(actions):
    """Get the location where each action started.

    Parameters
    ----------
    actions : pd.DataFrame
        The actions of a game.

    Returns
    -------
    pd.DataFrame
        The 'start_x' and 'start_y' location of each action.
    """
    return actions[['start_x', 'start_y']]


@required_fields(["end_x", "end_y"])
@fs.simple
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

@required_fields(["start_x", "start_y"])
@fs.simple
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
    dx = abs(_goal_x - actions['start_x'])
    dy = abs(_goal_y - actions['start_y'])
    polardf['start_dist_to_goal'] = np.sqrt(dx ** 2 + dy ** 2)
    with np.errstate(divide='ignore', invalid='ignore'):
        polardf['start_angle_to_goal'] = np.nan_to_num(np.arctan(dy / dx))
    return polardf


@required_fields(["end_x", "end_y"])
@fs.simple
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
    dx = abs(_goal_x - actions['end_x'])
    dy = abs(_goal_y - actions['end_y'])
    polardf['end_dist_to_goal'] = np.sqrt(dx ** 2 + dy ** 2)
    with np.errstate(divide='ignore', invalid='ignore'):
        polardf['end_angle_to_goal'] = np.nan_to_num(np.arctan(dy / dx))
    return polardf

@required_fields(["start_x", "start_y", "end_x", "end_y"])
@fs.simple
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
# @fs.simple: STATE FEATURES
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
        teamdf['team_' + (str(i + 1))] = a.team_id == a0.team_id
    return teamdf

@required_fields(["time_seconds"])
#@fs.simple: STATE FEATURES
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
        dt['time_delta_' + (str(i + 1))] = a0.time_seconds - a.time_seconds
    return dt

@required_fields([["start_x", "start_y", "end_x", "end_y"]])
#@fs.simple: : STATE FEATURES
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

@required_fields([["team_id", "type_name", "result_id"]])
#@fs.simple: STATE FEATURES
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
    owngoals = actions['type_name'].str.contains('shot') & (
        actions['result_id'] == config.results.index('owngoal')
    )
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

@required_fields(["start_x", "start_y"])
@fs.simple
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
    actions["_start_dist_sideline"] = _spadl_cfg["width"] - actions["start_y"]
    actions["start_dist_sideline"] = actions[["_start_dist_sideline", "start_y"]].min(axis=1)
    actions["start_dist_goalline"] = _spadl_cfg["length"] - actions["start_x"]
    return actions[["start_dist_sideline", "start_dist_goalline"]]

@required_fields(["end_x", "end_y"])
@fs.simple
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
    actions["_end_dist_sideline"] = _spadl_cfg["width"] - actions["end_y"]
    actions["end_dist_sideline"] = actions[["_end_dist_sideline", "end_y"]].min(axis=1)
    actions["end_dist_goalline"] = _spadl_cfg["length"] - actions["end_x"]
    return actions[["end_dist_sideline", "end_dist_goalline"]]

@required_fields(["start_x", "start_y", "end_x", "end_y"])
@fs.simple
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


@required_fields(["under_pressure"])
@fs.simple
def under_pressure(actions):
    """Get the value of StatsBomb's 'under_pressure' attribute.

    Every on-the-ball event that overlaps the duration of a pressure event is
    annotated as being 'under_pressure'. For example, if a pressure
    event appears before a pass, and the pressure’s timestamp plus its
    duration encompasses the pass’s timestamp, that pass is said to have been
    made under pressure. If a pressure event occurs after a pass, but before
    the end of the pass (as calculated by using its duration), that pass is
    said to have been received under pressure. Events which are naturally
    performed under pressure like duels, dribbles etc, all pick up the
    attribute, even in the absence of an actual pressure event. Carries can be
    pressured not just by pressure events, but other defensive events (defined
    in change 2.) that happen during or at the end of the carry

    Parameters
    ----------
    actions : SPADLActions
        The actions of a game.

    Returns
    -------
    Features
        The 'under_pressure' attribute of each action.
    """
    return actions[["under_pressure"]].fillna(False)


@required_fields(["period_id", "time_seconds", "player_id", "type_name"])
@fs.simple
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

@required_fields(["extra"])
@fs.simple
def ball_height_onehot(actions):
    """Get the one_hot_encoded height of a pass.

    The height is defined as:
        - "ground": ball doesn’t come off the ground.
        - "low": ball comes off the ground but is under shoulder level at peak height.
        - "high": ball goes above shoulder level at peak height.

    This feature is only defined for pass-like actions.

    Parameters
    ----------
    actions : SPADLActions
        The actions of a game.

    Returns
    -------
    Features
        The ball height during each pass-like action.
    """
    df = pd.DataFrame(index=actions.index)
    df["ball_height_ground"] = False
    df["ball_height_low"] = False
    df["ball_height_high"] = False
    passes = actions[actions.type_name.isin(_pass_like)]
    for idx, pas in passes.iterrows():
        pass_height = pas["extra"]["pass"].get("height", {"name": None}).get("name")
        if pass_height == "Ground Pass":
            df.at[idx, "ball_height_ground"] = True
        elif pass_height == "Low Pass":
            df.at[idx, "ball_height_low"] = True
        elif pass_height == "High Pass":
            df.at[idx, "ball_height_high"] = True
    return df

def _is_inside_triangle(pnt, triangle):
    """Compute whether the given point is in the given triangle.

    Parameters
    ----------
    pnt : tuple (x, y)
        The given point.

    triangle : list of tuples [(x0, y0), (x1, y1), (x2, y2)]
        The corners of the triangle, clockwise.

    Returns
    -------
        Boolean
    """

    def _is_right_of(line):
        return (
            (line[1][0] - line[0][0]) * (pnt[1] - line[0][1])
            - (pnt[0] - line[0][0]) * (line[1][1] - line[0][1])
        ) <= 0

    return (
        _is_right_of([triangle[0], triangle[1]])
        & _is_right_of([triangle[1], triangle[2]])
        & _is_right_of([triangle[2], triangle[0]])
    )

def _get_passing_cone(start, end, dist=1):
    """Compute the corners of the triangular passing cone between the given start and end location.

    The cone starts at the start location and has a width of 2*dist at the end location, with the end location
    indicating the middle of the line that connects the two adjacent corners.

    Parameters
    ----------
    start : tuple (x, y)
        The given start location.

    end : tuple (x, y)
        The given end location.

    dist : int
        The distance between the end location and its two adjacent corners of the triangle.

    Returns
    -------
    List of tuples [(x0, y0), (x1, y1), (x2, y2)] containing the corners of the triangle, clockwise.

    """
    if (start[0] == end[0]) | (start[1] == end[1]):
        slope = 0
    else:
        slope = (end[1] - start[1]) / (end[0] - start[0])

    dy = math.sqrt(dist**2 / (slope**2 + 1))
    dx = -slope * dy

    if start[0] == end[0]:  # have treated vertical line as horizontal one, rotate
        dx, dy = dy, dx

    pnt1 = (end[0] + dx, end[1] + dy)
    pnt2 = (end[0] - dx, end[1] - dy)
    return [start, pnt1, pnt2]


@required_fields(["freeze_frame_360", "start_x", "start_y", "end_x", "end_y"])
@fs.simple
def nb_opp_in_path(actions, path_width: int = 1):
    """Get the number of opponents in the path between the start and end location of a pass.

    The path is defined as a triangular corridor between the pass origin and
    the receiver's location with a base of `x` meters at the receiver's
    location.

    Parameters
    ----------
    actions : SPADLActions
        The actions of a game.
    path_width : float
        The width (in meters) of the triangular path at the receiver's location.

    Returns
    -------
    Features
        The number of opponents in the path of each pass.
    """
    df = pd.DataFrame(index=actions.index)
    df["nb_opp_in_path"] = 0

    passes = actions[actions.type_name.isin(_pass_like)]
    for idx, pas in passes.iterrows():
        if not pas["freeze_frame_360"]:
            continue

        start_coo = [pas["start_x"], pas["start_y"]]
        end_coo = [pas["end_x"], pas["end_y"]]

        if start_coo == end_coo:
            continue

        opponents_coo = [(o["x"], o["y"]) for o in pas["freeze_frame_360"] if not o["teammate"]]
        triangle = _get_passing_cone(start_coo, end_coo, path_width)
        df.at[idx, "nb_opp_in_path"] = sum(_is_inside_triangle(o, triangle) for o in opponents_coo)
    return df

@required_fields(["freeze_frame_360", "start_x", "start_y", "end_x", "end_y"])
@fs.simple
def packing_rate(actions):
    """Get the number of defenders that are outplayed by a pass.

    Parameters
    ----------
    actions : SPADLActions
        The actions of a game.

    Returns
    -------
    Features
        The number of defenders outplayed by a pass.
    """
    df = pd.DataFrame(index=actions.index)
    df["packing_rate"] = 0

    goal_coo = np.array([_goal_x, _goal_y])

    passes = actions[actions.type_name.isin(_pass_like)]
    for idx, pas in passes.iterrows():
        if not pas["freeze_frame_360"]:
            continue

        opponents_coo = np.array(
            [(o["x"], o["y"]) for o in pas["freeze_frame_360"] if not o["teammate"]]
        )
        if len(opponents_coo) == 0:
            continue

        ball_coo = np.array([pas["start_x"], pas["start_y"]])
        end_coo = np.array([pas["end_x"], pas["end_y"]])

        dist_ball_goal = np.sqrt(
            (goal_coo[0] - ball_coo[0]) ** 2 + (goal_coo[1] - ball_coo[1]) ** 2
        )
        dist_destination_goal = np.sqrt(
            (goal_coo[0] - end_coo[0]) ** 2 + (goal_coo[1] - end_coo[1]) ** 2
        )
        dist_def_goal = np.sqrt(
            (opponents_coo[:, 0] - goal_coo[0]) ** 2 + (opponents_coo[:, 1] - goal_coo[1]) ** 2
        )
        outplayed = (
            # The defender is between the ball and the goal before the pass
            (dist_def_goal <= dist_ball_goal)
            # The defender is further from the goal than the ball after the pass
            & (dist_def_goal > dist_destination_goal)
            # The ball moved closer to the goal
            & (dist_destination_goal <= dist_ball_goal)
        )
        df.at[idx, "packing_rate"] = np.sum(outplayed)
    return df

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
    defenders_in_radius = np.zeros((len(actions), 2), dtype=int)
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
@fs.simple
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

    dist = np.ones((len(actions), 3), dtype=float) * 20
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

@required_fields(["freeze_frame_360", "start_x", "start_y"])
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

    cloeset_teammates_xy = np.zeros((len(actions), num_players*3), dtype=int) # x, y, distance for each player
    cloeset_opponents_xy = np.zeros((len(actions), num_players*3), dtype=int) # x, y, distance for each player
    for i, (_, action) in enumerate(actions.iterrows()):
        if not action["freeze_frame_360"]:
            continue

        freeze_frame = pd.DataFrame.from_records(action["freeze_frame_360"])
        start_x, start_y = action.start_x, action.start_y
        teammate_locs = freeze_frame[freeze_frame.teammate & ~freeze_frame.actor].copy()
        opponent_locs = freeze_frame[~freeze_frame.teammate].copy()

        teammate_locs['distance'] = np.sqrt((teammate_locs['x'] - start_x) ** 2 + (teammate_locs['y'] - start_y) ** 2)
        opponent_locs['distance'] = np.sqrt((opponent_locs['x'] - start_x) ** 2 + (opponent_locs['y'] - start_y) ** 2)

        # Get the n closest teammates and opponents
        num_teammates = min(num_players, len(teammate_locs))
        num_opponents = min(num_players, len(opponent_locs))

        closest_teammates_df = teammate_locs.nsmallest(num_teammates, 'distance')[['x', 'y', 'distance']]
        closest_opponents_df = opponent_locs.nsmallest(num_opponents, 'distance')[['x', 'y', 'distance']]

        for j in range(num_teammates):
            cloeset_teammates_xy[i, j * 3] = closest_teammates_df.iloc[j]['x']
            cloeset_teammates_xy[i, j * 3 + 1] = closest_teammates_df.iloc[j]['y']
            cloeset_teammates_xy[i, j * 3 + 2] = closest_teammates_df.iloc[j]['distance']

        for j in range(num_opponents):
            cloeset_opponents_xy[i, j * 3] = closest_opponents_df.iloc[j]['x']
            cloeset_opponents_xy[i, j * 3 + 1] = closest_opponents_df.iloc[j]['y']
            cloeset_opponents_xy[i, j * 3 + 2] = closest_opponents_df.iloc[j]['distance']

    # Combine teammate and opponent data
    cloeset_xy = np.hstack((cloeset_teammates_xy, cloeset_opponents_xy))

    # Generate column names dynamically
    columns = []
    for k in range(1, num_players + 1):
        columns.extend([f"teammate_{k}_x", f"teammate_{k}_y", f"teammate_{k}_distance"])
    for k in range(1, num_players + 1):
        columns.extend([f"opponent_{k}_x", f"opponent_{k}_y", f"opponent_{k}_distance"])

    return pd.DataFrame(cloeset_xy, index=actions.index, columns=columns)

@required_fields(["type_id"])
@fs.simple
def prev_action(actions):
    """For each action, find previous events.

    Parameters
    ----------
    actions : SPADLActions
        The actions of a game.
    
    Returns:
    - DataFrame: event types of actions.
    """
    df = pd.DataFrame(actions['type_id'])
    
    # 첫 event 결측치 0(pass)으로 대체
    df['prev_type_id'] = df['type_id'].shift(1).fillna(0)

    return pd.DataFrame(df['prev_type_id'])

# parameter(radius) set
defenders_in_3m_radius = required_fields(
    ["start_x", "start_y", "end_x", "end_y", "freeze_frame_360"]
)(fs.simple(partial(_opponents_in_radius, radius=3)))
defenders_in_3m_radius.__name__ = "defenders_in_3m_radius"

closest_3_players = required_fields(["freeze_frame_360", "start_x", "start_y"])(
    fs.simple(partial(closest_players, num_players=3))
)
closest_3_players.__name__ = "closest_3_players"

all_features = [
    actiontype_onehot,
    result_onehot,
    bodypart_onehot,

    time,

    startlocation,
    relative_startlocation,
    endlocation,
    relative_endlocation,

    startpolar,
    endpolar,

    movement,
    team,
    time_delta,
    space_delta,
    goalscore,

    angle,
    under_pressure,
    packing_rate,
    ball_height_onehot,

    player_possession_time,
    speed,
    nb_opp_in_path,
    dist_opponent,
    defenders_in_3m_radius,
    closest_3_players,
]


def get_features(
    db: Database,
    game_id: int,
    xfns: List[Callable] = all_features,
    actionfilter: Optional[Callable] = None,
    nb_prev_actions: int = 3,
):
    """Apply a list of feature generators.

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
    # filter actions of interest
    if actionfilter is None:
        idx = pd.Series([True] * len(actions), index=actions.index)
    else:
        idx = actionfilter(actions)

    # check if we have to return an empty dataframe
    if idx.sum() < 1:
        column_names = feature_column_names(xfns, nb_prev_actions)
        return pd.DataFrame(columns=column_names)
    
    if len(xfns) < 1:
        return pd.DataFrame(index=actions.index.values[idx])
    

    # convert actions to gamestates
    home_team_id, _ = db.get_home_away_team_id(game_id)
    gamestates = play_left_to_right(to_gamestates(actions, nb_prev_actions), home_team_id)

    # compute features
    df_features = reduce(
        lambda left, right: pd.merge(left, right, how="outer", left_index=True, right_index=True),
        (fn(gamestates).loc[idx] for fn in xfns),
    )

    return df_features