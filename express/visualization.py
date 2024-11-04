"""Data visualisation."""
import copy
import numpy as np
import pandas as pd
from mplsoccer import Pitch
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import express.config as config

def plot_action(
    action: pd.Series,
    show_action=True,
    show_visible_area=True,
    home_team_id=None,
    ax=None,
) -> None:
    """Plot a SPADL(include pressing) action with 360 freeze frame.

    Parameters
    ----------
    action : pandas.Series
        A row from the actions DataFrame.
    surface : np.arry, optional
        A surface to visualize on top of the pitch.
    ax : matplotlib.axes.Axes, optional
        The axes to plot on.
    surface_kwargs : dict, optional
        Keyword arguments to pass to the surface plotting function.
    """
    color_list = ["b", "r", "g"]

    if action["team_id"] == home_team_id:
        color_list = ["b", "r", "g"] # home-team, away-team, event_player
        label_list = ["Home", "Away"]
    else:
        color_list = ["r", "b", "g"] # away-team, home-team, event_player
        label_list = ["Away", "Home"]

    # parse freeze frame
    freeze_frame = pd.DataFrame.from_records(action["freeze_frame_360"])
    visible_area = action["visible_area_360"]

    teammate_locs = freeze_frame[freeze_frame.teammate].copy()
    opponent_locs = freeze_frame[~freeze_frame.teammate].copy()
    event_player_loc = freeze_frame[freeze_frame.actor].copy()

    # set up pitch
    # p = Pitch(pitch_type="custom", pitch_length=105, pitch_width=68, color="green")
    p = Pitch(pitch_type="custom", linewidth=1, pitch_length=config.field_length,
                      pitch_width=config.field_width, half=False, corner_arcs=True,
                      pad_left=1, pad_right=1, pad_bottom=1, pad_top=1,
                      pitch_color= (0/255, 128/255, 0/255, 0.5), line_color='white')
    if ax is None:
        _, ax = p.draw(figsize=(8, 12))
    else:
        p.draw(ax=ax)

    # plot action
    if show_action:
        p.arrows(
            action["start_x"],
            action["start_y"],
            action["end_x"],
            action["end_y"],
            color="black",
            headwidth=5,
            headlength=5,
            width=3,
            ax=ax,
        )
        
    # plot visible area
    if show_visible_area:
        p.polygon([visible_area], color=(236 / 256, 236 / 256, 236 / 256, 0.5), ax=ax)

    # Calculate distances to teammates
    teammate_locs['distance'] = teammate_locs.apply(lambda row: calculate_distance(action.start_x, action.start_y, row.x, row.y), axis=1)
    closest_teammates = teammate_locs.nsmallest(3, 'distance')

    # Calculate distances to opponents
    opponent_locs['distance'] = opponent_locs.apply(lambda row: calculate_distance(action.start_x, action.start_y, row.x, row.y), axis=1)
    closest_opponents = opponent_locs.nsmallest(3, 'distance')

    p.scatter(closest_teammates.x, closest_teammates.y, color=color_list[0], s=200, ec="k", edgecolor=color_list[0], ax=ax)
    p.scatter(closest_opponents.x, closest_opponents.y, color=color_list[1], s=200, ec="k", edgecolor=color_list[1], ax=ax)

    p.scatter(teammate_locs.x, teammate_locs.y, c=color_list[0], s=200, ec="k", alpha=0.5, ax=ax)
    p.scatter(opponent_locs.x, opponent_locs.y, c=color_list[1], s=200, ec="k", alpha=0.5, ax=ax)
    p.scatter(event_player_loc.x, event_player_loc.y, c=color_list[2], s=400, ec="k", marker="*", ax=ax)

    ax.set_title(f'{action["type_name"]}')
    
    hometeam_dot = mlines.Line2D([], [], color="b" , marker='o', linestyle='None', markersize=10, label='Home')
    awayteam_dot = mlines.Line2D([], [], color='r', marker='o', linestyle='None', markersize=10, label='Away')
    event_player_dot = mlines.Line2D([], [], color='g', marker='o', linestyle='None', markersize=10, label=f'{action["type_name"]} player')

    ax.legend(handles=[hometeam_dot, awayteam_dot, event_player_dot], loc='upper left')

    return ax

def calculate_distance(x1, y1, x2, y2):
    """Calculate Euclidean distance between two points."""
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

