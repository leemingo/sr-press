import os
import sys
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(base_path)

import pandas as pd
import socceraction.spadl.config as _spadl

field_length = _spadl.field_length
field_width = _spadl.field_width

results = _spadl.results
bodyparts = _spadl.bodyparts

actiontypes = _spadl.actiontypes + [
    "pressing"
]

_spadl_cfg = {
    "length": field_length,
    "width": field_width,
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

def actiontypes_df() -> pd.DataFrame:
    """Return a dataframe with the type id and type name of each Atomic-SPADL action type.

    Returns
    -------
    pd.DataFrame
        The 'type_id' and 'type_name' of each Atomic-SPADL action type.
    """
    return pd.DataFrame(list(enumerate(actiontypes)), columns=["type_id", "type_name"])

def results_df() -> pd.DataFrame:
    """Return a dataframe with the result id and result name of each SPADL action type.

    Returns
    -------
    pd.DataFrame
        The 'result_id' and 'result_name' of each SPADL action type.
    """
    return pd.DataFrame(list(enumerate(results)), columns=["result_id", "result_name"])


def bodyparts_df() -> pd.DataFrame:
    """Return a dataframe with the bodypart id and bodypart name of each SPADL action type.

    Returns
    -------
    pd.DataFrame
        The 'bodypart_id' and 'bodypart_name' of each SPADL action type.
    """
    return pd.DataFrame(list(enumerate(bodyparts)), columns=["bodypart_id", "bodypart_name"])
