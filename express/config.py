import os
import sys
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(base_path)

import pandas as pd
import socceraction.spadl.config as _spadl

field_length = _spadl.field_length
field_width = _spadl.field_width

results = _spadl.results
results_df = _spadl.results_df()

bodyparts = _spadl.bodyparts
bodyparts_df = _spadl.bodyparts_df()

actiontypes = _spadl.actiontypes + [
    "pressing"
]


def actiontypes_df() -> pd.DataFrame:
    """Return a dataframe with the type id and type name of each Atomic-SPADL action type.

    Returns
    -------
    pd.DataFrame
        The 'type_id' and 'type_name' of each Atomic-SPADL action type.
    """
    return pd.DataFrame(list(enumerate(actiontypes)), columns=["type_id", "type_name"])
