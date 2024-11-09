"""A dataset containing all pressing."""
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import pandas as pd
from rich.progress import track
from torch.utils.data import Dataset

import express.config as config
from express import features, labels

class PressingDataset(Dataset):
    """A dataset containing Pressing.

    Parameters
    ----------
    xfns : dict(str or callable -> list(str))
        The feature generators and columns to use.
    yfns : list(str or callable)
        The label generators.
    transform : Callable, optional
        A function/transform that takes a sample and returns a transformed
        version of it.
    path : Path
        The path to the directory where pre-computed features are stored. By
        default all features and labels are computed on the fly, but
        pre-computing them will speed up training significantly.
    load_cached : bool, default: True
        Whether to attempt to load the dataset from disk.
    """

    def __init__(
        self,
        xfns: Union[List, Dict[Union[str, Callable], Optional[List]]],
        yfns: List[Union[str, Callable]],
        transform: Optional[Callable] = None, 
        path: Optional[os.PathLike[str]] = None, 
        load_cached: bool = True,
        nb_prev_actions: int = 1,
        min_players: int = None,
    ):

        # Check requested features and labels
        self.nb_prev_actions = nb_prev_actions
        self.min_players = min_players
        self.transform = transform

        self.xfns = self._parse_xfns(xfns)
        self.yfns = self._parse_yfns(yfns)

        # Try to load the dataset
        self.store = Path(path) if path is not None else None

        self._features = None   
        self._labels = None

        if load_cached:
            if self.store is None:
                raise ValueError("No path to cached dataset provided.")
            try:
                if len(self.xfns):
                    self._features = pd.concat(
                        [
                            pd.read_parquet(self.store / f"x_{xfn.__name__}.parquet")[cols]
                            for xfn, cols in self.xfns.items()
                        ],
                        axis=1,
                    )

                if len(self.yfns):
                    self._labels = pd.concat(
                        [
                            pd.read_parquet(self.store / f"y_{yfn.__name__}.parquet")
                            for yfn in self.yfns
                        ],
                        axis=1,
                    )
            except FileNotFoundError:
                raise FileNotFoundError("No complete dataset found at %s. Run 'create' to create it.", self.store)

    # The @staticmethod decorator is removed to access `nb_prev_actions` from the PressingDataset instance.
    # By removing it, we can utilize `self.nb_prev_actions` directly within this method.
    # @staticmethod
    def _parse_xfns(
        self,
        xfns: Union[List, Dict[Union[str, Callable], Optional[List]]]
    ) -> Dict[Callable, Optional[List]]:
        parsed_xfns = {}

        if isinstance(xfns, list):
            xfns = {xfn: None for xfn in xfns}
        for xfn, cols in xfns.items():
            parsed_xfn = xfn
            parsed_cols = cols

            if isinstance(parsed_xfn, str):
                try:
                    parsed_xfn = getattr(features, parsed_xfn)
                except AttributeError:
                    raise ValueError(f"No feature function found that matches '{parsed_xfn}'.")
            if parsed_cols is None:
                parsed_cols = features.feature_column_names([parsed_xfn], nb_prev_actions=self.nb_prev_actions)
            parsed_xfns[parsed_xfn] = parsed_cols
        
        return parsed_xfns
    
    # The @staticmethod decorator is removed to access `nb_prev_actions` from the PressingDataset instance.
    # By removing it, we can utilize `self.nb_prev_actions` directly within this method.
    # @staticmethod
    def _parse_yfns(
        self,
        yfns: List[Union[str, Callable]]) -> List[Callable]:
        parsed_yfns = []
        for yfn in yfns:
            if isinstance(yfn, str):
                try:
                    parsed_yfns.append(getattr(labels, yfn))
                except AttributeError:
                    raise ValueError(f"No labeling function found that matches '{yfn}'.")
            else:
                parsed_yfns.append(yfn)
        return parsed_yfns

    # The @staticmethod decorator is removed to access `min_players` from the PressingDataset instance.
    # By removing it, we can utilize `self.min_players` directly within this method.
    # @staticmethod
    def actionfilter(self, actions: pd.DataFrame) -> pd.Series:
        is_pressing = (actions.type_id == config.actiontypes.index("pressing"))  # pressing
        is_visible_area_360 = actions["visible_area_360"].notna()  # visible_area_360

        # Check if freeze_frame_360 has at least 3 teammates and 3 opponents
        has_required_players = actions["freeze_frame_360"].apply(lambda frame: (
            frame is not None and
            sum(player["teammate"] for player in frame) >= self.min_players and
            sum(not player["teammate"] for player in frame) >= self.min_players
        ))

        # Ensure the action starts in the attacking third
        in_attacking_third = actions["start_x"] > config.field_length * (2 / 3)
        return is_pressing & is_visible_area_360 & has_required_players & in_attacking_third

    def create(self, db) -> None:
        """Create the dataset.

        Parameters
        ----------
        db : Database
            The database with raw data.
        """
        # Create directory 
        if self.store is not None:
            self.store.mkdir(parents=True, exist_ok=True)

        # Select games to include
        self.games_idx = list(db.games().index)

        # Compute features for each pressing
        if len(self.xfns):
            df_features = []
            for xfn, _ in self.xfns.items():
                df_features_xfn = []
                for game_id in track(self.games_idx, description=f"Computing {xfn.__name__} feature"):
                    df_features_xfn.append(
                        features.get_features(
                            db,
                            game_id=game_id,
                            xfns=[xfn],
                            nb_prev_actions=self.nb_prev_actions,
                            actionfilter=self.actionfilter,
                        )
                    )
                
                df_features_xfn = pd.concat(df_features_xfn)
                if self.store is not None:
                    assert self.store is not None
                    df_features_xfn.to_parquet(self.store / f"x_{xfn.__name__}.parquet")
                df_features.append(df_features_xfn)
            self._features = pd.concat(df_features, axis=1)

        # Compute labels for each pass
        if len(self.yfns):
            df_labels = []
            for yfn in self.yfns:
                df_labels_yfn = []
                for game_id in track(self.games_idx, description=f"Computing {yfn.__name__} label"):
                    df_labels_yfn.append(
                        labels.get_labels(
                            db,
                            game_id=game_id,
                            yfns=[yfn],
                            actionfilter=self.actionfilter,
                        )
                    )
                df_labels_yfn = pd.concat(df_labels_yfn)
                if self.store is not None:
                    df_labels_yfn.to_parquet(self.store / f"y_{yfn.__name__}.parquet")
                df_labels.append(df_labels_yfn)
            self._labels = pd.concat(df_labels, axis=1)

    @property
    def features(self):
        if self._features is None:
            assert self._labels is not None, "First, create the dataset."
            return pd.DataFrame(index=self._labels.index)
        return self._features

    @property
    def labels(self):
        if self._labels is None:
            assert self._features is not None, "First, create the dataset."
            return pd.DataFrame(index=self._features.index)
        return self._labels

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        if self.features is not None:
            return len(self.features)
        if self.labels is not None:
            return len(self.labels)
        return 0

    def __getitem__(self, idx: int) -> Dict:
        """Return a sample from the dataset at the given index.

        Parameters
        ----------
        idx : int
            The index of the sample to return.

        Returns
        -------
        sample : (dict, dict)
            A dictionary containing the sample and target.
        """
        game_id, action_id = None, None

        sample_features = {}
        if self.features is not None:
            sample_features = self.features.iloc[idx].to_dict()
            game_id, action_id = self.features.iloc[idx].name

        sample_target = {}
        if self.labels is not None:
            sample_target = self.labels.iloc[idx].to_dict()
            game_id, action_id = self.labels.iloc[idx].name

        sample = {
            "game_id": game_id,
            "action_id": action_id,
            **sample_features,
            **sample_target,
        }

        if self.transform:
            sample = self.transform(sample)

        return sample
