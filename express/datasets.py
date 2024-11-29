"""A dataset containing all pressing."""
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import random
import pandas as pd
from rich.progress import track
from tqdm import tqdm
from torch.utils.data import Dataset

import express.config as config
from express import features, labels
from express.utils import pressed_prev_actions, load_actions

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
        pressing_filter: bool = False
    ):

        # Check requested features and labels
        self.nb_prev_actions = nb_prev_actions
        self.transform = transform
        self.augment = False

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

        # nb_prev_action내 압박을 당한 액션이 존재하는 데이터셋 한정한 분석
        # 압박 당한 액션: 압박액션시작시점
        if pressing_filter:
            print("pressing_filtering.....")
            db_actions = {game_id: load_actions(game_id) for game_id in self._features.index.get_level_values("game_id").unique()}
            pressing_mask = self._features.reset_index().apply(
                lambda row: pressed_prev_actions(
                    row, 
                    db_actions[row.game_id], 
                    self.nb_prev_actions
                ), 
                axis=1
            )
            pressing_mask.index = self._features.index
            self._features = self._features.loc[pressing_mask]
            self._labels = self._labels.loc[pressing_mask]

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

        return is_pressing & is_visible_area_360

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
            for game_id in tqdm(self.games_idx):
                pressure_features = features.get_features(
                        db,
                        game_id=game_id,
                        xfns=self.xfns.keys(),
                        nb_prev_actions=self.nb_prev_actions,
                        actionfilter=self.actionfilter,
                    )

                if not pressure_features.empty:
                    df_features.append(pressure_features)
                
            self._features = pd.concat(df_features, axis=0, ignore_index=False)

        # Compute labels for each pass
        if len(self.yfns):
            df_labels = []
            for game_id in tqdm(self.games_idx):
                df_labels.append(
                    labels.get_labels(
                        db,
                        game_id=game_id,
                        yfns=self.yfns,
                        actionfilter=self.actionfilter,
                    )
                )

            self._labels = pd.concat(df_labels, axis=0, ignore_index=False)

        if self.store is not None:
            assert self.store is not None
            
            for xfn, col in self.xfns.items():           
                self._features[col].to_parquet(self.store / f"x_{xfn.__name__}.parquet")

            for yfn in self.yfns:
                self._labels[[yfn.__name__]].to_parquet(self.store / f"y_{yfn.__name__}.parquet")
         
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

        if self.augment:
            freeze_frame = pd.DataFrame.from_records(sample_features["freeze_frame_360_a0"]).copy()

            augment_num = 1#random.randint(1, len(freeze_frame)) # 1<= num <= total_player
            augment_idxs = random.sample(list(freeze_frame.index), augment_num) 

            random_shift_x = (random.random() - 0.5) * config.field_length / 5  # from -0.1 to 0.1 * pitch_size
            random_shift_y = (random.random() - 0.5) * config.field_width / 5  # from -0.1 to 0.1 * pitch_size
            freeze_frame.loc[augment_idxs, ["x"]] += random_shift_x
            freeze_frame.loc[augment_idxs, ["y"]] += random_shift_y
            
            sample_features["freeze_frame_360_a0"] = freeze_frame.to_dict(orient='records')

        sample = {
            "game_id": game_id,
            "action_id": action_id,
            **sample_features,
            **sample_target,
        }

        if self.transform:
            sample = self.transform(sample)

        return sample
