"""A dataset containing all pressing."""
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset

import express.config as config
from express.features import state, pressure
from express import labels

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
        state_xfns: Union[List, Dict[Union[str, Callable], Optional[List]]],
        pressure_xfns: Union[List, Dict[Union[str, Callable], Optional[List]]],
        yfns: List[Union[str, Callable]],
        transform: Optional[Callable] = None, 
        path: Optional[os.PathLike[str]] = None, 
        load_cached: bool = True,
        nb_prev_actions: int = 1,
    ):

        # Check requested features and labels
        self.nb_prev_actions = nb_prev_actions
        self.transform = transform

        self.state_xfns = self._parse_xfns(state_xfns, module=state, feature_fn=state.feature_column_names) # S_i = {a_i-2, a_i-1, a_i}
        self.pressure_state_xfns = self._parse_xfns(pressure_xfns["pressure_state_xfns"], module=pressure,  # p_i
                                                    feature_fn=pressure.feature_column_names_pressure_state, nb_prev_actions=1)
        self.pressure_with_context_xfns = self._parse_xfns(pressure_xfns["pressure_with_context_xfns"], module=pressure,  # {p_i, a_i}
                                                           feature_fn=pressure.feature_column_names_pressure_context, nb_prev_actions=1)
        
        self.yfns = self._parse_yfns(yfns)
        # Try to load the dataset
        self.store = Path(path) if path is not None else None

        self._features = None
        self._state_features = pd.DataFrame() 
        self._pressure_state_features = pd.DataFrame()
        self._pressure_with_context_features = pd.DataFrame()
        self._labels = None

        if load_cached:
            if self.store is None:
                raise ValueError("No path to cached dataset provided.")
            try:
                if len(self.state_xfns):
                    self._state_features = pd.concat(
                        [
                            pd.read_parquet(self.store / f"s_{xfn.__name__}.parquet")[cols]
                            for xfn, cols in self.state_xfns.items()
                        ],
                        axis=1,
                    )
                if len(self.pressure_state_xfns):
                    self._pressure_state_features = pd.concat(
                        [
                            pd.read_parquet(self.store / f"p_{xfn.__name__}.parquet")[cols]
                            for xfn, cols in self.pressure_state_xfns.items()
                        ],
                        axis=1,
                    )
                if len(self.pressure_with_context_xfns):
                    self._pressure_with_context_features = pd.concat(
                        [
                            pd.read_parquet(self.store / f"p_{xfn.__name__}.parquet")[cols]
                            for xfn, cols in self.pressure_with_context_xfns.items()
                        ],
                        axis=1,
                    )
                self._features = pd.concat([self._state_features, self._pressure_state_features, self._pressure_with_context_features], axis=1)
                
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
        xfns: Union[List, Dict[Union[str, Callable], Optional[List]]],
        module: object,
        feature_fn: Callable,
        nb_prev_actions: Optional[int] = None
    ) -> Dict[Callable, Optional[List]]:
        parsed_xfns = {}

        if isinstance(xfns, list):
            xfns = {xfn: None for xfn in xfns}
        for xfn, cols in xfns.items():
            parsed_xfn = xfn
            parsed_cols = cols

            if isinstance(parsed_xfn, str):
                try:
                    parsed_xfn = getattr(module, parsed_xfn)
                except AttributeError:
                    raise ValueError(f"No feature function found that matches '{parsed_xfn}'.")
            if parsed_cols is None:
                parsed_cols = feature_fn([parsed_xfn], nb_prev_actions=nb_prev_actions or self.nb_prev_actions)
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
        is_pressing = actions.type_id == config.actiontypes.index("pressing")  # pressing
        is_visible_area_360 = actions["visible_area_360"].notna()  # visible_area_360

        return is_pressing & is_visible_area_360 #& has_required_players & in_attacking_third

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

        df_features = []
        df_labels = []
        for game_id in tqdm(self.games_idx):     
            # pressure_features.index: 압박 당하는 액션(a_i)를 기준으로 병합 -> a_i동안 발생하는 압박이벤트를 조사하기 위함(pressure.get_features 참고)
            # we extend SPADL with a tenth attribute containing a set pi that describes all pressure events of the opposing team during action ai     
            pressure_features = pressure.get_features(
                    db,
                    game_id=game_id,
                    xfns={"pressure_state_xfns": self.pressure_state_xfns,
                            "pressure_with_context_xfns": self.pressure_with_context_xfns},
                    nb_prev_actions=1,
                    actionfilter=self.actionfilter,
                )

            state_features = state.get_features(
                    db,
                    game_id=game_id,
                    xfns=self.state_xfns.keys(),
                    nb_prev_actions=self.nb_prev_actions,
                    actionfilter=lambda actions: pd.Series(actions.index.isin(pressure_features.index), index=actions.index), # 압박을 당한 액션(a_i)에 한정하여 feature engineering수행: faster!
                )
            
            df_features.append(
                pd.merge(state_features, pressure_features, 
                         left_index=True, right_index=True, how="left")
                )
            
            # Compute labels for each pressing
            if len(self.yfns):
                df_labels.append(
                    labels.get_labels(
                        db,
                        game_id=game_id,
                        yfns=self.yfns,
                        actionfilter= None,
                        pressured_idx = pressure_features.index
                    )
                )

        self._features = pd.concat(df_features, axis=0, ignore_index=False)
        self._labels = pd.concat(df_labels)

        if self.store is not None:
            assert self.store is not None

            # save features
            for xfn, col in self.state_xfns.items():           
                self._features[col].to_parquet(self.store / f"s_{xfn.__name__}.parquet")
            for xfn, col in self.pressure_state_xfns.items():
                self._features[col].to_parquet(self.store / f"p_{xfn.__name__}.parquet")
            for xfn, col in self.pressure_with_context_xfns.items():
                self._features[col].to_parquet(self.store / f"p_{xfn.__name__}.parquet")

            # save labels
            for yfn in self.yfns:
                col = [yfn.__name__]
                self._labels[col].to_parquet(self.store / f"y_{yfn.__name__}.parquet")
         
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