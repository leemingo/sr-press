"""Model architectures."""
import pickle
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import mlflow
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import xgboost as xgb
from gplearn.genetic import SymbolicClassifier
from rich.progress import track
from sklearn.model_selection import cross_val_score, train_test_split
from torch.utils.data import DataLoader, Subset, random_split

from express.datasets import PressingDataset


class exPressComponent(ABC):
    """Base class for all components."""

    component_name = "default"

    def __init__(
        self, features: Union[List, Dict], label: List, transform: Optional[Callable] = None
    ):
        self.features = features
        self.label = label
        self.transform = transform

    def initialize_dataset(self, dataset: Union[PressingDataset, Callable]) -> PressingDataset:
        if callable(dataset):
            return dataset(xfns=self.features, yfns=self.label, transform=self.transform)
        return dataset

    @abstractmethod
    def train(self, dataset: Callable, optimized_metric=None) -> Optional[float]:
        pass

    @abstractmethod
    def test(self, dataset: Callable) -> Dict[str, float]:
        pass

    def _get_metrics(self, y_true, y_hat):
        return {}

    @abstractmethod
    def predict(self, dataset: Callable) -> pd.Series:
        pass

    def save(self, path: Path):
        pickle.dump(self, path.open(mode="wb"))

    @classmethod
    def load(cls, path: Path):
        return pickle.load(path.open(mode="rb"))


class expressXGBoostComponent(exPressComponent):
    """Base class for an XGBoost-based component."""

    def __init__(self, model, features, label):
        super().__init__(features, label)
        self.model = model

    def train(self, dataset, optimized_metric=None, **train_cfg) -> Optional[float]:
        mlflow.xgboost.autolog()

        # Load data
        data = self.initialize_dataset(dataset)
        X_train, X_val, y_train, y_val = train_test_split(
            data.features, data.labels, test_size=0.2
        )

        self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)], **train_cfg)

        # Return metric score for hyperparameter optimization
        if optimized_metric is not None:
            idx = self.model.best_iteration
            return self.model.evals_result()["validation_0"][optimized_metric][idx]

        return None

    def test(self, dataset) -> Dict[str, float]:
        data = self.initialize_dataset(dataset)
        X_test, y_test = data.features, data.labels
        if isinstance(self.model, xgb.XGBClassifier):
            y_hat = self.model.predict_proba(X_test)[:, 1]
        elif isinstance(self.model, xgb.XGBRegressor):
            y_hat = self.model.predict(X_test)
        else:
            raise AttributeError(f"Unsupported xgboost model: {type(self.model)}")
        return self._get_metrics(y_test, y_hat)

    def predict(self, dataset) -> pd.Series:
        data = self.initialize_dataset(dataset)
        if isinstance(self.model, xgb.XGBClassifier):
            y_hat = self.model.predict_proba(data.features)[:, 1]
        elif isinstance(self.model, xgb.XGBRegressor):
            y_hat = self.model.predict(data.features)
        else:
            raise AttributeError(f"Unsupported xgboost model: {type(self.model)}")
        return pd.Series(y_hat, index=data.features.index)

class expressSymbolicComponent(exPressComponent):
    """Base class for an Symbolic-based component."""

    def __init__(self, model, features, label):
        super().__init__(features, label)
        self.model = model

    def train(self, dataset, optimized_metric=None, **train_cfg) -> Optional[float]:

        # Load data
        data = self.initialize_dataset(dataset)
        X_train, y_train = data.features, data.labels


        self.model.fit(X_train, y_train, **train_cfg)

        # Return metric score for hyperparameter optimization
        if optimized_metric is not None:
            idx = self.model.best_iteration
            return self.model.evals_result()["validation_0"][optimized_metric][idx]

        return None

    def test(self, dataset) -> Dict[str, float]:
        data = self.initialize_dataset(dataset)
        X_test, y_test = data.features, data.labels
        if isinstance(self.model, SymbolicClassifier):
            y_hat = self.model.predict_proba(X_test)[:, 1]
        elif isinstance(self.model, SymbolicClassifier):
            y_hat = self.model.predict(X_test)
        else:
            raise AttributeError(f"Unsupported Symbolic model: {type(self.model)}")
        return self._get_metrics(y_test, y_hat)

    def predict(self, dataset) -> pd.Series:
        data = self.initialize_dataset(dataset)
        if isinstance(self.model, SymbolicClassifier):
            y_hat = self.model.predict_proba(data.features)[:, 1]
        elif isinstance(self.model, SymbolicClassifier):
            y_hat = self.model.predict(data.features)
        else:
            raise AttributeError(f"Unsupported Symbolic model: {type(self.model)}")
        return pd.Series(y_hat, index=data.features.index)