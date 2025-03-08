"""Model architectures."""

import pickle
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import torch
from xgboost import XGBClassifier, XGBRegressor
from gplearn.genetic import SymbolicClassifier
from rich.progress import track
from tqdm import tqdm
from sklearn.model_selection import cross_val_score, train_test_split
from torch.utils.data import DataLoader, Subset, random_split

from express.datasets import PressingDataset


class exPressComponent(ABC):
    """Base class for all components."""

    component_name = "default"

    def __init__(self, features: Union[List, Dict], label: List, transform: Optional[Callable] = None):
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
        # Load data
        data = self.initialize_dataset(dataset)
        X_train, X_val, y_train, y_val = train_test_split(data.features, data.labels, test_size=0.2)

        self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)], **train_cfg)

        # Return metric score for hyperparameter optimization
        if optimized_metric is not None:
            idx = self.model.best_iteration
            return self.model.evals_result()["validation_0"][optimized_metric][idx]

        return None

    def test(self, dataset) -> Dict[str, float]:
        data = self.initialize_dataset(dataset)
        X_test, y_test = data.features, data.labels
        if isinstance(self.model, XGBClassifier):
            y_hat = self.model.predict_proba(X_test)[:, 1]
        else:
            raise AttributeError(f"Unsupported xgboost model: {type(self.model)}")

        return self._get_metrics(y_test, y_hat)

    def predict(self, dataset) -> pd.Series:
        data = self.initialize_dataset(dataset)
        if isinstance(self.model, XGBClassifier):
            y_hat = self.model.predict_proba(data.features)[:, 1]
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
        else:
            raise AttributeError(f"Unsupported Symbolic model: {type(self.model)}")
        return self._get_metrics(y_test, y_hat)

    def predict(self, dataset) -> pd.Series:
        data = self.initialize_dataset(dataset)
        if isinstance(self.model, SymbolicClassifier):
            y_hat = self.model.predict_proba(data.features)[:, 1]
        else:
            raise AttributeError(f"Unsupported Symbolic model: {type(self.model)}")
        return pd.Series(y_hat, index=data.features.index)


class exPressPytorchComponent(exPressComponent):
    """Base class for a PyTorch-based component."""

    def __init__(self, model, features, label, transform, params):

        super().__init__(features, label, transform)
        self.model = model
        self.params = params
        self.save_path = params["save_path"]

        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss", dirpath=self.save_path, filename="{val_loss: .2f}", **self.params["ModelCheckpoint"]
        )

        early_stop_callback = EarlyStopping(monitor="val_loss", **self.params["EarlyStopConfig"])

        # Init lightning trainer
        self.trainer = pl.Trainer(callbacks=[checkpoint_callback, early_stop_callback], **self.params["TrainerConfig"])

    def train(
        self,
        dataset,
        param_grid=None,
        optimized_metric=None,
    ) -> Optional[float]:

        # Load data
        print()
        print("Generating datasets...")
        data = self.initialize_dataset(dataset)

        nb_train = int(len(data) * 0.8)
        lengths = [nb_train, len(data) - nb_train]
        _data_train, _data_val = random_split(data, lengths)

        train_dataloader = DataLoader(_data_train, shuffle=True, **self.params["DataConfig"])
        val_dataloader = DataLoader(_data_val, shuffle=False, **self.params["DataConfig"])

        self.trainer.fit(
            model=self.model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
        )

        return None

    def test(self, dataset) -> Dict[str, float]:
        # Load dataset
        data = self.initialize_dataset(dataset)
        dataloader = DataLoader(data, shuffle=False, **self.params["DataConfig"])

        outputs = self.trainer.predict(dataloaders=dataloader, ckpt_path="best")
        preds, targets = zip(*outputs)

        all_preds = np.concatenate(preds, axis=0)
        all_targets = np.concatenate(targets, axis=0)

        # Compute metricsreturn
        return self._get_metrics(all_targets, all_preds)

    def predict(self, dataset) -> pd.Series:
        # Load dataset
        data = self.initialize_dataset(dataset)
        dataloader = DataLoader(data, shuffle=False, **self.params["DataConfig"])

        outputs = self.trainer.predict(dataloaders=dataloader)
        preds, _ = zip(*outputs)

        all_preds = np.concatenate(preds, axis=0)

        # Ensure all_preds is 1-dimensional
        all_preds = all_preds.flatten()

        return pd.Series(all_preds, index=data.features.index)
