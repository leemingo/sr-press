"""Implements the pass success probability component."""
from typing import Any, Dict, List

import mlflow
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier
from gplearn.genetic import SymbolicClassifier
from .base import (
    exPressComponent,
    expressXGBoostComponent,
    expressSymbolicComponent

)

class PressingComponent(exPressComponent):
    """The pass success probability component.

    From any given game situation where a player controls the ball, the model
    estimates the success probability of a pass attempted towards a potential
    destination location.
    """

    component_name = "pass_success"

    def _get_metrics(self, y, y_hat):
        y_pred = y_hat > 0.5
        return {
            "precision": precision_score(y, y_pred),
            "recall": recall_score(y, y_pred),
            "f1": f1_score(y, y_pred),
            "log_loss": log_loss(y, y_hat),
            "brier": brier_score_loss(y, y_hat),
            "roc_auc": roc_auc_score(y, y_hat),
        }


class XGBoostComponent(PressingComponent, expressXGBoostComponent):
    """A XGBoost model based on handcrafted features."""

    def __init__(
        self, model: XGBClassifier, features: Dict[str, List[str]], label: List[str] = ["counterpress"]
    ):
        super().__init__(
            model=model,
            features=features,
            label=label,
        )

class SymbolicComponent(PressingComponent, expressSymbolicComponent):
    """A XGBoost model based on handcrafted features."""

    def __init__(
        self, model: SymbolicClassifier, features: Dict[str, List[str]], label: List[str] = ["counterpress"]
    ):
        super().__init__(
            model=model,
            features=features,
            label=label,
        )