"""Implements the pass success probability component."""
import os
from typing import Any, Dict, List

import pandas as pd
import pytorch_lightning as pl
from typing import Callable, Dict, List, Optional, Union
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from sklearn.metrics import (
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier, XGBRegressor
from gplearn.genetic import SymbolicClassifier
from .base import (
    exPressComponent,
    expressXGBoostComponent,
    expressSymbolicComponent,
    exPressPytorchComponent,
    expressScikitComponent
)
from .soccermap import PytorchSoccerMapModel, ToSoccerMapTensor

class PressingComponent(exPressComponent):
    """The pressing success probability component."""

    component_name = "pressing"

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
        self, model: XGBClassifier, 
        features: Dict[str, List[str]], 
        label: List[str],
        params: Dict[str, Dict[str, Union[int, str, bool]]] 
    ):
        super().__init__(
            model=model,
            features=features,
            label=label,
        )
        self.save_path = params["save_path"]

    def train(self, dataset, param_grid=None, optimized_metric=None, **train_cfg) -> Optional[float]:
        # Load data
        data = self.initialize_dataset(dataset)
        X_train, y_train = data.features, data.labels

        if param_grid:
            # GridSearchCV to find the best hyperparameters
            grid_search = GridSearchCV(
                estimator=self.model,
                param_grid=param_grid,
                scoring=optimized_metric if optimized_metric else 'accuracy',
                cv=5,
                verbose=1,
                n_jobs=-1,
                refit=True     
            )
            grid_search.fit(X_train, y_train)
            self.model = grid_search.best_estimator_
            print("Best Parameters from GridSearchCV:", grid_search.best_params_)
            print("Best Score from GridSearchCV:", grid_search.best_score_)
        else:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=0.2, stratify=y_train
            )
            self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)], **train_cfg)

        with open(os.path.join(self.save_path, "log.txt"), "w") as f:
            # save xgboost's parameter
            f.write("##### Model Configuration #####\n")
            for key, value in self.model.get_params().items():
                f.write(f"{key}: {value}\n")

            # save xgboost's metrics
            eval_log = self.model.evals_result()["validation_0"]
            for metric_name, metric_values in eval_log.items():
                f.write(f"\n##### {metric_name} #####\n")
                for epoch, value in enumerate(metric_values):
                    f.write(f"Epoch {epoch + 1}: {value}\n")

class ScikitComponent(PressingComponent, expressScikitComponent):
    """A XGBoost model based on handcrafted features."""

    def __init__(
        self, model, 
        features: Dict[str, List[str]], 
        label: List[str],
        params: Dict[str, Dict[str, Union[int, str, bool]]] 
    ):
        super().__init__(
            model=model,
            features=features,
            label=label,
        )
        self.save_path = params["save_path"]

    def train(self, dataset, param_grid=None, optimized_metric=None, **train_cfg) -> Optional[float]:
        # Load data
        data = self.initialize_dataset(dataset)

        
        X_train, y_train = data.features, data.labels

        X_train = X_train.fillna(0)

        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, stratify=y_train
        )
        self.model.fit(X_train, y_train)

        with open(os.path.join(self.save_path, "log.txt"), "w") as f:
            # save xgboost's parameter
            f.write("##### Model Configuration #####\n")
            for key, value in self.model.get_params().items():
                f.write(f"{key}: {value}\n")

class SymbolicComponent(PressingComponent, expressSymbolicComponent):
    """A XGBoost model based on handcrafted features."""

    def __init__(
        self, 
        model: SymbolicClassifier, 
        features: Dict[str, List[str]], 
        label: List[str],
        params: Dict[str, Dict[str, Union[int, str, bool]]],
    ):
        super().__init__(
            model=model,
            features=features,
            label=label,
        )
        self.save_path = params["save_path"]

    def train(self, dataset, param_grid=None, optimized_metric=None, **train_cfg) -> Optional[float]:
        # Load data
        data = self.initialize_dataset(dataset)
        X_train, y_train = data.features, data.labels
        X_train = X_train.fillna(0)

        if param_grid:
            # GridSearchCV to find the best hyperparameters
            grid_search = GridSearchCV(
                estimator=self.model,
                param_grid=param_grid,
                scoring=optimized_metric if optimized_metric else 'accuracy',
                cv=5,
                verbose=1,
                n_jobs=-1,
                refit=True
            )
            grid_search.fit(X_train, y_train)
            self.model = grid_search.best_estimator_
            print("Best Parameters from GridSearchCV:", grid_search.best_params_)
            print("Best Score from GridSearchCV:", grid_search.best_score_)
        else:
            # Train the model without GridSearch
            self.model.fit(X_train, y_train, **train_cfg)

        with open(os.path.join(self.save_path, "log.txt"), "w") as f:
            f.write("##### Best Symbolic Regression Expression #####\n")
            f.write(f"{self.model._program}\n")

            # save xgboost's parameter
            f.write("\n##### Model Configuration #####\n")
            for key, value in self.model.get_params().items():
                f.write(f"{key}: {value}\n")

            run_details = self.model.run_details_
            f.write(
                "\n-----------------------------------------------------------------------------------------------------------\n"
                " Gen   Length            Fitness          Length             Fitness          OOB Fitness       Time Left\n"
            )
            for i in range(len(run_details['generation'])):
                f.write(
                    f"{run_details['generation'][i]:<4}   "  # Gen
                    f"{run_details['average_length'][i]:<15.2f}   "  # Length
                    f"{run_details['average_fitness'][i]:<15.5f}   "  # Fitness
                    f"{run_details['best_length'][i]:<15}   "  # Best Length
                    f"{run_details['best_fitness'][i]:<15.6f}   "  # Best Fitness
                    f"{run_details['best_oob_fitness'][i]:<15.6f}   "  # OOB Fitness
                    f"{run_details['generation_time'][i]:<6.2f}s\n"  # Time Left
                )

    def test(self, dataset) -> Dict[str, float]:
        data = self.initialize_dataset(dataset)
        X_test, y_test = data.features, data.labels
        X_test = X_test.fillna(0)
        
        if isinstance(self.model, SymbolicClassifier):
            y_hat = self.model.predict_proba(X_test)[:, 1]
        else:
            raise AttributeError(f"Unsupported xgboost model: {type(self.model)}")
        return self._get_metrics(y_test, y_hat)
    
    def predict(self, dataset) -> pd.Series:
        data = self.initialize_dataset(dataset)
        X_test, y_test = data.features, data.labels
        X_test = X_test.fillna(0)

        if isinstance(self.model, SymbolicClassifier):
            y_hat = self.model.predict_proba(X_test)[:, 1]
        else:
            raise AttributeError(f"Unsupported Symbolic model: {type(self.model)}")
        return pd.Series(y_hat, index=X_test.index)
    
class SoccerMapComponent(PressingComponent, exPressPytorchComponent):
    """A SoccerMap deep-learning model."""

    def __init__(self, 
                 model: PytorchSoccerMapModel, 
                 features, 
                 label, 
                 transform: ToSoccerMapTensor, 
                 params: dict):
        
        super().__init__(
            model=model,
            features=features,
            label=label,
            transform=transform,
            params=params
        )