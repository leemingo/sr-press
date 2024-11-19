import os
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm

base_path = os.path.abspath(os.path.join(os.getcwd()))
sys.path.append(base_path)

from functools import partial
from xgboost import XGBClassifier
from gplearn.genetic import SymbolicClassifier
import json
import argparse

from express.datasets import PressingDataset
from express.components import press
from express.components.soccermap import PytorchSoccerMapModel, ToSoccerMapTensor

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model type (e.g., xgboost, symbolic_regression, soccermap)."
    )
    parser.add_argument(
        "--trial",
        type=int,
        required=True,
        help="Trial number for the experiment."
    )

    parser.add_argument(
        "--params_file",
        type=str,
        required=True,
        help="Path to the JSON file containing model parameters."
    )
    parser.add_argument(
        "--optimized_metric",
        type=str,
        default=None,
        help="Metric to optimize (e.g., f1_weighted)."
    )

    parser.add_argument(
        "--nb_prev_actions",
        type=int,
        default=1,
        help="Number of previous actions to consider (default: 1, only current action)."
    )
    parser.add_argument(
        "--xfns",
        type=str,
        nargs="+",
        default=["startlocation"],
        help="Feature functions to use (default: ['startlocation'])."
    )
    parser.add_argument(
        "--yfns",
        type=str,
        nargs="+",
        default=["counterpress"],
        help="Label functions to use (default: ['counterpress'])."
    )

    args = parser.parse_args()

    params = {}
    if args.params_file:
        with open(args.params_file, 'r') as f:
            all_params = json.load(f)
            params = all_params.get(args.model, {}) # get model's parameter

    args = parser.parse_args()

    model_path = os.path.join(base_path, "stores", "model", args.model)
    save_path = os.path.join(model_path, f"{args.trial:03d}")
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        params["save_path"] = save_path
    else:
        raise FileExistsError(f"The directory '{save_path}' already exists.")

    dataset_train = partial(PressingDataset, path=os.path.join(base_path, "stores", "datasets", "train"))
    dataset_test = partial(PressingDataset, path=os.path.join(base_path, "stores", "datasets", "test"))
    
    train_dataset = PressingDataset(
        path= os.path.join(base_path, "stores", "datasets", "train"),
        xfns= args.xfns,
        yfns= args.yfns,
        load_cached= True,
        nb_prev_actions= args.nb_prev_actions
    )
    test_dataset = PressingDataset(
        path= os.path.join(base_path, "stores", "datasets", "test"),
        xfns= args.xfns,
        yfns= args.yfns,
        load_cached= True,
        nb_prev_actions= args.nb_prev_actions
    )
    print(f"\nTrain DataSet: {train_dataset.features.shape}, Test DataSet: {test_dataset.features.shape}")
    print(f"Train Dataset: {train_dataset.labels.value_counts().to_dict()}, Test Dataset: {test_dataset.labels.value_counts().to_dict()}\n")

    # Compute features and labels
    features = {}
    label = args.yfns
    for xfn in train_dataset.xfns.items():
        key = xfn[0].__name__ # Callable: function
        values = xfn[1] # List: columns

    
        # 설명: nb_prev_actions는 현재 액션을 포함해 사용할 액션의 수를 지정합니다.
        # nb_prev_actions=1: 현재 액션만 사용
        # nb_prev_actions=2: 현재 액션 + 이전 액션
        if key in args.xfns:
            features[key] = [value for value in values if int(value[-1:]) < args.nb_prev_actions] 

    if args.model == "xgboost":
        component= press.XGBoostComponent(
            model = XGBClassifier(**params["ModelConfig"]),
            features = features,
            label = label,
            params = params
        )
    elif args.model == "symbolic_regression":
        params["feature_names"] = [fs for fs_lst in features.values() for fs in fs_lst]
        for key in ["const_range", "init_depth"]:
            if key in params.get("ModelConfig", {}):
                params["ModelConfig"][key] = tuple(params["ModelConfig"][key])

        component= press.SymbolicComponent(
            model = SymbolicClassifier(**params["ModelConfig"]),
            features = features,
            label = label,
            params = params
        )
    elif args.model == "soccermap":
        component= press.SoccerMapComponent(
            model = PytorchSoccerMapModel(model_config= params["ModelConfig"], 
                                          optimizer_params= params["OptimizerConfig"]),
            features = features,
            label = label,
            transform = ToSoccerMapTensor(dim=(68, 104)),
            params=params,
        )
    else:
        raise ValueError(f"Invalid model: {args.model}")

    # If parameter tuning is required, pass `param_grid` and 'optimized_metric' to the train function
    optimized_metric= "f1_weighted"
    param_grid = {                      # xgboost paramter
        'n_estimators': [100, 200],  
        'max_depth': [4, 6],  
        'learning_rate': [0.05, 0.1],  
        'subsample': [0.8, 1.0],  
        'colsample_bytree': [0.8, 1.0], 
        'min_child_weight': [1, 3]  
    }

    component.train(dataset_train, param_grid= None, optimized_metric= None)

    print(f"########## Train Metrics ##########\n{component.test(dataset_train)}\n")
    print(f"########## Test Metrics ##########\n{component.test(dataset_test)}\n")
    component.save(Path(os.path.join(save_path, "component.pkl")))
