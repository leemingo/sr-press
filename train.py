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
from express.components.model import PytorchModel, BaseLine
from express.features import state, pressure
from express import labels as ls

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
        "--state_xfns",
        type=str,
        nargs="+",
        default=["startlocation"],
        help="Feature functions to use (default: ['startlocation'])."
    )
    parser.add_argument(
        "--pressure_state_xfns",
        type=str,
        nargs="+",
        default=[],
        help="Feature functions to use (default: [])."
    )
    parser.add_argument(
        "--pressure_with_context_xfns",
        type=str,
        nargs="+",
        default=[],
        help="Feature functions to use (default: [])."
    )
    parser.add_argument(
        "--yfns",
        type=str,
        nargs="+",
        default=["possession_change_by_4_actions"],
        help="Label functions to use (default: ['possession_change_by_4_actions'])."
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
    
    # Save params.json to save_path
    with open(os.path.join(save_path, args.params_file), 'w') as f:
        json.dump(params, f, indent=4)

    # Save shell command to a text file
    with open(os.path.join(save_path, "train_model_sh"), 'w') as f:
        f.write(" ".join(sys.argv))

    dataset_train = partial(PressingDataset, path=os.path.join(base_path, "stores", "datasets", "train"))
    dataset_test = partial(PressingDataset, path=os.path.join(base_path, "stores", "datasets", "test"))
    
    train_dataset = PressingDataset(
        path= os.path.join(base_path, "stores", "datasets", "train"),
        state_xfns=args.state_xfns,
        pressure_xfns={"pressure_state_xfns": args.pressure_state_xfns,
                        "pressure_with_context_xfns": args.pressure_with_context_xfns},
        yfns=args.yfns,
        load_cached =True,
        nb_prev_actions = 3,
    )

    test_dataset = PressingDataset(
        path= os.path.join(base_path, "stores", "datasets", "test"),
        state_xfns=args.state_xfns,
        pressure_xfns={"pressure_state_xfns": args.pressure_state_xfns,
                        "pressure_with_context_xfns": args.pressure_with_context_xfns},
        yfns=args.yfns,
        load_cached =True,
        nb_prev_actions = 3,
    )
    print(f"\nTrain DataSet: {train_dataset.features.shape}, Test DataSet: {test_dataset.features.shape}")
    print(f"Train Dataset {args.yfns}: {train_dataset.labels[args.yfns].value_counts().to_dict()}, Test Dataset: {test_dataset.labels[args.yfns].value_counts().to_dict()}\n")
    
    # Compute features and labels
    state_features = {}
    pressure_state_features = {}
    pressure_with_context_features = {}

    for xfn in train_dataset.state_xfns.items():
        key = xfn[0].__name__ # Callable: function
        values = xfn[1] # List: columns

        # 설명: nb_prev_actions는 현재 액션을 포함해 사용할 액션의 수를 지정합니다.
        # nb_prev_actions=1: {a_i, p_i}
        # nb_prev_actions=2: {a_{i-1}, p_i}
        # nb_prev_actions=3: {a_{i-2}, a_{i-1}, p_i}
        # a_i정보를 사용하고 싶지 않으면 state_xfns를 명령어 쉘에서 제외하면 된다. -> a_i로 부터 생성되는 feature를 추출하지 않는다.
        state_features[key] = []
        for value in values:
            if value[-1:].isdigit():  
                if int(value[-1:]) < args.nb_prev_actions:
                    state_features[key].append(value)
            else:  # context features: ['goalscore_team', 'goalscore_opponent', 'goalscore_diff'...]
                state_features[key].append(value)

    for xfn in train_dataset.pressure_state_xfns.items():
        key = xfn[0].__name__ # Callable: function
        values = xfn[1] # List: columns
        pressure_state_features[key] = values

    for xfn in train_dataset.pressure_with_context_xfns.items():
        key = xfn[0].__name__ # Callable: function
        values = xfn[1] # List: columns

        pressure_with_context_features[key] = values
    features = {"state_xfns": state_features, 
                "pressure_xfns": {"pressure_state_xfns": pressure_state_features, "pressure_with_context_xfns": pressure_with_context_features}}

    if args.model == "xgboost":
        scale_pos_weight = train_dataset.labels[args.yfns].value_counts()[False] / train_dataset.labels[args.yfns].value_counts()[True]
        print("scale_pos_weight: ", scale_pos_weight)
        params["scale_pos_weight"] = scale_pos_weight
        
        component= press.XGBoostComponent(
            model = XGBClassifier(**params["ModelConfig"]),
            features = features,
            label = args.yfns,
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
            label = args.yfns,
            params = params
        )
    elif args.model == "soccermap":
        component= press.SoccerMapComponent(
            model = PytorchSoccerMapModel(model_config= params["ModelConfig"], 
                                          optimizer_params= params["OptimizerConfig"]),
            features = features,
            label = args.yfns,
            transform = ToSoccerMapTensor(dim=(68, 105)),
            params=params,
        )
    elif args.model == "baseline":
        component= press.SoccerMapComponent(
            model = PytorchModel(model=BaseLine(params["ModelConfig"]), 
                                 optimizer_params= params["OptimizerConfig"]),
            features = features,
            label = args.yfns,
            transform = ToSoccerMapTensor(dim=(68, 105)),
            params=params,
        )
    else:
        raise ValueError(f"Invalid model: {args.model}")

    # If parameter tuning is required, pass `param_grid` and 'optimized_metric' to the train function
    optimized_metric= "f1_weighted"
    param_grid = {                      # xgboost paramter(small version)
        'n_estimators': [50, 100],  
        'max_depth': [4, 6],  
        'learning_rate': [0.05, 0.1],  
        'subsample': [0.8, 1.0],  
        'scale_pos_weight': [1, 2, 5], 
    }
    # param_grid = {                      # xgboost paramter(small version)
    #     'n_estimators': [50, 100],  
    #     'max_depth': [4, 6],  
    #     'learning_rate': [0.05, 0.1],  
    #     'subsample': [0.8, 1.0],  
    #     'min_child_weight': [1, 3]  
    # }
    param_grid = {
        'n_estimators': [50, 100],              
        'max_depth': [4, 6, 8],                 
        'learning_rate': [0.01, 0.05, 0.1],    
        'subsample': [0.6, 0.8, 1.0],            
        'colsample_bytree': [0.6, 0.8, 1.0],
        'min_child_weight': [1, 3, 5],         
        'scale_pos_weight': [1, 2, 5],        
    }

    try:
        component.train(dataset_train, param_grid= None, optimized_metric= None)
        #component.train(dataset_train, param_grid= param_grid, optimized_metric= optimized_metric)
    except Exception as e: 
        print(f"예외 발생: {e}")  
        os.rmdir(save_path)
        print(f"\n{save_path} 디렉토리가 삭제되었습니다.")
        exit()

    print(f"########## Train Metrics ##########\n{component.test(dataset_train)}\n")
    print(f"########## Test Metrics ##########\n{component.test(dataset_test)}\n")
    component.save(Path(os.path.join(save_path, "component.pkl")))
