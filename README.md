# sr-press
This repository presents exPress (Explainable & Expected Press), a framework for quantifying pressing situations in football by analyzing individual player positioning and movements

## installation and environment
```
git clone https://github.com/UoS-CIDA-Lab/exPress.git
cd exPress
pip install -r requirements.txt
```

## Library
- This project was developed using Python 3.9.20.
- Install the necessary libraries:
- To use the latest version of the socceraction library, you need to clone it directly from GitHub because it cannot be updated through standard installation methods.
- Since only the socceraction package is required, set socceraction/socceraction as the root directory and remove all other unnecessary files.
```bash
git clone https://github.com/statsbomb/open-data.git
git clone https://github.com/ML-KULeuven/socceraction.git
```

## How to access data
- To access the data, navigate to the [notebook](https://github.com/leemingo/sr-press/tree/geonhee/notebook).
    1. [1-load-and-convert-statsbomb-data.ipynb](https://github.com/GeonHeeJo2000/sr-press/blob/main/notebook/1-load-and-convert-statsbomb-data.ipynb): Load the StatsBomb dataset and build the training dataset. Estimated time: 8 minutes.
    2. [2-computing-and-storing-features.ipynb](https://github.com/GeonHeeJo2000/sr-press/blob/main/notebook/2-computing-and-storing-features.ipynb): Extract features and labels from the built dataset. Estimated time: 27 minutes.
- The data is stored in the [stores](https://github.com/leemingo/sr-press/tree/geonhee/stores) directory.
 * The stores directory contains the features and labels, but due to storage limitations, not all features have been processed and included here.

```
🗂️ stores
├── 📄 train_database.sqlite      # Database for training data
├── 📄 test_database.sqlite       # Database for testing data
├── 🗂️ model                      # Directory for storing trained models
└── 🗂️ datasets                   # Directory containing feature and label datasets
    ├── 🗂️ train                  
    │   ├── 📄 x_actiontype_onehot.parquet     
    │   ├── 📄 x_ball_height_onehot.parquet    
    │   ├── 📄 x_extract_all_players.parquet  
    │   ├── 📄 x_speed.parquet      
    │   ├── 📄 ...              
    │   └── 📄 y_counterpress.parquet          
    └── 🗂️ test                    
        ├── 📄 x_actiontype_onehot.parquet     
        ├── 📄 x_ball_height_onehot.parquet   
        ├── 📄 x_extract_all_players.parquet   
        ├── 📄 x_speed.parquet  
        ├── 📄 ...         
        └── 📄 y_counterpress.parquet  
```

## exPress Module
- [databases](https://github.com/leemingo/sr-press/tree/geonhee/express/databases): Data access and SPADL conversion (implemented _fix_pressure in base.py).
- [features](https://github.com/leemingo/sr-press/blob/geonhee/express/features.py), [labels](https://github.com/leemingo/sr-press/blob/geonhee/express/labels.py): Define features and labels.
- [component](https://github.com/leemingo/sr-press/tree/geonhee/express/components): Train the exPress model.
- [simulates.py](https://github.com/leemingo/sr-press/blob/geonhee/express/simulates.py): Predict xP based on various position changes.

## Training Module
- Once the features and labels are stored in the stores directory, you can perform training using the following steps:
    1. [3-evaluating-model-performance.ipynb](https://github.com/leemingo/sr-press/blob/main/notebook/3-evaluating-model-performance.ipynb): : Analyze the results of the training performed using [train.py](https://github.com/leemingo/sr-press/blob/main/train.py)
       
**Training the XGBoost Model**
```
python train.py \
--model xgboost \
--trial 0 \
--params_file params.json \
--nb_prev_actions 3 \
--xfns "startlocation" "closest_11_players" \
--yfns "counterpress"
```

**Training SoccerMap Model** <br/>
```
python train.py \
--model soccermap \
--trial 1 \
--params_file params.json \
--nb_prev_actions 3 \
--xfns "startlocation" "freeze_frame_360" \
--yfns "concede_shots"

```

### Research
please consider citing the following papers:
- Minho Lee, Geonhee Jo, Miru Hong, Pascal Bauer, and Sang-Ki Ko. **Contextual Valuation of Individual Players Within Pressing Situations in Football.** In 2025 MIT Sloan Sports Analytics Conference. <br/>[ [paper](https://www.sloansportsconference.com/research-papers/contextual-valuation-of-individual-players-within-pressing-situations-in-football) | [github](https://github.com/leemingo/sr-press) ]

### Affiliations
This research is conducted by **[Korea AI Research Society for Sports (KAISports)](https://sites.google.com/view/kaisport)**.  
