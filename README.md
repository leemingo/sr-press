# sr-press
This is the repository for valuing individual players within pressing situation in football with symbolic regression methods.

### Library
- This project was developed using Python 3.9.20.
- Install the necessary libraries:
- To use the latest version of the socceraction library, you need to clone it directly from GitHub because it cannot be updated through standard installation methods.
```bash
git clone https://github.com/statsbomb/open-data.git
git clone https://github.com/ML-KULeuven/socceraction.git
```

### How to access data
- To access the data, navigate to the [notebook](https://github.com/leemingo/sr-press/tree/geonhee/notebook).
- The data is stored in the [stores](https://github.com/leemingo/sr-press/tree/geonhee/stores).
 * The `stores` directory contains the features and labels, but due to storage limitations, not all features have been processed and included here. For some larger features (e.g., `extract_all_players.parquet`), please download them directly from [Google Drive](https://drive.google.com/drive/folders/1MJ5UIXgQ2EEcQNstz4Jq2uNRGdOPKLl-).
   
### exPress Module
- [databases](https://github.com/leemingo/sr-press/tree/geonhee/express/databases): Data access and SPADL conversion (implemented _fix_pressure in base.py).
- [features](https://github.com/leemingo/sr-press/blob/geonhee/express/features.py), [labels](https://github.com/leemingo/sr-press/blob/geonhee/express/labels.py): Define features and labels
- [component](https://github.com/leemingo/sr-press/tree/geonhee/express/components): Train the exPress model
