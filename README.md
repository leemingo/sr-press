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
    └── 🗂️ test                    - 
        ├── 📄 x_actiontype_onehot.parquet     
        ├── 📄 x_ball_height_onehot.parquet   
        ├── 📄 x_extract_all_players.parquet   
        ├── 📄 x_speed.parquet  
        ├── 📄 ...         
        └── 📄 y_counterpress.parquet  
```
