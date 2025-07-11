import yaml
import os

config = {
    "model_params": {
        "n_estimators": 300,
        "max_depth": 8,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "random_state": 42,
        "class_weight": "balanced",
        "criterion": "gini",
        "max_features": "sqrt",
        "n_jobs": -1
    },
    "data_paths": {
        "X_train_path": "data/processed/X_train.csv",
        "y_train_path": "data/processed/y_train.csv",
        "save_model_path": "deployment/rforest_model.pkl"
    }
}

os.makedirs("config", exist_ok=True)
with open(os.path.join("config", "train_config.yml"), "w") as f:
    yaml.dump(config, f)

print("Đã tạo thành công")