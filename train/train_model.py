import pandas as pd
import joblib
import yaml
import os
from src.models.rforest import build_rforest
from src.utils.path_utils import get_project_root

def train_rforest(X_path, y_path, save_path, **model_params):
    X_train = pd.read_csv(X_path)
    y_train = pd.read_csv(y_path).squeeze()

    model = build_rforest(**model_params)

    model.fit(X_train, y_train)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(model, save_path)

if __name__ == '__main__':
    BASE_DIR = get_project_root()
    config_path = os.path.join(BASE_DIR, "config", "train_config.yml")

    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

        model_params = config["model_params"]
        data_cfg = config["data_paths"]

        X_path = os.path.join(BASE_DIR, data_cfg["X_train_path"])
        y_path = os.path.join(BASE_DIR, data_cfg["y_train_path"])
        save_path = os.path.join(BASE_DIR, data_cfg["save_model_path"])

        train_rforest(X_path, y_path, save_path, **model_params)
        print("training success!")


