import pandas as pd
import os
from src.data.data_loader import extract_features_labels
from src.utils.path_utils import get_project_root

def _split_demo_sample(X, y):
    label_0_idx = y[y == 0].index[0]
    label_1_idx = y[y == 1].index[0]
    test_idx = [label_0_idx, label_1_idx]

    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
    X_train, y_train = X.drop(index=test_idx), y.drop(index=test_idx)

    return X_train, X_test, y_train, y_test

def _save_split(X_train, X_test, y_train, y_test):
    BASE_DIR = get_project_root()
    output_dir = os.path.join(BASE_DIR, "data/processed")
    os.makedirs(output_dir, exist_ok=True)
    X_test.to_csv(os.path.join(BASE_DIR, "data/processed/X_test.csv"), index=False)
    y_test.to_csv(os.path.join(BASE_DIR, "data/processed/y_test.csv"), index=False)
    X_train.to_csv(os.path.join(BASE_DIR, "data/processed/X_train.csv"), index=False)
    y_train.to_csv(os.path.join(BASE_DIR, "data/processed/y_train.csv"), index=False)

def run_split():
    X, y = extract_features_labels()
    X_train, X_test, y_train, y_test = _split_demo_sample(X, y)
    _save_split(X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    run_split()
    print("Đã tạo thành công")