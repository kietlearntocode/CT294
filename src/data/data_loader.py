import os
import pandas as pd
from src.utils.path_utils import get_project_root

def load_data():
    BASE_DIR = get_project_root()
    data_path = os.path.join(BASE_DIR, 'data/raw/parkinsons.data')
    return pd.read_csv(data_path, sep=',')

def extract_features_labels():
    data = load_data()
    X = data.drop(columns=['name', 'status'])
    y = data['status']
    return X, y
