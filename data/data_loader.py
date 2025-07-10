import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'parkinsons.data')

data = pd.read_csv(DATA_PATH, sep=',')

def data_frame():
    return data

def extract_features_labels():
    X = data.drop(columns=['name', 'status'])
    y = data['status']
    return X, y