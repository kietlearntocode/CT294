import pandas as pd

path = '..\data\parkinsons.data'
data = pd.read_csv(path, sep=',')

def data_frame():
    return data

def extract_features_labels():
    X = data.drop(columns=['name', 'status'])
    y = data['status']
    return X, y