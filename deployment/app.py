import joblib
import pandas as pd
RForest_model = joblib.load('rforest_model.pkl')


def predict(sample):
    print(sample.shape)
    return RForest_model.predict(sample)

def data_demo():
    X_test = pd.read_csv("../train/fixed_split/X_test.csv")
    y_test = pd.read_csv("../train/fixed_split/y_test.csv").squeeze()
    y_pred = predict(X_test)
    print("=== DỰ ĐOÁN TRÊN 2 MẪU DEMO ===")
    for i in range(len(y_pred)):
        print(f"Mẫu {i + 1}: Dự đoán = {y_pred[i]} | Thật = {y_test.iloc[i]}")

if __name__ == "__main__":
    data_demo()