import pandas as pd
import os

def split_train_and_demo(data_path):
    data = pd.read_csv(data_path)
    X = data.drop(columns=['name', 'status'])
    y = data['status']

    # Tìm 1 mẫu mỗi lớp làm test
    label_0_idx = y[y == 0].index[0]
    label_1_idx = y[y == 1].index[0]
    test_idx = [label_0_idx, label_1_idx]

    # Tạo thư mục
    os.makedirs("fixed_split", exist_ok=True)

    # Lưu tập test
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
    X_test.to_csv("fixed_split/X_test.csv", index=False)
    y_test.to_csv("fixed_split/y_test.csv", index=False)

    # Lưu tập train
    X_train, y_train = X.drop(index=test_idx), y.drop(index=test_idx)
    X_train.to_csv("fixed_split/X_train.csv", index=False)
    y_train.to_csv("fixed_split/y_train.csv", index=False)

if __name__ == "__main__":
    split_train_and_demo('..\data\parkinsons.data')