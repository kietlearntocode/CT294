import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

save_path = '../deployment/rforest_model.pkl'

def train_Rforest(best_max_depth, best_min_samples_split, best_min_samples_leaf, n_estimators):
    X_train = pd.read_csv("fixed_split/X_train.csv")
    y_train = pd.read_csv("fixed_split/y_train.csv").squeeze()

    # ⚠️ SỬA LỖI ở đây: bạn đang truyền nhầm `min_samples_leaf=best_min_samples_split`
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=best_max_depth,
        min_samples_split=best_min_samples_split,
        min_samples_leaf=best_min_samples_leaf,
        random_state=42,
        criterion='gini',
        max_features='sqrt',
        class_weight='balanced',
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    # Tạo thư mục nếu chưa có
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    joblib.dump(model, save_path)
    print(f"Mô hình đã được lưu tại: {save_path}")

if __name__ == "__main__":
    train_Rforest(7, 8, 9, 100)
