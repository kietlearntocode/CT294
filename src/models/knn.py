from sklearn.neighbors import KNeighborsClassifier

def build_knn(**kwargs):
    model = KNeighborsClassifier(**kwargs)
    return model