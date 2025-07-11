from sklearn.tree import DecisionTreeClassifier

def build_dtree(**kwargs):
    model = DecisionTreeClassifier(**kwargs)
    return model