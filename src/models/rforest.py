from sklearn.ensemble import RandomForestClassifier

def build_rforest(**kwargs):
    model = RandomForestClassifier(**kwargs)
    return model