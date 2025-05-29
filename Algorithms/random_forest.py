from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def random_forest(X_train, y_train):

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        max_features='sqrt',
        random_state=42
    )

    model.fit(X_train, y_train)

    return model

