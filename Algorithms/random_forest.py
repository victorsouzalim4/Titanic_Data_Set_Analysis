from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from Utils.data_set_reader import csv_reader
from Utils.pre_processing import treat_data
from Utils.metrics_visual import save_confusion_matrix, save_classification_report_image
import pandas as pd 

def random_forest():

    true_labels = pd.read_csv('Titanic_data_set/rf_mod_Solution.csv')
    data_test = csv_reader("Titanic_data_set/test.csv")

    columns_to_remove = ["PassengerId", "Ticket", "Cabin", "Embarked", "Name", "Fare"]
    essential_columns = ["Pclass", "Sex", "Age", "SibSp", "Parch"]
    label_encoding_columns = ["Sex"]

    data_test = treat_data(data_test, columns_to_remove, essential_columns, label_encoding_columns)

    model = forest_training()

    X_test = data_test.dropna()
    left_indexes = X_test.index

    y_test = model.predict(X_test)
    y_true = true_labels['Survived']

    y_true_clean = y_true.loc[left_indexes]

    save_confusion_matrix(
        y_true=y_true_clean,
        y_pred=y_test,
        class_labels=["Died", "Survived"],
        save_path="Analysis/Random_forest/conf_matrix_rf.png",
        show=True,
        title="Confusion Matrix - Test Set"
    )

    save_classification_report_image(
        y_true=y_true_clean,
        y_pred=y_test,
        class_labels=["Died", "Survived"],
        save_path="Analysis/Random_Forest/classification_report_rf.png",
        title="Classification Report - Random Forest"
    )



def forest_training():

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=7,
        min_samples_split=5,
        max_features='sqrt',
        random_state=42
    )

    data_train = csv_reader("Titanic_data_set/train.csv")

    columns_to_remove = ["PassengerId", "Ticket", "Cabin", "Embarked", "Name", "Fare"]
    essential_columns = ["Survived", "Pclass", "Sex", "Age", "SibSp", "Parch"]
    label_encoding_columns = ["Sex"]

    data_train = treat_data(data_train, columns_to_remove, essential_columns, label_encoding_columns)

    X_train = data_train.drop('Survived', axis=1)
    y_train = data_train['Survived']

    model.fit(X_train, y_train)

    return model