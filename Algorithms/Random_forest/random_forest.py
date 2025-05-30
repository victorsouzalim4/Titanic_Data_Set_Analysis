from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from Utils.data_set_reader import csv_reader
from Utils.pre_processing import treat_data
from Utils.metrics_visual import save_confusion_matrix, save_classification_report_image, plot_roc_curve, plot_survival_distribution
import pandas as pd 
from sklearn.tree import export_text, plot_tree
import matplotlib.pyplot as plt

def random_forest():

    true_labels = pd.read_csv('Titanic_data_set/rf_mod_Solution.csv')
    data_test = csv_reader("Titanic_data_set/test.csv")

    columns_to_remove = ["PassengerId", "Ticket", "Cabin", "Embarked", "Name", "Fare"]
    essential_columns = ["Pclass", "Sex", "Age", "SibSp", "Parch"]
    label_encoding_columns = ["Sex"]

    data_test = treat_data(
        data = data_test, 
        columns_to_remove = columns_to_remove, 
        essential_columns = essential_columns, 
        label_encoding_columns = label_encoding_columns
    )

    model = forest_training()

    X_test = data_test.dropna()
    left_indexes = X_test.index

    y_test = model.predict(X_test)
    y_true = true_labels['Survived']

    y_true_clean = y_true.loc[left_indexes]

    y_proba_test = model.predict_proba(X_test)[:, 1]

    plot_roc_curve(
        y_true=y_true_clean,
        y_proba=y_proba_test,
        title='Curva ROC - Random Forest (Test)',
        save_path='Analysis/Random_forest/roc_curve_rf_test.png',
        show=True
    )

    save_confusion_matrix(
        y_true=y_true_clean,
        y_pred=y_test,
        class_labels=["Died", "Survived"],
        save_path="Analysis/Random_forest/conf_matrix_rf_test.png",
        show=True,
        title="Confusion Matrix - Test Set"
    )

    save_classification_report_image(
        y_true=y_true_clean,
        y_pred=y_test,
        class_labels=["Died", "Survived"],
        save_path="Analysis/Random_forest/classification_report_rf_test.png",
        title="Classification Report - Random Forest"
    )


def forest_training():

    data_train = csv_reader("Titanic_data_set/train.csv")

    columns_to_remove = ["PassengerId", "Ticket", "Cabin", "Embarked", "Name", "Fare"]
    essential_columns = ["Survived", "Pclass", "Sex", "Age", "SibSp", "Parch"]
    label_encoding_columns = ["Sex"]

    data_test = treat_data(
        data = data_test, 
        columns_to_remove = columns_to_remove, 
        essential_columns = essential_columns, 
        label_encoding_columns = label_encoding_columns
    )

    X_train = data_train.drop('Survived', axis=1)
    y_train = data_train['Survived']

    plot_survival_distribution(
        train_labels=y_train,
        title='Original Survival Distribution - Train Set',
        save_path='Analysis/Random_Forest/original_distribution.png',
        show=True
    )

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    plot_survival_distribution(
        train_labels=y_resampled,
        title='After SMOTE (Oversampling)',
        save_path='Analysis/Random_Forest/after_smote_distribution.png',
        show=True
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=7,
        min_samples_split=5,
        max_features='sqrt',
        random_state=42
    )
    model.fit(X_resampled, y_resampled)

            # Seleciona uma árvore da floresta
    estimator = model.estimators_[0]  # ou outro índice

    # Texto explicando regras da árvore
    tree_rules = export_text(estimator, feature_names=list(X_train.columns))
    print(tree_rules)

    # Visualização gráfica
    plt.figure(figsize=(20, 10))
    plot_tree(estimator, feature_names=X_train.columns, class_names=["Died", "Survived"], filled=True)
    plt.show()

    return model
