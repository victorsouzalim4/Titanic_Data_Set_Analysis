from Neural_Networks_Sub.Back_Propagation.back_propagation import backPropagation
from Neural_Networks_Sub.Neural_Network.test_neural_network import testNeuralNetwork
from Utils.data_set_reader import csv_reader
from Utils.pre_processing import treat_data
from Utils.metrics_visual import plot_roc_curve, plot_survival_distribution, save_confusion_matrix, save_classification_report_image
from imblearn.over_sampling import SMOTE
import pandas as pd

def neural_network():

    true_labels = pd.read_csv('Titanic_data_set/rf_mod_Solution.csv')
    data_test = csv_reader("Titanic_data_set/test.csv")

    columns_to_remove = ["PassengerId", "Ticket", "Cabin", "Embarked", "Name", "Fare"]
    essential_columns = ["Pclass", "Sex", "Age", "SibSp", "Parch"]
    one_hot_encoding_columns = ["Sex", "Pclass"]
    columns_to_normalize = ["Age", "SibSp", "Parch"]

    data_test = treat_data(
        data = data_test, 
        columns_to_remove = columns_to_remove, 
        essential_columns = essential_columns, 
        one_hot_columns = one_hot_encoding_columns,
        columns_to_scale = columns_to_normalize
    )

    X_test = data_test.dropna().astype(float)

    model = train_model()


    left_indexes = X_test.index

    y_true = true_labels['Survived'].replace(0, -1)
    y_test = testNeuralNetwork(layers = model, test_inputs = X_test, test_expected_outputs = y_true, activation="tanh")

    y_true_clean = y_true.loc[left_indexes]

    # y_proba_test = model.predict_proba(X_test)[:, 1]

    # plot_roc_curve(
    #     y_true=y_true_clean,
    #     y_proba=y_proba_test,
    #     title='Curva ROC - Random Forest (Test)',
    #     save_path='Analysis/Neural_network/roc_curve_rf_test.png',
    #     show=True
    # )

    y_test_metrics = [0 if y == -1 else 1 for y in y_test]
    y_true_metrics = y_true_clean.replace(-1, 0)

    save_confusion_matrix(
        y_true=y_true_metrics,
        y_pred=y_test_metrics,
        class_labels=["Died", "Survived"],
        save_path="Analysis/Neural_network/conf_matrix_rf_test.png",
        show=True,
        title="Confusion Matrix - Test Set"
    )

    save_classification_report_image(
        y_true=y_true_metrics,
        y_pred=y_test_metrics,
        class_labels=["Died", "Survived"],
        save_path="Analysis/Neural_network/classification_report_rf_test.png",
        title="Classification Report - Neural Network"
    )

    return y_true_metrics

def train_model():

    data_train = csv_reader("Titanic_data_set/train.csv")

    columns_to_remove = ["PassengerId", "Ticket", "Cabin", "Embarked", "Name", "Fare"]
    essential_columns = ["Survived", "Pclass", "Sex", "Age", "SibSp", "Parch"]
    one_hot_encoding_columns = ["Sex", "Pclass"]
    columns_to_normalize = ["Age", "SibSp", "Parch"]

    data_train = treat_data(
        data = data_train, 
        columns_to_remove = columns_to_remove, 
        essential_columns = essential_columns, 
        one_hot_columns = one_hot_encoding_columns,
        columns_to_scale = columns_to_normalize
    )

    X_train = data_train.drop('Survived', axis=1).astype(float)
    y_train = data_train['Survived'].replace(0, -1)



    plot_survival_distribution(
        train_labels=y_train,
        title='Original Survival Distribution - Train Set',
        save_path='Analysis/Neural_network/original_distribution.png',
        show=True
    )

    smote = SMOTE(random_state=42, k_neighbors=3)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    plot_survival_distribution(
        train_labels=y_resampled,
        title='After SMOTE (Oversampling)',
        save_path='Analysis/Neural_network/after_smote_distribution.png',
        show=True
    )

    print(X_resampled)
    print(y_resampled)

    model = backPropagation(
        4, 
        3, 
        X_resampled.astype(float).to_numpy().tolist(), 
        y_resampled.to_numpy().tolist(), 
        1000, 
        0.01, 
        "Titanic", 
        "tanh", 
        "batch"
    )

    return model
