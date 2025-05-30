from Neural_Networks_Sub.Back_Propagation.back_propagation import backPropagation
from Neural_Networks_Sub.Neural_Network.test_neural_network import testNeuralNetwork
from Utils.data_set_reader import csv_reader
from Utils.pre_processing import treat_data
from Utils.metrics_visual import plot_roc_curve, plot_survival_distribution
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

    model = train_model()

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

    print(X_resampled)
    print(y_resampled)

    model = backPropagation(
        3, 
        3, 
        X_resampled.to_numpy().tolist(), 
        y_resampled.to_numpy().tolist(), 
        2000, 
        0.01, 
        "Titanic", 
        "tanh", 
        "batch"
    )

    return model
