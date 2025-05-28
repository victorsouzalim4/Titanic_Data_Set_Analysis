from Utils.data_set_reader import csv_reader
from Utils.pre_processing import treat_data
import pandas as pd 
from sklearn.preprocessing import LabelEncoder


file_name = "titanic_data_set/train.csv"

data_train = csv_reader(file_name)
data_test = csv_reader(file_name)

columns_to_remove = ["PassengerId", "Ticket", "Cabin", "Embarked", "Name", "Fare"]
essential_columns = ["Survived", "Pclass", "Sex", "Age", "SibSp", "Parch"]
label_encoding_columns = ["Sex"]

data_train = treat_data(data_train, columns_to_remove, essential_columns, label_encoding_columns)
data_test = treat_data(data_test, columns_to_remove, essential_columns, label_encoding_columns)

X_train = data_train.drop('Survived', axis=1)
y_train = data_train['Survived']

X_test = data_test

print(data_train)