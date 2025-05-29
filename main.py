from Utils.data_set_reader import csv_reader
from Utils.pre_processing import treat_data
from sklearn.preprocessing import LabelEncoder
from Algorithms.random_forest import random_forest

import pandas as pd 


file_name_train = "titanic_data_set/train.csv"
file_name_test = "titanic_data_set/test.csv"

data_train = csv_reader(file_name_train)
data_test = csv_reader(file_name_test)

columns_to_remove = ["PassengerId", "Ticket", "Cabin", "Embarked", "Name", "Fare"]
essential_columns = ["Survived", "Pclass", "Sex", "Age", "SibSp", "Parch"]
label_encoding_columns = ["Sex"]

data_train = treat_data(data_train, columns_to_remove, essential_columns, label_encoding_columns)

essential_columns = ["Pclass", "Sex", "Age", "SibSp", "Parch"]

data_test = treat_data(data_test, columns_to_remove, essential_columns, label_encoding_columns)

X_train = data_train.drop('Survived', axis=1)
y_train = data_train['Survived']

X_test = data_test

print(X_train)
print(X_test)

model = random_forest(X_train, y_train)

y_test = model.predict(X_test)

print(y_test)

