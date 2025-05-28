from Utils.data_set_reader import csv_reader
from Utils.pre_processing import treat_data

file_name = "titanic_data_set/train.csv"

data = csv_reader(file_name)

columns_to_remove = ["PassengerId", "Ticket", "Cabin", "Embarked", "Name"]
essential_columns = ["Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]
label_encoding_columns = ["Sex"]

data = treat_data(data, columns_to_remove, essential_columns, label_encoding_columns)

print(data)