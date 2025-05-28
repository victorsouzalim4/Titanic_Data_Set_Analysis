from Utils.data_set_reader import csv_reader

file_name = "titanic_data_set/train.csv"

columns_to_remove = ["PassengerId", "Ticket", "Cabin", "Embarked", "Name"]

essential_columns = ["Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]

data = csv_reader(file_name, columns_to_remove, essential_columns)

print(len(data))