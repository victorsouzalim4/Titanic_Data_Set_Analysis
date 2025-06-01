from Utils.data_set_reader import csv_reader
from Utils.pre_processing import treat_data
from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd

def Apriori():

    true_labels = pd.read_csv('Titanic_data_set/rf_mod_Solution.csv')
    data_test = csv_reader("Titanic_data_set/test.csv")
    data_test['Survived'] = true_labels['Survived']

    columns_to_remove = ["PassengerId", "Ticket", "Cabin", "Embarked", "Name", "Fare"]
    essential_columns = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Survived"]
    binning_encoding = [["Age range"], ["Age"], [['Jovem', 'Adulto', 'Meia-idade', 'Sênior']]]
    one_hot_encoding = ["Pclass", "Age range", "Sex", "Survived"]

    data_test = treat_data(
        data = data_test, 
        columns_to_remove = columns_to_remove, 
        essential_columns = essential_columns, 
        binning_encoding = binning_encoding
    )

    data_test = treat_data(
        data = data_test,
        one_hot_columns = one_hot_encoding,
    )

    data_test['Parch range'] = data_test['Parch'].apply(lambda x: 0 if x >= 2 else 1)
    data_test['SibSp range'] = data_test['SibSp'].apply(lambda x: 0 if x <= 2 else 1)
    data_test = data_test.drop(["Parch", "SibSp"], axis=1, errors='ignore')


    print(data_test)
    

    frequent_itemsets = apriori(data_test, min_support=0.1, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)

    rules_filtered = rules[
    (rules['lift'] > 1)
    ][['antecedents', 'consequents', 'support', 'confidence', 'lift']]


    print(rules_filtered)

    rules_filtered.to_csv("regras_filtradas.csv", index=False)
