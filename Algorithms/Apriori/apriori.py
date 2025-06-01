from Utils.data_set_reader import csv_reader
from Utils.pre_processing import treat_data
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
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
    

    frequent_itemsets = apriori(data_test, min_support=0.3, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)

    rules_filtered = rules[
    (rules['lift'] > 1)
    ][['antecedents', 'consequents', 'support', 'confidence', 'lift']]


    print(rules_filtered)

    rules_filtered.to_csv("regras_filtradas.csv", index=False)

    #scatter plot
    plt.figure(figsize=(10,6))
    plt.scatter(rules_filtered['support'], rules_filtered['confidence'],
                s=rules_filtered['lift']*50, alpha=0.6)
    plt.xlabel('Support')
    plt.ylabel('Confidence')
    plt.title('Regras de Associação')
    plt.grid(True)
    plt.savefig("Analysis/Apriori/scatter_regras.png")  # 🔽 salva o gráfico
    plt.show()

    
    #heatmap
    pivot = rules_filtered.pivot_table(index='antecedents', columns='consequents', values='lift', fill_value=0)

    plt.figure(figsize=(14,10))  # um pouco maior
    sns.heatmap(pivot, annot=True, cmap='coolwarm', fmt=".1f")

    plt.title('Lift entre antecedents e consequents', fontsize=14)
    plt.xticks(rotation=45, ha='right')   # rotaciona colunas
    plt.yticks(rotation=0)                # mantém linhas horizontais
    plt.tight_layout()                    # evita corte de rótulos

    plt.savefig("heatmap_regras_legivel.png", dpi=300)
    plt.show()



    #graph
    G = nx.DiGraph()

    for _, row in rules_filtered.iterrows():
        for antecedent in row['antecedents']:
            for consequent in row['consequents']:
                G.add_edge(antecedent, consequent, weight=row['lift'])

    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, k=0.5)
    edges = G.edges(data=True)
    weights = [edge[2]['weight'] for edge in edges]

    nx.draw(G, pos, with_labels=True, node_color='skyblue', edge_color=weights,
            edge_cmap=plt.cm.viridis, width=2, arrowsize=15)
    plt.title('Rede de Regras de Associação')
    plt.savefig("Analysis/Apriori/grafo_regras.png")  # 🔽 salva o gráfico
    plt.show()


    rules_filtered.sort_values(by='lift', ascending=False).style.background_gradient(cmap='YlGnBu')


