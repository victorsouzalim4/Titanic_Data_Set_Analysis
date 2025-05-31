from Utils.data_set_reader import csv_reader
from Utils.pre_processing import treat_data
from Utils.metrics_visual import (
    plot_silhouette,
    plot_tsne,
    plot_contingency_matrix,
    save_cluster_profile
)
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import pandas as pd
import os


def dbscan():
    os.makedirs("Analysis/DBSCAN", exist_ok=True)

    # Carregar os dados e os rótulos reais separados
    data_test = csv_reader("Titanic_data_set/test.csv")
    labels_test = pd.read_csv('Titanic_data_set/rf_mod_Solution.csv')

    columns_to_remove = ["PassengerId", "Ticket", "Cabin", "Embarked", "Name", "Fare"]
    essential_columns = ["Pclass", "Sex", "Age", "SibSp", "Parch"]
    one_hot_encoding_columns = ["Sex", "Pclass"]
    columns_to_normalize = ["Age", "SibSp", "Parch"]

    # Pré-processamento
    data_test = treat_data(
        data=data_test,
        columns_to_remove=columns_to_remove,
        essential_columns=essential_columns,
        one_hot_columns=one_hot_encoding_columns,
        columns_to_scale=columns_to_normalize
    )

    # Remover NaNs
    X_test = data_test.dropna().astype(float)

    # Alinhar y_test com as instâncias válidas restantes
    valid_indices = X_test.index
    y_test = labels_test.loc[valid_indices, 'Survived'].reset_index(drop=True)

    print(y_test)

    # Clustering com DBSCAN
    model = train_dbscan(X_test)
    cluster_labels_test = model.labels_

    # Verificação: pode haver ruído (label = -1)
    n_clusters = len(set(cluster_labels_test)) - (1 if -1 in cluster_labels_test else 0)
    print(f"Número de clusters encontrados (excluindo ruído): {n_clusters}")
    print(f"Rótulos de cluster DBSCAN: {set(cluster_labels_test)}")

    # Visualizações
    plot_silhouette(X_test, cluster_labels_test, "Analysis/DBSCAN/silhouette_dbscan_test.png",
                    title="Gráfico de Silhueta - Teste (DBSCAN)")

    plot_tsne(X_test, cluster_labels_test, "Analysis/DBSCAN/tsne_dbscan_test.png",
              title="Clusters no conjunto de TESTE (DBSCAN)")

    df_test = pd.DataFrame(X_test)
    df_test['Survived'] = y_test
    df_test['Cluster'] = cluster_labels_test

    print("\nTabela de Contingência entre Clusters e Classe Real:")
    contingency = plot_contingency_matrix(df_test, "Analysis/DBSCAN/contingency_matrix.png")
    print(contingency)

    print("\nPerfil médio de cada cluster:")
    cluster_profile = save_cluster_profile(df_test,
                                           "Analysis/DBSCAN/cluster_profiles.csv",
                                           "Analysis/DBSCAN/cluster_profiles_table.png")
    print(cluster_profile)


def train_dbscan(X, eps=0.5, min_samples=5):
    """
    Treina o DBSCAN com parâmetros ajustáveis.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    db = DBSCAN(eps=eps, min_samples=min_samples)
    db.fit(X_scaled)
    return db
