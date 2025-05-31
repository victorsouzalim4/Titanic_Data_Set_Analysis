
from Utils.data_set_reader import csv_reader
from Utils.pre_processing import treat_data
from Utils.metrics_visual import(
    plot_silhouette,
    plot_tsne,
    plot_contingency_matrix,
    save_cluster_profile
)
from imblearn.over_sampling import SMOTE
from sklearn.cluster import KMeans
import pandas as pd
import os


def k_means():
    os.makedirs("Analysis/KMeans", exist_ok=True)

    # Carregar os dados e os rótulos reais separados
    data_test = csv_reader("Titanic_data_set/test.csv")
    labels_test = pd.read_csv('Titanic_data_set/rf_mod_Solution.csv')

    columns_to_remove = ["PassengerId", "Ticket", "Cabin", "Embarked", "Name", "Fare"]
    essential_columns = ["Pclass", "Sex", "Age", "SibSp", "Parch"]
    one_hot_encoding_columns = ["Sex", "Pclass"]
    columns_to_normalize = ["Age", "SibSp", "Parch"]

    # Copiar índice antes do tratamento
    original_index = data_test.index.copy()

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

    # Clustering
    kmeans_model = train_model()
    cluster_labels_test = kmeans_model.predict(X_test)


    plot_silhouette(X_test, cluster_labels_test, "Analysis/KMeans/silhouette_kmeans_test.png",
                    title="Gráfico de Silhueta - Teste (K-Means)")

    plot_tsne(X_test, cluster_labels_test, "Analysis/KMeans/tsne_kmeans_test.png",
              title="Clusters no conjunto de TESTE (K-Means treinado no treino)")

    df_test = pd.DataFrame(X_test)
    df_test['Survived'] = y_test
    df_test['Cluster'] = cluster_labels_test

    print("\nTabela de Contingência entre Clusters e Classe Real:")
    contingency = plot_contingency_matrix(df_test, "Analysis/KMeans/contingency_matrix.png")
    print(contingency)

    print("\nPerfil médio de cada cluster:")
    cluster_profile = save_cluster_profile(df_test,
                                           "Analysis/KMeans/cluster_profiles.csv",
                                           "Analysis/KMeans/cluster_profiles_table.png")
    print(cluster_profile)



def train_model():
    data_train = csv_reader("Titanic_data_set/train.csv")

    columns_to_remove = ["PassengerId", "Ticket", "Cabin", "Embarked", "Name", "Fare"]
    essential_columns = ["Survived", "Pclass", "Sex", "Age", "SibSp", "Parch"]
    one_hot_encoding_columns = ["Sex", "Pclass"]
    columns_to_normalize = ["Age", "SibSp", "Parch"]

    data_train = treat_data(
        data=data_train,
        columns_to_remove=columns_to_remove,
        essential_columns=essential_columns,
        one_hot_columns=one_hot_encoding_columns,
        columns_to_scale=columns_to_normalize
    )

    X_train = data_train.drop('Survived', axis=1)
    y_train = data_train['Survived']

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    kmeans = KMeans(n_clusters=12, random_state=42)
    cluster_labels = kmeans.fit_predict(X_resampled)

    print("Rótulos dos clusters atribuídos pelo K-Means:")
    print(cluster_labels)

    return kmeans
