import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, accuracy_score, roc_curve, auc
import matplotlib.cm as cm
import numpy as np
import os
import pandas as pd
import seaborn as sns

def save_confusion_matrix(y_true, y_pred, class_labels, save_path, show=False, title="Confusion Matrix"):
    """
    Generates and saves the confusion matrix as an image file.

    Parameters:
    - y_true: Ground truth labels
    - y_pred: Predicted labels
    - class_labels: List of class display names (e.g., ["Died", "Survived"])
    - save_path: Full path to save the image (e.g., 'results/conf_matrix_rf.png')
    - show: If True, also display the plot
    - title: Plot title
    """

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    disp = ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred,
        display_labels=class_labels,
        cmap="Blues"
    )
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)

    if show:
        plt.show()

    plt.close()
    print(f"[INFO] Confusion matrix saved to: {save_path}")

def save_classification_report_image(y_true, y_pred, class_labels, save_path, title="Classification Report"):
    """
    Generates a styled image with only precision, recall and F1-score per class,
    and shows global accuracy separately. Saves the result as a PNG image.

    Parameters:
    - y_true: Ground truth labels
    - y_pred: Predicted labels
    - class_labels: List of class names
    - save_path: Path to save the image
    - title: Title to display on the plot
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Metrics
    report_dict = classification_report(
        y_true, y_pred,
        target_names=class_labels,
        output_dict=True
    )
    accuracy = accuracy_score(y_true, y_pred)
    df = pd.DataFrame(report_dict).transpose().round(3)

    # Keep only precision, recall, f1-score, remove accuracy/support/macro/weighted
    df_filtered = df.loc[class_labels, ['precision', 'recall', 'f1-score']]

    # Plotting
    fig, ax = plt.subplots(figsize=(8, len(df_filtered) * 0.6 + 2))
    ax.axis('off')

    # Table style
    table = ax.table(
        cellText=df_filtered.values,
        rowLabels=df_filtered.index,
        colLabels=df_filtered.columns,
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.1, 1.6)

    # Title and accuracy
    plt.title(f"{title}\nAccuracy: {accuracy:.2%}", fontsize=14, weight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"[INFO] Pretty classification report image saved to: {save_path}")

def plot_roc_curve(y_true, y_proba, title='ROC Curve', save_path=None, show=True):
    """
    Plots the ROC Curve with AUC for a binary classification model.

    Parameters:
    - y_true: array-like, true binary labels (0 or 1)
    - y_proba: array-like, predicted probabilities for the positive class (e.g., model.predict_proba()[:, 1])
    - title: str, plot title (optional)
    - save_path: str, path to save the image (optional)
    - show: bool, if True, displays the plot (optional)
    """

    # Compute False Positive Rate (FPR), True Positive Rate (TPR), and thresholds
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    # Plot ROC Curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random (AUC = 0.5)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True)

    # Save or display the plot
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()

def plot_survival_distribution(train_labels, title='Survival Distribution (Train Set)', save_path=None, show=True):
    """
    Plots the distribution of the 'Survived' variable for the training set.

    Parameters:
    - train_labels: Series or array-like of training set labels (0 or 1)
    - title: plot title (optional)
    - save_path: path to save the plot (optional)
    - show: if True, displays the plot
    """
    train_df = pd.DataFrame({'Survived': train_labels})

    # Plot
    plt.figure(figsize=(6, 5))
    sns.countplot(data=train_df, x='Survived', palette='Set2')
    plt.title(title)
    plt.xlabel('Survived')
    plt.ylabel('Count')
    plt.xticks(ticks=[0, 1], labels=['Died', 'Survived'])
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()

def plot_silhouette(X, labels, save_path, title="Gráfico de Silhueta"):
    from sklearn.metrics import silhouette_samples, silhouette_score

    score = silhouette_score(X, labels)
    sample_silhouette_values = silhouette_samples(X, labels)
    n_clusters = len(set(labels))

    fig, ax = plt.subplots(figsize=(10, 6))
    y_lower = 10
    for i in range(n_clusters):
        ith_cluster_silhouette_values = sample_silhouette_values[labels == i]
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        color = cm.nipy_spectral(float(i) / n_clusters)
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values,
                         facecolor=color, edgecolor=color, alpha=0.7)
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, f"Cluster {i}")
        y_lower = y_upper + 10

    ax.set_title(title)
    ax.set_xlabel("Coeficiente de Silhueta")
    ax.set_ylabel("Clusters")
    ax.axvline(x=score, color="red", linestyle="--", label="Média")
    ax.set_yticks([])
    ax.set_xlim([-0.2, 1])
    ax.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_tsne(X, labels, save_path, title="Visualização TSNE"):
    from sklearn.manifold import TSNE

    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X)

    plt.figure(figsize=(8, 6))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, cmap='viridis', alpha=0.7)
    plt.title(title)
    plt.xlabel("TSNE-1")
    plt.ylabel("TSNE-2")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_contingency_matrix(df, save_path):
    contingency = pd.crosstab(df['Cluster'], df['Survived'])
    plt.figure(figsize=(6, 4))
    sns.heatmap(contingency, annot=True, fmt='d', cmap='Blues')
    plt.title("Matriz de Contingência: Cluster x Survived")
    plt.xlabel("Classe Real (Survived)")
    plt.ylabel("Cluster")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return contingency


def save_cluster_profile(df, save_csv_path, save_table_path):
    cluster_profile = df.groupby('Cluster').mean().round(2)
    cluster_profile.to_csv(save_csv_path)

    fig, ax = plt.subplots(figsize=(10, 2 + 0.5 * len(cluster_profile)))
    ax.axis('off')
    tbl = ax.table(cellText=cluster_profile.values,
                   colLabels=cluster_profile.columns,
                   rowLabels=cluster_profile.index,
                   cellLoc='center',
                   loc='center')
    tbl.scale(1, 1.5)
    plt.title("Perfil Médio por Cluster")
    plt.tight_layout()
    plt.savefig(save_table_path)
    plt.close()
    return cluster_profile
