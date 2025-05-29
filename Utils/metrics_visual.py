import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, accuracy_score, roc_curve, auc
import os
import pandas as pd


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


def plot_roc_curve(y_true, y_proba, title='Curve ROC', save_path=None, show=True):
    """
    Plota a Curva ROC com AUC para um modelo de classificação binária.

    Parâmetros:
    - y_true: array-like com os rótulos reais (0 ou 1)
    - y_proba: array-like com as probabilidades preditas da classe positiva (ex: .predict_proba()[:, 1])
    - title: título do gráfico (opcional)
    - save_path: caminho para salvar a imagem (opcional)
    - show: se True, exibe o gráfico ao final (opcional)
    """

    # Calcula FPR, TPR e AUC
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    # Plotagem
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Aleatório (AUC = 0.5)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()
