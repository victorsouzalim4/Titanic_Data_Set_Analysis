import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, accuracy_score
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