import os
import pickle
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, roc_auc_score, auc

def generate_precision_recall_auc_graphs(y_true, y_predict, version, show_threshold=False):
    """
    Generates and saves Precision-Recall and ROC curve plots.
    """
    output_dir = os.path.join('outputs', 'images')
    os.makedirs(output_dir, exist_ok=True)

    # Precision-Recall Curve
    precision, recall, thresholds_pr = precision_recall_curve(y_true, y_predict)
    auc_pr = auc(recall, precision)
    plt.figure()
    plt.plot(recall, precision, marker='.', label=f'AUC = {auc_pr:.2f}')
    if show_threshold:
        # Simplified threshold display
        for i in range(0, len(thresholds_pr), 5):
             plt.text(recall[i], precision[i], f'{thresholds_pr[i]:.2f}')
    plt.title(f'Precision-Recall Curve {version}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f'precision_recall.png'))
    plt.close()

    # ROC Curve
    fpr, tpr, thresholds_roc = roc_curve(y_true, y_predict)
    auc_roc = roc_auc_score(y_true, y_predict)
    plt.figure()
    plt.plot(fpr, tpr, marker='.', label=f'AUC = {auc_roc:.2f}')
    if show_threshold:
        # Simplified threshold display
        for i in range(0, len(thresholds_roc), 5):
            plt.text(fpr[i], tpr[i], f'{thresholds_roc[i]:.2f}')
    plt.title(f'ROC Curve {version}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f'roc_curve.png'))
    plt.close()

    return recall, precision, thresholds_pr, fpr, tpr, thresholds_roc

if __name__ == "__main__":
    # Example usage for testing the function directly
    y_true_example = [1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0]
    y_predict_example = np.linspace(1, 0, len(y_true_example))
    generate_precision_recall_auc_graphs(y_true_example, y_predict_example, version='test_v1')
    print("Generated example plots in outputs/images/")
