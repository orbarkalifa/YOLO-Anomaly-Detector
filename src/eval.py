import json
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve, roc_curve

logger = logging.getLogger(__name__)


def evaluate(gt_path: str, pred_path: str, output_dir: str):
    """
    Evaluates the anomaly detection results against the ground truth.

    Args:
        gt_path: Path to the ground truth labels file (labels.json).
        pred_path: Path to the tracked_objects.csv file.
        output_dir: Directory to save the evaluation plots.
    """
    with open(gt_path, "r") as f:
        ground_truth = json.load(f)

    predictions = pd.read_csv(pred_path)

    # Create a boolean array for ground truth frames
    num_frames = predictions[
        "time_date"
    ].nunique()  # A simplification, assuming time_date maps to frames
    y_true = np.zeros(num_frames)

    for anomaly in ground_truth:
        # This is a simplification. The `tagging.py` does not provide frame numbers.
        # We are assuming the index of the row in the original csv corresponds to the frame number.
        start_frame = anomaly["start_frame"]
        end_frame = anomaly["end_frame"]
        y_true[start_frame : end_frame + 1] = 1

    # Get the max score for each frame
    y_scores = predictions.groupby("time_date")["score"].max().values

    # Ensure y_true and y_scores have the same length
    min_len = min(len(y_true), len(y_scores))
    y_true = y_true[:min_len]
    y_scores = y_scores[:min_len]

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)

    plt.figure()
    plt.plot(recall, precision, label=f"PR curve (area = {pr_auc:.2f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.savefig(f"{output_dir}/precision_recall_curve.png")
    plt.close()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(
        fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.2f})"
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.savefig(f"{output_dir}/roc_curve.png")
    plt.close()

    logger.info("Evaluation complete. Plots saved to:", output_dir)
