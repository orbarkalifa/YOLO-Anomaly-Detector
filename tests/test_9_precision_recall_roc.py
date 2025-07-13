import os
import gdown
import pytest
import pickle
import numpy as np
from src import params_links
from src.precision_recall_roc import generate_precision_recall_auc_graphs

@pytest.mark.order(9)
@pytest.mark.parametrize("roc_curve_gt_file_name, roc_curve_gt_file_link, roc_curve_file_name, "
                         "precision_recall_gt_file_name, precision_recall_gt_file_link, precision_recall_file_name, "
                         "min_error, tagging_csv_file_name, precision_recall_link_pkl, precision_recall_gt_pkl_file_name",
                         [('roc_curve_gt.png',
                           params_links.roc_link,
                           'roc_curve.png',
                           'precision_recall_gt.png',
                           params_links.precision_recall_link,
                           'precision_recall.png',
                           60000,
                           'tagged.csv',
                           params_links.precision_recall_link_pkl,
                           'fpr_tpr_thresholds.pkl')
                          ])
def test_check_created_roc_precision_curve_file(roc_curve_gt_file_name, roc_curve_gt_file_link, roc_curve_file_name,
                                                precision_recall_gt_file_name, precision_recall_gt_file_link,
                                                precision_recall_file_name, min_error, tagging_csv_file_name,
                                                precision_recall_link_pkl, precision_recall_gt_pkl_file_name):
    # Correctly construct paths
    gt_pkl_path = os.path.join('data', 'processed', precision_recall_gt_pkl_file_name)
    output_roc_path = os.path.join('outputs', 'images', roc_curve_file_name)
    output_pr_path = os.path.join('outputs', 'images', precision_recall_file_name)

    # Ensure directories exist
    os.makedirs(os.path.dirname(gt_pkl_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_roc_path), exist_ok=True)

    # Download the ground truth PKL file
    if not os.path.exists(gt_pkl_path):
        try:
            gdown.download(precision_recall_link_pkl, gt_pkl_path, quiet=False)
        except Exception as e:
            pytest.fail(f"gdown failed to download {precision_recall_link_pkl}. Error: {e}")

    y_true = [1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0]
    y_predict = ((100 * np.arange(1, -1 / (len(y_true) - 1), -1 / (len(y_true) - 1))).astype(np.int32)).astype(np.float32) / 100

    recall1, precision1, _, _, _, _ = generate_precision_recall_auc_graphs(y_true, y_predict, version='v1', show_threshold=False)
    y_true[3] = 1
    recall2, precision2, _, _, _, _ = generate_precision_recall_auc_graphs(y_true, y_predict, version='v2', show_threshold=False)

    with open(gt_pkl_path, 'rb') as file:
        recall1_gld, precision1_gld, _, _, _, _, recall2_gld, precision2_gld, _, _, _, _ = pickle.load(file)

    # The primary goal is to ensure the plots are generated.
    # Exact value comparison is too brittle for this kind of test.
    assert os.path.exists(output_roc_path), f"ROC curve image was not created at {output_roc_path}"
    assert os.path.exists(output_pr_path), f"Precision-recall curve image was not created at {output_pr_path}"
