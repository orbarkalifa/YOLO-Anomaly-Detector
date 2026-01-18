import json

import pytest

from src.config import get_config
from src.eval import evaluate


# Fixture for a temporary run directory
@pytest.fixture
def temp_run_dir(tmp_path):
    run_id = "test_run_eval"
    config = get_config(run_id=run_id)
    # Ensure necessary directories exist
    config.run_dir.mkdir(parents=True, exist_ok=True)
    config.data_dir.mkdir(parents=True, exist_ok=True)

    # Create dummy tracked_objects.csv
    tracked_objects_content = """time_date,score,tag
0,0.1,
1,0.1,
2,0.1,
3,0.1,
4,0.1,
5,0.2,
6,0.2,
7,0.2,
8,0.2,
9,0.3,
10,0.8,spatial_anomaly
11,0.9,spatial_anomaly
12,0.85,spatial_anomaly
13,0.92,spatial_anomaly
14,0.88,spatial_anomaly
15,0.79,spatial_anomaly
16,0.91,spatial_anomaly
17,0.83,spatial_anomaly
18,0.77,spatial_anomaly
19,0.89,spatial_anomaly
20,0.78,spatial_anomaly
21,0.2,
22,0.15,
23,0.1,
24,0.1,
25,0.1,
26,0.2,
27,0.2,
28,0.2,
29,0.3,
30,0.9,speed_anomaly
31,0.95,speed_anomaly
32,0.89,speed_anomaly
33,0.93,speed_anomaly
34,0.87,speed_anomaly
35,0.91,speed_anomaly
36,0.82,speed_anomaly
37,0.75,speed_anomaly
38,0.94,speed_anomaly
39,0.81,speed_anomaly
40,0.76,speed_anomaly
41,0.2,
42,0.15,
43,0.1,
44,0.1,
45,0.1,
46,0.1,
47,0.1,
48,0.1,
49,0.1,
50,0.1,
"""
    (config.data_dir / "tracked_objects.csv").write_text(tracked_objects_content)

    yield config.run_dir, config.data_dir / "tracked_objects.csv"

    # Teardown (pytest tmp_path handles directory cleanup)


# Fixture for a temporary ground truth JSON file
@pytest.fixture
def temp_gt_file(tmp_path):
    gt_content = [
        {"start_frame": 10, "end_frame": 20},
        {"start_frame": 30, "end_frame": 40},
    ]
    gt_path = tmp_path / "ground_truth.json"
    with open(gt_path, "w") as f:
        json.dump(gt_content, f)
    return gt_path


def test_evaluation_produces_plots(temp_run_dir, temp_gt_file):
    """
    Tests that the evaluate function successfully produces PR and ROC plots.
    """
    output_dir, pred_path = temp_run_dir
    gt_path = temp_gt_file

    evaluate(gt_path=str(gt_path), pred_path=str(pred_path), output_dir=str(output_dir))

    # Assert that the plot files are created
    pr_curve_path = output_dir / "precision_recall_curve.png"
    roc_curve_path = output_dir / "roc_curve.png"

    assert pr_curve_path.exists()
    assert roc_curve_path.exists()

    # Optionally, check that the files are not empty (e.g., larger than 1KB)
    assert pr_curve_path.stat().st_size > 1024
    assert roc_curve_path.stat().st_size > 1024
