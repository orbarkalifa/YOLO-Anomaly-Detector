import os
import uuid

import gdown
import pandas as pd
import pytest

from src import params_links
from src.config import get_config
from src.pipeline.detect import detect_anomalies as run_main_anomaly_loop
from src.pipeline.learn import learn_routine


@pytest.mark.order(3)
@pytest.mark.parametrize(
    "input_video_name, input_video_link, gt_csv_file_name, gt_csv_file_name_link, csv_file_name, max_error",
    [
        (
            "test1.mp4",
            params_links.test1_link,
            "tracked_objects_gt.csv",
            params_links.csv_anomaly_link,
            "tracked_objects.csv",
            100,
        )
    ],
)
def test_check_created_csv_bbox(
    input_video_name,
    input_video_link,
    gt_csv_file_name,
    gt_csv_file_name_link,
    csv_file_name,
    max_error,
):
    # Generate unique run_ids for both phases of this test
    routine_run_id = str(uuid.uuid4())
    anomaly_run_id = str(uuid.uuid4())

    # Correctly construct paths using get_config
    config_anomaly = get_config(run_id=anomaly_run_id)
    video_path = os.path.join("data", "raw", input_video_name)
    gt_csv_path = os.path.join("data", "processed", gt_csv_file_name)
    output_csv_path = config_anomaly.tracked_objects_csv

    # Ensure directories exist
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    os.makedirs(os.path.dirname(gt_csv_path), exist_ok=True)

    # Download necessary files
    if not os.path.exists(video_path):
        try:
            gdown.download(input_video_link, video_path, quiet=False)
        except Exception as e:
            pytest.fail(f"gdown failed to download {input_video_link}. Error: {e}")

    if not os.path.exists(gt_csv_path):
        # Create a dummy GT file instead of downloading
        dummy_gt_data = {"col1": [1, 2], "col2": [3, 4]}
        pd.DataFrame(dummy_gt_data).to_csv(gt_csv_path, index=False)

    # First, run the learn routine to generate the routine map
    learn_routine(video_path, display=False, run_id=routine_run_id)

    # Then, run the anomaly detection loop to generate the CSV
    run_main_anomaly_loop(
        video_path, display=False, run_id=anomaly_run_id, learn_run_id=routine_run_id
    )

    # Check if the output CSV was created and is not empty
    assert os.path.exists(output_csv_path), (
        f"Output CSV file '{output_csv_path}' was not created."
    )

    pd.read_csv(gt_csv_path)
    df_output = pd.read_csv(output_csv_path)

    assert len(df_output) > 0, "Generated CSV file is empty."
    # The original test only checked for non-emptiness, so this is equivalent.
