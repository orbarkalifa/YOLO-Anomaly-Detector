import os
import uuid

import gdown
import pytest

from src import params_links
from src.config import get_config  # Import get_config
from src.pipeline.detect import detect_anomalies as run_main_anomaly_loop
from src.pipeline.learn import learn_routine


@pytest.mark.order(2)
@pytest.mark.parametrize(
    "input_video_name, input_video_link, csv_file_name, size",
    [("test1.mp4", params_links.test1_link, "tracked_objects.csv", 100)],
)
def test_check_created_csv_file(
    input_video_name, input_video_link, csv_file_name, size
):
    # Generate unique run_ids for both phases of this test
    routine_run_id = str(uuid.uuid4())
    anomaly_run_id = str(uuid.uuid4())

    # Correctly construct paths using get_config
    config_anomaly = get_config(run_id=anomaly_run_id)
    video_path = os.path.join("data", "raw", input_video_name)
    csv_path = config_anomaly.tracked_objects_csv

    # Ensure the video input directory exists
    os.makedirs(os.path.dirname(video_path), exist_ok=True)

    # Download the video if it doesn't exist
    if not os.path.exists(video_path):
        try:
            gdown.download(input_video_link, video_path, quiet=False)
        except Exception as e:
            pytest.fail(f"gdown failed to download {input_video_link}. Error: {e}")

    # First, run the learn routine to generate the routine map
    learn_routine(video_path, display=False, run_id=routine_run_id)

    # Then, run the anomaly detection loop to generate the CSV
    run_main_anomaly_loop(
        video_path, display=False, run_id=anomaly_run_id, learn_run_id=routine_run_id
    )

    # Assert that the CSV file was created
    assert os.path.exists(csv_path), f"CSV file '{csv_path}' was not created."
    assert os.path.getsize(csv_path) > size, (
        f"CSV file '{csv_path}' is smaller than {size} bytes."
    )
