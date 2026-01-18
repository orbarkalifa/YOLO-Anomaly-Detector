import os
import uuid

import cv2
import gdown
import pytest

from src import params_links
from src.config import get_config  # Import get_config
from src.pipeline.detect import detect_anomalies as run_main_anomaly_loop
from src.pipeline.learn import learn_routine


@pytest.mark.order(7)
@pytest.mark.parametrize(
    "input_video_name, input_video_link, anomaly_detection_file_name",
    [
        (
            "test1.mp4",
            params_links.test1_link,
            "anomaly_detection.mp4",
        )
    ],
)
def test_check_created_anomaly_detection_motion_video(
    input_video_name,
    input_video_link,
    anomaly_detection_file_name,
):
    # Generate unique run_ids for both phases of this test

    routine_run_id = str(uuid.uuid4())

    anomaly_run_id = str(uuid.uuid4())

    # Correctly construct paths using get_config

    # For anomaly_detect output

    config_anomaly = get_config(run_id=anomaly_run_id)

    video_path = os.path.join("data", "raw", input_video_name)

    output_video_path = config_anomaly.anomaly_detection_video_path

    # Ensure video input directory exists

    os.makedirs(os.path.dirname(video_path), exist_ok=True)

    # Download necessary files

    if not os.path.exists(video_path):
        try:
            gdown.download(input_video_link, video_path, quiet=False)

        except Exception as e:
            pytest.fail(f"gdown failed to download {input_video_link}. Error: {e}")

    # Run the main loops

    learn_routine(video_path, display=False, run_id=routine_run_id)

    run_main_anomaly_loop(
        video_path, display=False, run_id=anomaly_run_id, learn_run_id=routine_run_id
    )

    # Compare the generated video with the ground truth

    assert os.path.exists(output_video_path), (
        f"Output video '{output_video_path}' was not created."
    )

    cap = cv2.VideoCapture(str(output_video_path))

    assert cap.isOpened(), f"Could not open generated video: {output_video_path}"

    # Check that the output video is not empty

    assert cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0, "The generated video has no frames."

    cap.release()
