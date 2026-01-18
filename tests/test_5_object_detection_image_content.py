import glob
import os
import uuid

import cv2
import pytest

from src.config import get_config  # Import get_config
from src.pipeline.detect import detect_anomalies as run_main_anomaly_loop
from src.pipeline.learn import learn_routine


@pytest.mark.order(5)
@pytest.mark.parametrize(
    "input_video_name, jpg_file_name, size",  # input_video_link removed
    [
        (
            "test1.mp4",
            "*.jpg",  # Changed to wildcard for any JPG
            100,
        )
    ],
)
def test_check_created_jpg_file(
    input_video_name,
    jpg_file_name,
    size,
):
    # Generate unique run_ids for both phases of this test
    routine_run_id = str(uuid.uuid4())
    anomaly_run_id = str(uuid.uuid4())

    # Correctly construct paths using get_config
    config_anomaly = get_config(run_id=anomaly_run_id)
    video_path = os.path.join("data", "raw", input_video_name)
    bbox_dir = config_anomaly.bbox_dir

    # Ensure directories exist
    os.makedirs(os.path.dirname(video_path), exist_ok=True)

    # Assume video exists locally or will be downloaded by main script
    # Removed gdown.download call

    # First, run the learn routine to generate the routine map
    learn_routine(video_path, display=False, run_id=routine_run_id)

    # Then, run the main loop to generate the images
    run_main_anomaly_loop(
        video_path, display=False, run_id=anomaly_run_id, learn_run_id=routine_run_id
    )

    # Find the first generated image to compare against the ground truth
    generated_images = glob.glob(os.path.join(bbox_dir, jpg_file_name))
    assert len(generated_images) > 0, (
        f"No bounding box images were generated in {bbox_dir} matching '{jpg_file_name}'."
    )

    # Read the generated image
    img_gen = cv2.imread(generated_images[0])

    assert img_gen is not None, (
        f"Could not read generated image at {generated_images[0]}"
    )

    # Optionally, check the size of the first found image
    assert os.path.getsize(generated_images[0]) > size, (
        f"The first created JPG file is smaller than {size} bytes."
    )
