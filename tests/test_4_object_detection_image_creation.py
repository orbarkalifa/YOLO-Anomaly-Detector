import glob
import os
import uuid

import gdown
import pytest

from src import params_links
from src.config import get_config
from src.pipeline.detect import detect_anomalies as run_main_anomaly_loop
from src.pipeline.learn import learn_routine


@pytest.mark.order(4)
@pytest.mark.parametrize(
    "input_video_name, input_video_link, jpg_file_name, size",
    [("test1.mp4", params_links.test1_link, "*.jpg", 100)],
)
def test_check_created_jpg_file(
    input_video_name, input_video_link, jpg_file_name, size
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

    # Download the video if it doesn't exist
    if not os.path.exists(video_path):
        try:
            gdown.download(input_video_link, video_path, quiet=False)
        except Exception as e:
            pytest.fail(f"gdown failed to download {input_video_link}. Error: {e}")

    # First, run the learn routine to generate the routine map
    learn_routine(video_path, display=False, run_id=routine_run_id)

    # Then, run the main loop to generate the images
    run_main_anomaly_loop(
        video_path, display=False, run_id=anomaly_run_id, learn_run_id=routine_run_id
    )

    # Check if any JPG files were created in the bbox directory
    jpg_files = glob.glob(os.path.join(bbox_dir, jpg_file_name))
    assert len(jpg_files) > 0, (
        f"No JPG files matching '{jpg_file_name}' were created in {bbox_dir}."
    )

    # Optionally, check the size of the first found image
    assert os.path.getsize(jpg_files[0]) > size, (
        f"The first created JPG file is smaller than {size} bytes."
    )
