import os
import uuid

import gdown
import pytest

from src import params_links
from src.config import get_config
from src.pipeline.learn import learn_routine as run_main_routine_loop


@pytest.mark.order(12)
@pytest.mark.parametrize(
    "input_video_name, input_video_link",  # Removed gt_heatmap params
    [
        (
            "routine_frame.mp4",
            params_links.routine_frame_link,
        )
    ],
)
def test_left_side_routine_map_as_pkl_when_input_moving_left_side(
    input_video_name,
    input_video_link,
):
    # Generate a unique run_id for this test
    test_run_id = str(uuid.uuid4())

    # Correctly construct paths using get_config
    config = get_config(run_id=test_run_id)
    video_path = os.path.join("data", "raw", input_video_name)
    output_pkl_path = config.routine_map_path

    # Ensure directories exist
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    # The learn_routine will create the run-specific output directories

    # Download necessary files
    if not os.path.exists(video_path):
        try:
            gdown.download(input_video_link, video_path, quiet=False)
        except Exception as e:
            pytest.fail(f"gdown failed to download {input_video_link}. Error: {e}")

    # Run the main loop
    run_main_routine_loop(video_path, display=False, run_id=test_run_id)

    # Compare the generated pkl with the ground truth
    assert os.path.exists(output_pkl_path), (
        f"Output PKL file '{output_pkl_path}' was not created."
    )

    # Assert that the file is not empty (e.g., larger than a minimal size)
    assert os.path.getsize(output_pkl_path) > 100, (
        "Generated PKL file is too small or empty."
    )
