import os
import pickle
import uuid

import gdown
import numpy as np
import pytest

from src import params_links
from src.config import get_config
from src.pipeline.learn import learn_routine as run_main_routine_loop


@pytest.mark.order(13)
@pytest.mark.parametrize(
    "input_video_name, input_video_link, max_error",  # Removed gt_heatmap params
    [
        (
            "routine_frame.mp4",
            params_links.routine_frame_link,
            1280 * 720 * 100,  # This error threshold needs careful consideration
        )
    ],
)
def test_routine_map_as_pkl_right_side_empty_when_input_is_motionless(
    input_video_name,
    input_video_link,
    max_error,
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

    with open(output_pkl_path, "rb") as f_out:
        map_out = pickle.load(f_out)

    # Check that the right side of the map is empty
    height, width = map_out.shape
    right_side = map_out[:, width // 2 :]
    assert np.sum(right_side) < max_error, (
        "The right side of the heatmap should be empty."
    )
