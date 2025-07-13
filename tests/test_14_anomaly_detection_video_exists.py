import os
import gdown
import pytest
from src import params_links
from src.main import run_main_anomaly_loop

@pytest.mark.order(14)
@pytest.mark.parametrize("input_video_name, input_video_link, gt_video_file_name, gt_video_file_name_link, max_error",
                         [('test1.mp4',
                           params_links.test1_link,
                           'anomaly_detection_gt.mp4',
                           params_links.anomaly_detection_gt_link,
                           1280 * 720 * 5)
                          ])
def test_check_created_motion_video(input_video_name, input_video_link, gt_video_file_name, gt_video_file_name_link,  max_error):
    # Correctly construct paths
    video_path = os.path.join('data', 'raw', input_video_name)
    output_video_path = os.path.join('outputs', 'videos', 'anomaly_detection.mp4')

    # Ensure directories exist
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)

    # Download the video if it doesn't exist
    if not os.path.exists(video_path):
        try:
            gdown.download(input_video_link, video_path, quiet=False)
        except Exception as e:
            pytest.fail(f"gdown failed to download {input_video_link}. Error: {e}")

    # Run the main loop to generate the video
    run_main_anomaly_loop(video_path)

    # Assert that the video file was created
    assert os.path.exists(output_video_path), f"Output video '{output_video_path}' was not created."
    assert os.path.getsize(output_video_path) > 0, f"Output video '{output_video_path}' is empty."
