import os
import gdown
import pytest
import cv2
import numpy as np
from src import params_links
from src.main import run_main_anomaly_loop

@pytest.mark.order(15)
@pytest.mark.parametrize("input_video_name, input_video_link, gt_video_file_name, gt_video_file_name_link, max_error",
                         [('test1.mp4',
                           params_links.test1_link,
                           'anomaly_detection_gt.mp4',
                           params_links.anomaly_detection_gt_link,
                           1280 * 720 * 5)
                          ])
def test_check_created_anomaly_detection_motion_video(input_video_name, input_video_link, gt_video_file_name, gt_video_file_name_link, max_error):
    # Correctly construct paths
    video_path = os.path.join('data', 'raw', input_video_name)
    gt_video_path = os.path.join('data', 'raw', gt_video_file_name)
    output_video_path = os.path.join('outputs', 'videos', 'anomaly_detection.mp4')

    # Ensure directories exist
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    os.makedirs(os.path.dirname(gt_video_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)

    # Download necessary files
    if not os.path.exists(video_path):
        try:
            gdown.download(input_video_link, video_path, quiet=False)
        except Exception as e:
            pytest.fail(f"gdown failed to download {input_video_link}. Error: {e}")

    if not os.path.exists(gt_video_path):
        try:
            gdown.download(gt_video_file_name_link, gt_video_path, quiet=False)
        except Exception as e:
            pytest.fail(f"gdown failed to download {gt_video_file_name_link}. Error: {e}")

    # Run the main loop
    run_main_anomaly_loop(video_path)

    # Compare the generated video with the ground truth
    assert os.path.exists(output_video_path), f"Output video '{output_video_path}' was not created."
    
    cap = cv2.VideoCapture(output_video_path)
    cap_gt = cv2.VideoCapture(gt_video_path)

    assert cap.isOpened(), f"Could not open generated video: {output_video_path}"
    assert cap_gt.isOpened(), f"Could not open ground truth video: {gt_video_path}"

    # Check that the output video is not empty
    assert cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0, "The generated video has no frames."
    
    cap.release()
    cap_gt.release()