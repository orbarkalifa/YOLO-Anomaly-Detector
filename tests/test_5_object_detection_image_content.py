import os
import gdown
import pytest
import cv2
import numpy as np
from src import params_links
from src.main import run_main_anomaly_loop

@pytest.mark.order(5)
@pytest.mark.parametrize("input_video_name, input_video_link, jpg_file_name, gt_jpg_file_name, gt_jpg_file_name_link, min_error",
                         [('test1.mp4',
                           params_links.test1_link,
                           '1.0_*.jpg',
                           '1.0_2024-06-12__17_17_46_476904.jpg',
                           params_links.image_link,
                           800000)
                          ])
def test_check_created_jpg_file(input_video_name, input_video_link, jpg_file_name, gt_jpg_file_name, gt_jpg_file_name_link, min_error):
    # Correctly construct paths
    video_path = os.path.join('data', 'raw', input_video_name)
    gt_jpg_path = os.path.join('data', 'raw', gt_jpg_file_name) # Assuming GT images are raw data
    bbox_dir = os.path.join('outputs', 'images', 'bbox_images')

    # Ensure directories exist
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    os.makedirs(os.path.dirname(gt_jpg_path), exist_ok=True)
    os.makedirs(bbox_dir, exist_ok=True)

    # Download necessary files
    if not os.path.exists(video_path):
        try:
            gdown.download(input_video_link, video_path, quiet=False)
        except Exception as e:
            pytest.fail(f"gdown failed to download {input_video_link}. Error: {e}")

    if not os.path.exists(gt_jpg_path):
        try:
            gdown.download(gt_jpg_file_name_link, gt_jpg_path, quiet=False)
        except Exception as e:
            pytest.fail(f"gdown failed to download {gt_jpg_file_name_link}. Error: {e}")

    # Run the main loop
    run_main_anomaly_loop(video_path)

    # Find the first generated image to compare against the ground truth
    import glob
    generated_images = glob.glob(os.path.join(bbox_dir, '1.0_*.jpg'))
    assert len(generated_images) > 0, "No bounding box images were generated."
    
    # Read images
    img_gt = cv2.imread(gt_jpg_path)
    img_gen = cv2.imread(generated_images[0])

    assert img_gt is not None, f"Could not read ground truth image at {gt_jpg_path}"
    assert img_gen is not None, f"Could not read generated image at {generated_images[0]}"

    # Compare images
    diff = cv2.absdiff(img_gt, img_gen)
    assert np.sum(diff) < min_error, f"Image difference ({np.sum(diff)}) exceeds threshold ({min_error})."