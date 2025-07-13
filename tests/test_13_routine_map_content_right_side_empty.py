import os
import gdown
import pytest
import pickle
import numpy as np
from src import params_links
from src.main import run_main_routine_loop

@pytest.mark.order(13)
@pytest.mark.parametrize("input_video_name, input_video_link, heatmap_gt_file_name, heatmap_gt_file_link, max_error",
                         [('routine_frame.mp4',
                           params_links.routine_frame_link,
                           'routine_map_left_road_gt.pkl',
                           params_links.routine_map_left_road_gt_link,
                           1280 * 720 * 100)
                          ])
def test_routine_map_as_pkl_right_side_empty_when_input_is_motionless(input_video_name, input_video_link, heatmap_gt_file_name, heatmap_gt_file_link, max_error):
    # Correctly construct paths
    video_path = os.path.join('data', 'raw', input_video_name)
    gt_pkl_path = os.path.join('data', 'processed', heatmap_gt_file_name)
    output_pkl_path = os.path.join('outputs', 'data', 'routine_map.pkl')

    # Ensure directories exist
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    os.makedirs(os.path.dirname(gt_pkl_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_pkl_path), exist_ok=True)

    # Download necessary files
    if not os.path.exists(video_path):
        try:
            gdown.download(input_video_link, video_path, quiet=False)
        except Exception as e:
            pytest.fail(f"gdown failed to download {input_video_link}. Error: {e}")

    if not os.path.exists(gt_pkl_path):
        try:
            gdown.download(heatmap_gt_file_link, gt_pkl_path, quiet=False)
        except Exception as e:
            pytest.fail(f"gdown failed to download {heatmap_gt_file_link}. Error: {e}")

    # Run the main loop
    run_main_routine_loop(video_path)

    # Compare the generated pkl with the ground truth
    assert os.path.exists(output_pkl_path), f"Output PKL file '{output_pkl_path}' was not created."
    
    with open(gt_pkl_path, 'rb') as f_gt, open(output_pkl_path, 'rb') as f_out:
        map_gt = pickle.load(f_gt)
        map_out = pickle.load(f_out)
        
    # Check that the right side of the map is empty
    height, width = map_out.shape
    right_side = map_out[:, width // 2:]
    assert np.sum(right_side) < max_error, "The right side of the heatmap should be empty."