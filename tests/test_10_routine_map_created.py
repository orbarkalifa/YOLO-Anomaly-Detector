import os
import gdown
import pytest
import pickle
from src import params_links
from src.main import run_main_routine_loop

@pytest.mark.order(10)
@pytest.mark.parametrize("input_video_name, input_video_link, heatmap_gt_file_name, heatmap_gt_file_link, max_error",
                         [('test1.mp4',
                           params_links.test1_link,
                           'routine_map_left_road_gt.pkl',
                           params_links.routine_map_left_road_gt_link,
                           1280 * 720 / 2)
                          ])
def test_save_heat_map_as_pkl(input_video_name, input_video_link, heatmap_gt_file_name, heatmap_gt_file_link, max_error):
    # Correctly construct paths
    video_path = os.path.join('data', 'raw', input_video_name)
    output_pkl_path = os.path.join('outputs', 'data', 'routine_map.pkl')

    # Ensure directories exist
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_pkl_path), exist_ok=True)

    # Download the video if it doesn't exist
    if not os.path.exists(video_path):
        try:
            gdown.download(input_video_link, video_path, quiet=False)
        except Exception as e:
            pytest.fail(f"gdown failed to download {input_video_link}. Error: {e}")

    # Run the main loop to generate the pkl
    run_main_routine_loop(video_path)

    # Assert that the pkl file was created
    assert os.path.exists(output_pkl_path), f"Output PKL file '{output_pkl_path}' was not created."
    assert os.path.getsize(output_pkl_path) > 0, f"Output PKL file '{output_pkl_path}' is empty."
