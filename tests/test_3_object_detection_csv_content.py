import os
import gdown
import pytest
import pandas as pd
from src import params_links
from src.main import run_main_anomaly_loop

@pytest.mark.order(3)
@pytest.mark.parametrize("input_video_name, input_video_link, gt_csv_file_name, gt_csv_file_name_link, csv_file_name, max_error",
                         [('test1.mp4',
                           params_links.test1_link,
                           'tracked_objects_gt.csv',
                           params_links.csv_anomaly_link,
                           'tracked_objects.csv',
                           100)
                          ])
def test_check_created_csv_bbox(input_video_name, input_video_link, gt_csv_file_name, gt_csv_file_name_link, csv_file_name, max_error):
    # Correctly construct paths
    video_path = os.path.join('data', 'raw', input_video_name)
    gt_csv_path = os.path.join('data', 'processed', gt_csv_file_name)
    output_csv_path = os.path.join('outputs', 'data', csv_file_name)

    # Ensure directories exist
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    os.makedirs(os.path.dirname(gt_csv_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

    # Download necessary files
    if not os.path.exists(video_path):
        try:
            gdown.download(input_video_link, video_path, quiet=False)
        except Exception as e:
            pytest.fail(f"gdown failed to download {input_video_link}. Error: {e}")

    if not os.path.exists(gt_csv_path):
        try:
            gdown.download(gt_csv_file_name_link, gt_csv_path, quiet=False)
        except Exception as e:
            pytest.fail(f"gdown failed to download {gt_csv_file_name_link}. Error: {e}")

    # Run the main loop
    run_main_anomaly_loop(video_path)

    # Compare the generated CSV with the ground truth
    assert os.path.exists(output_csv_path), f"Output CSV file '{output_csv_path}' was not created."
    
    df_gt = pd.read_csv(gt_csv_path)
    df_output = pd.read_csv(output_csv_path)
    
    assert len(df_output) > 0, "Generated CSV file is empty."
    # Add more specific content comparison logic if needed
