import os
import gdown
import pytest
from src import params_links
from src.main import run_main_anomaly_loop

@pytest.mark.order(2)
@pytest.mark.parametrize("input_video_name, input_video_link, csv_file_name, size",
                         [('test1.mp4',
                           params_links.test1_link,
                           'tracked_objects.csv',
                           100)
                          ])
def test_check_created_csv_file(input_video_name, input_video_link, csv_file_name, size):
    # Correctly construct paths relative to the project root
    video_path = os.path.join('data', 'raw', input_video_name)
    csv_path = os.path.join('outputs', 'data', csv_file_name)

    # Ensure the directories exist
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    # Download the video if it doesn't exist
    if not os.path.exists(video_path):
        try:
            gdown.download(input_video_link, video_path, quiet=False)
        except Exception as e:
            pytest.fail(f"gdown failed to download {input_video_link}. Error: {e}")

    # Run the main loop to generate the CSV
    run_main_anomaly_loop(video_path)

    # Assert that the CSV file was created
    assert os.path.exists(csv_path), f"CSV file '{csv_path}' was not created."
    assert os.path.getsize(csv_path) > size, f"CSV file '{csv_path}' is smaller than {size} bytes."