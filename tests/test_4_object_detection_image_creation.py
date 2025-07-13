import os
import gdown
import pytest
import glob
from src import params_links
from src.main import run_main_anomaly_loop

@pytest.mark.order(4)
@pytest.mark.parametrize("input_video_name, input_video_link, jpg_file_name, size",
                         [('test1.mp4',
                           params_links.test1_link,
                           '1.0_*.jpg',
                           100)
                          ])
def test_check_created_jpg_file(input_video_name, input_video_link, jpg_file_name, size):
    # Correctly construct paths
    video_path = os.path.join('data', 'raw', input_video_name)
    bbox_dir = os.path.join('outputs', 'images', 'bbox_images')

    # Ensure directories exist
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    os.makedirs(bbox_dir, exist_ok=True)

    # Download the video if it doesn't exist
    if not os.path.exists(video_path):
        try:
            gdown.download(input_video_link, video_path, quiet=False)
        except Exception as e:
            pytest.fail(f"gdown failed to download {input_video_link}. Error: {e}")

    # Run the main loop to generate the images
    run_main_anomaly_loop(video_path)

    # Check if any JPG files were created in the bbox directory
    jpg_files = glob.glob(os.path.join(bbox_dir, jpg_file_name))
    assert len(jpg_files) > 0, f"No JPG files matching '{jpg_file_name}' were created in {bbox_dir}."

    # Optionally, check the size of the first found image
    assert os.path.getsize(jpg_files[0]) > size, f"The first created JPG file is smaller than {size} bytes."