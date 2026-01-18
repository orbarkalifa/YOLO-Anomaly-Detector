import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass
class ProjectConfig:
    """
    Configuration for the YOLO Anomaly Detection project.
    """

    # Run ID
    run_id: str = None

    # Output Directories
    outputs_dir: Path = Path("outputs")
    base_run_dir: Path = outputs_dir / "runs"
    run_dir: Path = None  # To be set in get_config
    data_dir: Path = None
    images_dir: Path = None
    videos_dir: Path = None
    bbox_dir: Path = None
    anomaly_frames_dir: Path = None

    # Output File Paths
    routine_map_path: Path = None
    routine_flow_path: Path = None
    speed_stats_path: Path = None
    tracked_objects_csv: Path = None
    object_detection_video_path: Path = None
    anomaly_detection_video_path: Path = None
    background_image_path: Path = None
    config_path: Path = None

    # Anomaly Thresholds
    spatial_thresh: float = -25.0
    speed_thresh_factor: float = 2.0
    direction_thresh: float = -0.5

    # Tracking
    min_track_length_for_speed: int = 5
    max_center_history: int = 100


def get_config(run_id: str = None) -> ProjectConfig:
    """
    Returns an instance of the project configuration.
    If a run_id is provided, the paths will be updated to be run-specific.
    """
    config = ProjectConfig(run_id=run_id)

    if run_id:
        config.run_dir = config.base_run_dir / run_id
    else:
        config.run_dir = config.outputs_dir

    config.data_dir = config.run_dir / "data"
    config.images_dir = config.run_dir / "images"
    config.videos_dir = config.run_dir / "videos"
    config.bbox_dir = config.images_dir / "bbox_images"
    config.anomaly_frames_dir = config.images_dir / "anomaly_frames"

    config.routine_map_path = config.data_dir / "routine_map.pkl"
    config.routine_flow_path = config.data_dir / "routine_flow.npy"
    config.speed_stats_path = config.data_dir / "normal_speed_stats.npy"
    config.tracked_objects_csv = config.data_dir / "tracked_objects.csv"
    config.object_detection_video_path = config.videos_dir / "object_detection.mp4"
    config.anomaly_detection_video_path = config.videos_dir / "anomaly_detection.mp4"
    config.background_image_path = config.images_dir / "background.png"
    config.config_path = config.run_dir / "config.json"

    return config


def convert_to_serializable(obj):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_to_serializable(elem) for elem in obj]
    return obj


def save_config(config: ProjectConfig):
    """
    Saves the configuration to a JSON file.
    """
    config.run_dir.mkdir(parents=True, exist_ok=True)
    # Convert Path objects to strings and numpy types to Python natives for JSON serialization
    config_dict = convert_to_serializable(asdict(config))

    with open(config.config_path, "w") as f:
        json.dump(config_dict, f, indent=4)
