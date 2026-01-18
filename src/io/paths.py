from src.config import ProjectConfig, save_config


def setup_output_directories(config: ProjectConfig):
    """
    Creates the necessary output directories for the project and saves the config.
    """
    config.data_dir.mkdir(parents=True, exist_ok=True)
    config.images_dir.mkdir(parents=True, exist_ok=True)
    config.videos_dir.mkdir(parents=True, exist_ok=True)
    config.bbox_dir.mkdir(parents=True, exist_ok=True)
    config.anomaly_frames_dir.mkdir(parents=True, exist_ok=True)
    save_config(config)
