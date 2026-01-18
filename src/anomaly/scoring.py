import logging
from datetime import datetime

import cv2
import numpy as np

from src.config import ProjectConfig

logger = logging.getLogger(__name__)


def calculate_spatial_scores(tracked_object, routine_map, height, width):
    """
    Calculates the spatial anomaly score for a single tracked object.
    """
    probability_map = (
        (routine_map / routine_map.max())
        if routine_map.max() > 0
        else np.zeros_like(routine_map)
    )

    x1, y1, x2, y2 = tracked_object["bbox"]

    detection_frame = np.zeros((height, width), dtype=np.uint8)
    w, h = x2 - x1, y2 - y1
    if w > 0 and h > 0:
        detection = np.ones((h, w), dtype=np.uint8) * 255
        detection_frame[y1:y2, x1:x2] = detection

    normalized_detection = detection_frame.astype(np.float32) / 255.0
    prediction_map = normalized_detection * probability_map
    # Avoid log(0)
    prediction_map[prediction_map == 0] = 1.0
    log_likelihood_map = 10 * np.log(prediction_map)

    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    opened_map = cv2.morphologyEx(log_likelihood_map, cv2.MORPH_OPEN, kernel_open)
    closed_map = cv2.morphologyEx(opened_map, cv2.MORPH_CLOSE, kernel_close)

    score_region = closed_map[y1:y2, x1:x2]
    mean_score = np.mean(score_region) if score_region.size > 0 else 0.0
    return mean_score


def calculate_spatial_anomaly(
    tracked_objects, routine_map, height, width, config: ProjectConfig
):
    """
    Calculates spatial anomalies for all tracked objects based on a routine map.
    """
    for track_object_index, data in tracked_objects.items():
        score = calculate_spatial_scores(data, routine_map, height, width)
        tracked_objects[track_object_index]["score"] = score
        if score < config.spatial_thresh:
            data["tag"].append("spatial_anomaly")

    return tracked_objects


def calculate_speed_anomaly(
    tracked_objects, mean_speed, std_speed, config: ProjectConfig
):
    """
    Calculates speed anomalies for tracked objects.
    """
    for obj in tracked_objects.values():
        speed = obj.get("speed", 0.0)
        # Check if the track length is sufficient and speed deviates significantly
        if (
            len(obj.get("centers", [])) > config.min_track_length_for_speed
            and speed > mean_speed + config.speed_thresh_factor * std_speed
        ):
            obj["tag"].append("speed_anomaly")
    return tracked_objects


def calculate_directional_anomaly(
    tracked_objects, routine_flow_data, result, config: ProjectConfig
):
    """
    Calculates directional anomalies for tracked objects based on a routine flow histogram.
    `routine_flow_data` is expected to be a dictionary with 'hist' and 'bin_edges'.
    """
    if "hist" not in routine_flow_data or "bin_edges" not in routine_flow_data:
        logger.warning(
            "Warning: Routine flow data is not in expected histogram format. Skipping directional anomaly detection."
        )
        return tracked_objects

    hist = routine_flow_data["hist"]
    bin_edges = routine_flow_data["bin_edges"]
    max_hist_val = np.max(hist)

    for obj in tracked_objects.values():
        if obj.get("track_id") == -1 or "motion_vector" not in obj:
            continue
        current_vector = obj["motion_vector"]
        current_norm = np.linalg.norm(current_vector)
        if current_norm > 1e-6:  # Check for near-zero norm
            current_angle = np.arctan2(current_vector[1], current_vector[0])
            # Find the bin for the current angle
            bin_idx = np.digitize(current_angle, bin_edges) - 1
            # Ensure bin_idx is within valid range
            bin_idx = np.clip(bin_idx, 0, len(hist) - 1)

            # Use the normalized histogram value as a confidence score
            # A low value means it's an infrequent direction
            if max_hist_val > 0:
                direction_confidence = hist[bin_idx] / max_hist_val
            else:
                direction_confidence = 0.0

            # If confidence is below a threshold (e.g., config.direction_thresh can be this threshold)
            # The direction_thresh was previously used as a dot product threshold,
            # now re-purposing for histogram-based confidence.
            if (
                direction_confidence < (1 + config.direction_thresh) / 2
            ):  # Adjusting the range to 0-1
                obj["tag"].append("direction_anomaly")
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = (
                    config.anomaly_frames_dir
                    / f"anomaly_{obj['track_id']}_{timestamp}.jpg"
                )
                cv2.imwrite(str(filename), result.orig_img)
    return tracked_objects


def combine_anomaly_scores(tracked_objects):
    """
    Combines individual anomaly tags into a single 'anomaly' tag if any exist.
    """
    for obj in tracked_objects.values():
        if len(obj.get("tag", [])) > 0:
            obj["tag"].append("anomaly")
    return tracked_objects
