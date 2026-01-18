import logging
import pickle
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd

from src.anomaly.scoring import (
    calculate_directional_anomaly,
    calculate_spatial_anomaly,
    calculate_speed_anomaly,
    combine_anomaly_scores,
)
from src.config import get_config
from src.tracking.tracker import ObjectTracker

logger = logging.getLogger(__name__)


def detect_anomalies(video_path: str, display: bool, run_id: str, learn_run_id: str):
    config = get_config(run_id=run_id)
    # Note: We do not call setup_output_directories here,
    # as we assume the learn command has already created the run directory.
    # However, we need to create the directories for the detect phase.
    config.videos_dir.mkdir(parents=True, exist_ok=True)
    config.data_dir.mkdir(parents=True, exist_ok=True)

    tracker = ObjectTracker()

    # Load routine data from the specified run
    learn_config = get_config(run_id=learn_run_id)

    try:
        with open(learn_config.routine_map_path, "rb") as f:
            routine_map = pickle.load(f)
    except FileNotFoundError:
        logger.error(
            f"Error: Routine map not found at {learn_config.routine_map_path}. "
            f"Please run the 'learn' command with run_id='{learn_run_id}' first."
        )
        return

    routine_flow_data = (
        np.load(learn_config.routine_flow_path, allow_pickle=True).item()
        if learn_config.routine_flow_path.exists()
        else {"hist": np.array([]), "bin_edges": np.array([])}
    )

    try:
        stats = np.load(learn_config.speed_stats_path, allow_pickle=True).item()
        # Supporting both old and new stats file format
        if "median" in stats:
            mean_speed, std_speed = stats["median"], stats["mad"]
        else:
            mean_speed, std_speed = stats["mean"], stats["std"]

    except FileNotFoundError:
        logger.error(
            f"Error: Speed stats not found at {learn_config.speed_stats_path}. "
            f"Please run the 'learn' command with run_id='{learn_run_id}' first."
        )
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Error: Could not open video {video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(
        str(config.anomaly_detection_video_path), fourcc, fps, (width, height)
    )
    if not video_writer.isOpened():
        logger.error(
            f"Error: Could not open video writer for {config.anomaly_detection_video_path}"
        )
        cap.release()
        return

    tracked_objects = defaultdict(dict)

    if config.tracked_objects_csv.exists():
        config.tracked_objects_csv.unlink()

    csv_header = [
        "bbox",
        "track_id",
        "object_name",
        "time_date",
        "bbox_image_path",
        "confidence",
        "score",
        "tag",
    ]
    pd.DataFrame(columns=csv_header).to_csv(config.tracked_objects_csv, index=False)

    frame_counter = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_counter += 1
        results = tracker.model.track(source=frame, persist=True)
        result = results[0]

        # Reset tags for each frame to avoid accumulation
        for k in tracked_objects:
            tracked_objects[k]["tag"] = []

        bboxes, labels, confidences = tracker.detect_objects(result=result)
        tracked_objects = tracker.track_objects(
            result=result,
            bboxes=bboxes,
            labels=labels,
            confidences=confidences,
            tracked_objects=tracked_objects,
            bbox_path=config.bbox_dir,
            save_bbox_images=True,
            fps=fps,
            config=config,
        )

        tracked_objects = calculate_speed_anomaly(
            tracked_objects, mean_speed, std_speed, config
        )
        tracked_objects = calculate_directional_anomaly(
            tracked_objects, routine_flow_data, result, config
        )
        tracked_objects = calculate_spatial_anomaly(
            tracked_objects, routine_map, height, width, config
        )
        tracked_objects = combine_anomaly_scores(tracked_objects)  # Add this line

        output_frame = result.orig_img.copy()
        for obj in tracked_objects.values():
            if len(obj.get("tag", [])) > 0:
                x1, y1, x2, y2 = obj["bbox"]
                cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(
                    output_frame,
                    ",".join(obj["tag"]),
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    2,
                )

        if display:
            cv2.imshow("Frame", output_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        video_writer.write(output_frame)

        df = pd.DataFrame(list(tracked_objects.values()))
        for col in csv_header:
            if col not in df.columns:
                df[col] = None
        df = df[csv_header]
        df.to_csv(config.tracked_objects_csv, mode="a", header=False, index=False)

    cap.release()
    video_writer.release()
    if display:
        cv2.destroyAllWindows()
