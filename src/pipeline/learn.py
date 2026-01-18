import logging
import pickle

import cv2
import numpy as np

from src.anomaly.scoring import calculate_spatial_scores
from src.config import get_config, save_config
from src.io.paths import setup_output_directories
from src.tracking.tracker import ObjectTracker

logger = logging.getLogger(__name__)


def build_routine_map(tracked_objects, routine_map, width, height):
    detection_frame = np.zeros((height, width, 3), dtype=np.uint8)
    if tracked_objects:
        for _track_id, data in tracked_objects.items():
            x1, y1, x2, y2 = data["bbox"]
            w, h = x2 - x1, y2 - y1
            if w > 0 and h > 0:
                detection = np.ones((h, w, 3), dtype=np.uint8) * 255
                detection_frame[y1:y2, x1:x2] = detection
                if routine_map is not None:
                    routine_map[y1:y2, x1:x2] += 1.0
    return routine_map, detection_frame


def learn_routine(video_path: str, display: bool, run_id: str):
    config = get_config(run_id=run_id)
    setup_output_directories(config)

    tracker = ObjectTracker()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Error: Could not open video {video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    routine_map = np.zeros((height, width), dtype=np.float32)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(
        str(config.object_detection_video_path), fourcc, fps, (width, height)
    )

    cap_read_ret, background_frame = cap.read()
    if cap_read_ret:
        cv2.imwrite(str(config.background_image_path), background_frame)
    else:
        logger.warning(
            "Warning: Could not read first frame for background image. Skipping background image save."
        )

    tracked_objects = {}
    normal_speeds = []
    motion_vectors = []

    frame_counter = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_counter += 1
        results = tracker.model.track(source=frame, persist=True)
        result = results[0]

        bboxes, labels, confidences = tracker.detect_objects(result)
        tracked_objects = tracker.track_objects(
            result,
            bboxes,
            labels,
            confidences,
            tracked_objects,
            config.bbox_dir,
            fps=fps,
            config=config,
        )

        for obj in tracked_objects.values():
            if len(obj.get("centers", [])) > config.min_track_length_for_speed:
                normal_speeds.append(obj.get("speed", 0.0))
            if "motion_vector" in obj:
                motion_vectors.append(obj["motion_vector"])

        routine_map, detection_frame = build_routine_map(
            tracked_objects, routine_map, width, height
        )

        if display:
            cv2.imshow("Original Frame", result.plot())
            cv2.imshow("Detection Frame", detection_frame)
            normalized_map = cv2.normalize(routine_map, None, 0, 255, cv2.NORM_MINMAX)
            colored_map = cv2.applyColorMap(
                normalized_map.astype(np.uint8), cv2.COLORMAP_JET
            )
            cv2.imshow("Routine Map", colored_map)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        if detection_frame is not None:
            video_writer.write(detection_frame)

    cap.release()

    if routine_map.max() == 0:
        logger.warning(
            f"Warning: No objects were tracked in the routine video. Deleting run directory {config.run_dir} and exiting."
        )
        import shutil

        shutil.rmtree(config.run_dir)
        return

    # Use median and MAD for robust statistics
    if normal_speeds:
        median_speed = np.median(normal_speeds)
        mad_speed = np.median(np.abs(normal_speeds - median_speed))
        speed_stats = {"median": median_speed, "mad": mad_speed}
    else:
        speed_stats = {"median": 0, "mad": 0}
    np.save(config.speed_stats_path, speed_stats)

    # Calculate spatial scores and save threshold
    all_scores = []

    for obj in tracked_objects.values():
        score = calculate_spatial_scores(obj, routine_map, height, width)
        all_scores.append(score)

    if all_scores:
        spatial_thresh = np.percentile(all_scores, 5)
        config.spatial_thresh = float(spatial_thresh)

        save_config(config)
        logger.info(f"Spatial threshold calibrated and saved: {spatial_thresh}")

    if motion_vectors:
        angles = np.array([np.arctan2(v[1], v[0]) for v in motion_vectors])
        hist, bin_edges = np.histogram(angles, bins=36, range=(-np.pi, np.pi))
        np.save(config.routine_flow_path, {"hist": hist, "bin_edges": bin_edges})
        logger.info(
            f"Routine flow histogram learned and saved to {config.routine_flow_path}"
        )

    with open(config.routine_map_path, "wb") as f:
        pickle.dump(routine_map, f)
    logger.info(f"Routine map saved to {config.routine_map_path}")

    if video_writer.isOpened():
        video_writer.release()
    if display:
        cv2.destroyAllWindows()
