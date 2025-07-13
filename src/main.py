import pickle
import pandas as pd
from datetime import datetime
from collections import defaultdict
import copy
import time
import numpy as np
import cv2
from pathlib import Path
from ultralytics import YOLO
import os
import gdown
import argparse
from .params_links import *

# Define output paths
OUTPUTS_DIR = Path('outputs')
DATA_DIR = OUTPUTS_DIR / 'data'
IMAGES_DIR = OUTPUTS_DIR / 'images'
VIDEOS_DIR = OUTPUTS_DIR / 'videos'
BBOX_DIR = IMAGES_DIR / 'bbox_images'
ANOMALY_FRAMES_DIR = IMAGES_DIR / 'anomaly_frames'

# Create directories if they don't exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
BBOX_DIR.mkdir(parents=True, exist_ok=True)
ANOMALY_FRAMES_DIR.mkdir(parents=True, exist_ok=True)

ROUTINE_MAP_PATH = DATA_DIR / 'routine_map.pkl'
ROUTINE_FLOW_PATH = DATA_DIR / 'routine_flow.npy'
SPEED_STATS_PATH = DATA_DIR / 'normal_speed_stats.npy'
TRACKED_OBJECTS_CSV = DATA_DIR / 'tracked_objects.csv'
OBJECT_DETECTION_VIDEO_PATH = VIDEOS_DIR / 'object_detection.mp4'
ANOMALY_DETECTION_VIDEO_PATH = VIDEOS_DIR / 'anomaly_detection.mp4'
BACKGROUND_IMAGE_PATH = IMAGES_DIR / 'background.png'


def build_detections_and_routine_map(tracked_objects=None, routine_map=None, width=None, height=None):
    detection_frame = np.zeros((height, width, 3), dtype=np.uint8)
    if tracked_objects:
        for track_id, data in tracked_objects.items():
            x1, y1, x2, y2 = data['bbox']
            w, h = x2 - x1, y2 - y1
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)
            if w > 0 and h > 0:
                detection = np.ones((h, w, 3), dtype=np.uint8) * 255
                detection_frame[y1:y2, x1:x2] = detection
                if routine_map is not None:
                    routine_map[y1:y2, x1:x2] += 1.0
    return routine_map, detection_frame


def run_main_routine_loop(video_path=None, display=False):
    csv_filename, results, bbox_path, frame_counter, csv_header = init_system(video_path)
    normal_speeds = []
    motion_vectors = []
    tracked_objects = {}

    for result in results:
        frame_counter += 1
        if frame_counter == 1:
            height, width, _ = result.orig_img.shape
            cv2.imwrite(str(BACKGROUND_IMAGE_PATH), result.orig_img)
            routine_map = np.zeros((height, width), dtype=np.float32)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(str(OBJECT_DETECTION_VIDEO_PATH), fourcc, 30, (width, height))

        bboxes, labels, confidences = detect_objects(result)
        tracked_objects = track_objects(result, bboxes, labels, confidences, tracked_objects, bbox_path)
        normal_speeds.extend([obj.get('speed', 0.0) for obj in tracked_objects.values()])
        for obj in tracked_objects.values():
            if 'motion_vector' in obj:
                motion_vectors.append(obj['motion_vector'])

        routine_map, detection_frame = build_detections_and_routine_map(tracked_objects, routine_map, width, height)

        if display:
            cv2.imshow('Original Frame', result.plot())
            cv2.imshow('Detection Frame', detection_frame)
            normalized_map = cv2.normalize(routine_map, None, 0, 255, cv2.NORM_MINMAX)
            colored_map = cv2.applyColorMap(normalized_map.astype(np.uint8), cv2.COLORMAP_JET)
            cv2.imshow('Routine Map', colored_map)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if detection_frame is not None:
            video_writer.write(detection_frame)

    speed_stats = {'mean': np.mean(normal_speeds), 'std': np.std(normal_speeds)}
    np.save(SPEED_STATS_PATH, speed_stats)

    if motion_vectors:
        avg_flow = np.mean(motion_vectors, axis=0)
        norm_flow = avg_flow / np.linalg.norm(avg_flow)
        np.save(ROUTINE_FLOW_PATH, norm_flow)
        print(f"Routine flow learned and saved to {ROUTINE_FLOW_PATH}")

    with open(ROUTINE_MAP_PATH, 'wb') as f:
        pickle.dump(routine_map, f)
    print(f"Routine map saved to {ROUTINE_MAP_PATH}")

    if 'video_writer' in locals() and video_writer.isOpened():
        video_writer.release()
    if display:
        cv2.destroyAllWindows()


def detect_directional_anomalies(tracked_objects, routine_flow_vector, result):
    for obj in tracked_objects.values():
        if obj.get('track_id') == -1 or 'motion_vector' not in obj:
            continue
        current_vector = obj['motion_vector']
        current_norm = np.linalg.norm(current_vector)
        if current_norm > 0:
            normalized_current = current_vector / current_norm
            dot_product = np.dot(normalized_current, routine_flow_vector)
            if dot_product < -0.5:
                obj['tag'].append('direction_anomaly')
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                filename = ANOMALY_FRAMES_DIR / f"anomaly_{obj['track_id']}_{timestamp}.jpg"
                cv2.imwrite(str(filename), result.orig_img)
    return tracked_objects


def anomaly_detection(tracked_objects=None, routine_map=None, result=None, height=None, width=None, out=None):
    probability_map = (routine_map / routine_map.max()) if routine_map.max() else np.zeros_like(routine_map)
    _, detection_frame = build_detections_and_routine_map(tracked_objects, routine_map, width, height)
    gray_detection = cv2.cvtColor(detection_frame, cv2.COLOR_BGR2GRAY)
    normalized_detection = gray_detection.astype(np.float32) / 255.0
    prediction_map = normalized_detection * probability_map
    prediction_map[prediction_map == 0] = 1.0
    log_likelihood_map = 10 * np.log(prediction_map)

    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    opened_map = cv2.morphologyEx(log_likelihood_map, cv2.MORPH_OPEN, kernel_open)
    closed_map = cv2.morphologyEx(opened_map, cv2.MORPH_CLOSE, kernel_close)

    for track_object_index, data in tracked_objects.items():
        x1, y1, x2, y2 = data['bbox']
        score_region = closed_map[y1:y2, x1:x2]
        mean_score = np.mean(score_region) if score_region.size > 0 else 0.0
        tracked_objects[track_object_index]['score'] = mean_score

    spatial_thresh = -25.0
    for obj in tracked_objects.values():
        if obj.get('score', 0.0) < spatial_thresh:
            obj['tag'].append('spatial_anomaly')

    output_frame = np.zeros((height, width, 3), dtype=np.uint8)
    for obj in tracked_objects.values():
        if len(obj.get('tag', [])) > 0:
            x1, y1, x2, y2 = obj['bbox']
            if ((x1 + x2) // 2) > (width // 2):
                output_frame[y1:y2, x1:x2] = 255

    if out is not None:
        out.write(output_frame)
    return tracked_objects


def run_main_anomaly_loop(video_path=None, display=False):
    csv_filename, results, bbox_path, frame_counter, csv_header = init_system(video_path)

    with open(ROUTINE_MAP_PATH, 'rb') as f:
        routine_map = pickle.load(f)

    routine_flow = np.load(ROUTINE_FLOW_PATH) if ROUTINE_FLOW_PATH.exists() else np.zeros(2)
    stats = np.load(SPEED_STATS_PATH, allow_pickle=True).item()
    mean_speed, std_speed = stats['mean'], stats['std']

    video_writer_ptr = None
    tracked_objects = defaultdict(dict)

    for result in results:
        frame_counter += 1
        if frame_counter == 1:
            height, width, _ = result.orig_img.shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer_ptr = cv2.VideoWriter(str(ANOMALY_DETECTION_VIDEO_PATH), fourcc, 30, (width, height))

        bboxes, labels, confidences = detect_objects(result=result)
        tracked_objects = track_objects(result=result, bboxes=bboxes, labels=labels, confidences=confidences,
                                        tracked_objects=tracked_objects, bbox_path=bbox_path, save_bbox_images=True)

        for obj in tracked_objects.values():
            speed = obj.get('speed', 0.0)
            if speed > mean_speed + 2 * std_speed:
                obj['tag'].append('speed_anomaly')

        tracked_objects = detect_directional_anomalies(tracked_objects, routine_flow, result)
        tracked_objects = anomaly_detection(tracked_objects=tracked_objects, routine_map=routine_map, result=result,
                                            height=height, width=width, out=video_writer_ptr)

        if display:
            cv2.imshow('Frame', result.plot())
            # cv2.imshow('anomaly frame', output_frame) # output_frame is not defined here
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        df = pd.DataFrame(list(tracked_objects.values()))
        for col in csv_header:
            if col not in df.columns:
                df[col] = None
        df = df[csv_header]
        df.to_csv(csv_filename, mode='a', header=False, index=False)

    if video_writer_ptr is not None:
        video_writer_ptr.release()
    if display:
        cv2.destroyAllWindows()


def download_missing_video_file(video_link=None, video_path=None):
    """Downloads a file from a link if it doesn't exist."""
    video_path = Path(video_path)
    video_path.parent.mkdir(parents=True, exist_ok=True)
    if not video_path.exists():
        print(f"Downloading video to {video_path}...")
        gdown.download(video_link, str(video_path), quiet=False)


def detect_objects(result=None):
    bboxes, labels, confidences = [], [], []
    if result.boxes is not None:
        for box in result.boxes:
            xyxy = box.xyxy.cpu().numpy().astype(int)[0]
            label = int(box.cls.cpu().numpy()[0])
            confidence = float(box.conf.cpu().numpy()[0])
            bboxes.append(xyxy.tolist())
            labels.append(label)
            confidences.append(confidence)
    return bboxes, labels, confidences


def track_objects(result=None, bboxes=None, labels=None, confidences=None, tracked_objects=None, bbox_path=None,
                  save_bbox_images=False):
    track_ids = result.boxes.id.numpy() if result.boxes.id is not None else [-1] * len(bboxes)
    for i, bbox in enumerate(bboxes):
        x1, y1, x2, y2 = bbox
        track_id = track_ids[i]
        object_name = result.names[labels[i]]
        time_date = datetime.now().strftime('%Y%m%d_%H%M%S')

        if save_bbox_images:
            cropped_img = result.orig_img[y1:y2, x1:x2]
            resized_img = cv2.resize(cropped_img, (72, 72))
            filename = f"{track_id}_{time_date}.jpg"
            bbox_image_path = bbox_path / filename
            cv2.imwrite(str(bbox_image_path), resized_img)
        else:
            bbox_image_path = ""

        obj_data = tracked_objects.setdefault(track_id, {
            'track_id': track_id, 'object_name': object_name, 'centers': [], 'tag': []
        })
        obj_data.update({
            'bbox': [x1, y1, x2, y2], 'time_date': time_date, 'confidence': confidences[i],
            'bbox_image_path': str(bbox_image_path)
        })
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
        obj_data['centers'].append((center_x, center_y))

        if len(obj_data['centers']) >= 2:
            dx = obj_data['centers'][-1][0] - obj_data['centers'][-2][0]
            dy = obj_data['centers'][-1][1] - obj_data['centers'][-2][1]
            obj_data['motion_vector'] = (dx, dy)
            obj_data['speed'] = np.sqrt(dx ** 2 + dy ** 2)
        else:
            obj_data['speed'] = 0.0
    return tracked_objects


def init_system(video_path=None):
    model = YOLO('yolov8n.pt')
    results = model.track(video_path, stream=True)
    
    if TRACKED_OBJECTS_CSV.exists():
        TRACKED_OBJECTS_CSV.unlink()
        
    csv_header = ['bbox', 'track_id', 'object_name', 'time_date', 'bbox_image_path', 'confidence', 'score', 'tag']
    pd.DataFrame(columns=csv_header).to_csv(TRACKED_OBJECTS_CSV, index=False)
    
    return TRACKED_OBJECTS_CSV, results, BBOX_DIR, 0, csv_header


def main():
    parser = argparse.ArgumentParser(description="YOLO Anomaly Detection from video streams.")
    subparsers = parser.add_subparsers(dest='command', required=True, help='Available commands')

    # Learn command
    parser_learn = subparsers.add_parser('learn', help='Learn the routine from a video.')
    parser_learn.add_argument('--video_path', type=str, default=video_name_routine,
                              help=f'Path to the routine video file. Defaults to {video_name_routine}')
    parser_learn.add_argument('--display', action='store_true', help='Display video frames during processing.')

    # Detect command
    parser_detect = subparsers.add_parser('detect', help='Detect anomalies in a video.')
    parser_detect.add_argument('--video_path', type=str, default=video_name_anomaly,
                               help=f'Path to the video file for anomaly detection. Defaults to {video_name_anomaly}')
    parser_detect.add_argument('--display', action='store_true', help='Display video frames during processing.')

    args = parser.parse_args()

    # Download required videos if not present
    print("Checking for required video files...")
    download_missing_video_file(video_link=video_link_routine, video_path=video_name_routine)
    download_missing_video_file(video_link=video_link_anomaly, video_path=video_name_anomaly)
    print("Video files are ready.")


    if args.command == 'learn':
        # Now that video_path has a default, we can use it directly
        print(f"Starting routine learning from: {args.video_path}")
        run_main_routine_loop(video_path=args.video_path, display=args.display)
        print("Routine learning complete.")
    elif args.command == 'detect':
        print(f"Starting anomaly detection on: {args.video_path}")
        run_main_anomaly_loop(video_path=args.video_path, display=args.display)
        print("Anomaly detection complete.")

if __name__ == "__main__":
    main()
