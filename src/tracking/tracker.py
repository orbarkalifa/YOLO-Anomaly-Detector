from datetime import datetime

import cv2
import numpy as np
from ultralytics import YOLO


class ObjectTracker:
    def __init__(self, model_path="yolov8n.pt"):
        self.model = YOLO(model_path)

    def track_frame(self, frame):
        return self.model.track(frame, stream=True)

    def detect_objects(self, result=None):
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

    def track_objects(
        self,
        result=None,
        bboxes=None,
        labels=None,
        confidences=None,
        tracked_objects=None,
        bbox_path=None,
        save_bbox_images=False,
        fps=30,
        config=None,  # Add config parameter
    ):
        try:
            track_ids = (
                result.boxes.id.cpu().numpy()
                if result.boxes.id is not None
                else [-1] * len(bboxes)
            )
        except AttributeError:
            track_ids = [-1] * len(bboxes)

        dt = 1 / fps if fps > 0 else 1

        for i, bbox in enumerate(bboxes):
            x1, y1, x2, y2 = bbox
            track_id = int(track_ids[i])

            if track_id == -1:
                continue

            object_name = result.names[labels[i]]
            time_date = datetime.now().strftime("%Y%m%d_%H%M%S")

            if save_bbox_images:
                bbox_path.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
                cropped_img = result.orig_img[y1:y2, x1:x2]
                resized_img = cv2.resize(cropped_img, (72, 72))
                filename = f"{track_id}_{time_date}.jpg"
                bbox_image_path = bbox_path / filename
                cv2.imwrite(str(bbox_image_path), resized_img)
            else:
                bbox_image_path = ""

            obj_data = tracked_objects.setdefault(
                track_id,
                {
                    "track_id": track_id,
                    "object_name": object_name,
                    "centers": [],
                    "tag": [],
                },
            )
            obj_data.update(
                {
                    "bbox": [x1, y1, x2, y2],
                    "time_date": time_date,
                    "confidence": confidences[i],
                    "bbox_image_path": str(bbox_image_path),
                }
            )
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
            obj_data["centers"].append((center_x, center_y))

            # Trim centers list to optimize memory usage
            if config and len(obj_data["centers"]) > config.max_center_history:
                obj_data["centers"] = obj_data["centers"][-config.max_center_history :]

            if len(obj_data["centers"]) >= 2:
                dx = obj_data["centers"][-1][0] - obj_data["centers"][-2][0]
                dy = obj_data["centers"][-1][1] - obj_data["centers"][-2][1]
                obj_data["motion_vector"] = (dx, dy)
                speed_px_per_frame = np.sqrt(dx**2 + dy**2)
                obj_data["speed"] = speed_px_per_frame / dt  # pixels per second
            else:
                obj_data["speed"] = 0.0
        return tracked_objects
