import argparse
import copy
import json
import os
import pickle

import cv2
import numpy as np
import pandas as pd

from src.config import get_config


class TaggingSystem:
    def __init__(self, csv_file, background_image, routine_map_file_name):
        self.current_index_list = None
        self.csv_file = csv_file

        # Load the routine heat map
        with open(routine_map_file_name, "rb") as f:
            self.routine_map = pickle.load(f)

        # Overlay heatmap over background
        heat_map = self.create_heat_map()
        self.background_image = cv2.addWeighted(
            cv2.imread(background_image), 0.3, heat_map, 0.7, 0
        )

        # Load the tracked objects CSV
        self.data = pd.read_csv(self.csv_file)
        self.data["tag"] = ""  # Reset tags

        self.current_track_list = self.data["track_id"].unique()
        self.current_track_indx = 0
        self.current_track = self.current_track_list[self.current_track_indx]

        self.key_pressed = 0
        self.current_track_df = None
        self.extract_track_data()

        self.current_index = 0
        self.continous_play = False
        self.title = None
        self.run_main_loop()

    def create_heat_map(self):
        normalized = self.routine_map / np.max(self.routine_map)
        heatmap_uint8 = (normalized * 255).astype(np.uint8)
        return cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    def run_main_loop(self):
        while self.key_pressed != 27:  # ESC
            self.display_image()

            # Update the window title according to tag
            current_tag = self.data["tag"][self.current_index_list[self.current_index]]
            if isinstance(current_tag, str):
                self.title = "is this an anomaly"
            elif current_tag == 1:
                self.title = "Anomaly"
            else:
                self.title = "Routine"

            # Toggle play mode
            if self.key_pressed == 112:  # 'p'
                self.continous_play = not self.continous_play

            # Next image
            if self.key_pressed == 102:  # 'f'
                self.current_index = (self.current_index + 1) % len(
                    self.current_index_list
                )

            # Previous image
            if self.key_pressed == 98:  # 'b'
                self.current_index = (self.current_index - 1) % len(
                    self.current_index_list
                )

            # Mark anomaly
            if self.key_pressed == 121:  # 'y'
                print(f"✔ self.current_track={self.current_track} → Anomaly")
                found_pandas = self.data["track_id"] == self.current_track
                found_np = np.array(list(found_pandas))
                self.data.loc[found_pandas, ["tag"]] = list(
                    np.ones(found_np[found_np].shape)
                )
                self.data.to_csv(self.csv_file, index=False)
                self.key_pressed = 32  # move to next track

            # Mark routine
            if self.key_pressed == 110:  # 'n'
                print(f"✔ self.current_track={self.current_track} → Routine")
                found_pandas = self.data["track_id"] == self.current_track
                found_np = np.array(list(found_pandas))
                self.data.loc[found_pandas, ["tag"]] = list(
                    np.zeros(found_np[found_np].shape)
                )
                self.data.to_csv(self.csv_file, index=False)
                self.key_pressed = 32  # move to next track

            # Move to next track
            if self.key_pressed == 32:  # 'space'
                if self.current_track_indx < len(self.current_track_list) - 1:
                    self.current_track_indx += 1
                    self.current_track = self.current_track_list[
                        self.current_track_indx
                    ]
                    self.extract_track_data()
                else:
                    print("🎉 All tracks tagged. Exiting... 🏁")
                    break

            # Move to previous track
            if self.key_pressed == 8:  # 'backspace'
                self.current_track_indx = np.clip(
                    self.current_track_indx - 1, 0, len(self.current_track_list) - 1
                )
                self.current_track = self.current_track_list[self.current_track_indx]
                self.extract_track_data()

            # Auto-play
            if self.continous_play:
                self.current_index = (self.current_index + 1) % len(
                    self.current_index_list
                )

        self.data.to_csv(self.csv_file, index=False)
        self.save_labels_to_json()

    def save_labels_to_json(self):
        anomalies = self.data[self.data["tag"] == 1]
        anomaly_events = []
        for track_id in anomalies["track_id"].unique():
            track_data = anomalies[anomalies["track_id"] == track_id]
            # Assuming frame numbers can be inferred from index
            start_frame = track_data.index.min()
            end_frame = track_data.index.max()
            anomaly_events.append(
                {
                    "track_id": int(track_id),
                    "start_frame": int(start_frame),
                    "end_frame": int(end_frame),
                }
            )

        output_path = os.path.dirname(self.csv_file)
        with open(os.path.join(output_path, "labels.json"), "w") as f:
            json.dump(anomaly_events, f, indent=4)
        print(f"Anomaly labels saved to {os.path.join(output_path, 'labels.json')}")

    def extract_track_data(self):
        if self.data.empty:
            raise SystemExit("CSV is empty. Exiting.")

        found_pandas = [False]
        while True not in list(found_pandas) and self.current_track < 1000:
            found_pandas = self.data["track_id"] == self.current_track
            if True in list(found_pandas):
                self.current_track_df = self.data.loc[found_pandas]
                self.current_index_list = self.current_track_df.index
            else:
                self.current_track += 1  # Skip invalid track

    pass

    def display_image(self):
        current_indx = self.current_index_list[self.current_index]

        bbox = [
            int(np.ceil(float(x)))
            for x in self.current_track_df.loc[current_indx, "bbox"]
            .strip("[]")
            .split(",")
        ]
        self.display_img = copy.deepcopy(self.background_image)
        track_id = self.current_track_df.loc[current_indx]["track_id"]
        object_name = self.current_track_df.loc[current_indx, "object_name"]
        confidence = np.round(self.current_track_df.loc[current_indx, "confidence"], 2)
        # ---------------------------------------------------------lab3---------------------------------------------------

        # get count = the number of average events inside the bounding box of the self.routine_map rounded to 0 decimal points

        # draw rectangle with the bounding box boundaries over the self.display_img

        # above each bounding box draw text with the following info: f"ID: {track_id}, Obj: {object_name} {confidence}, counts: {counts}

        # get the bbox_image_path from the self.current_track_df at location current_indx

        # read the bbox_image_path image into self.bbox_image

        # resized_bbox_image the self.bbox_image to have the same size as the bounding bbox

        # put the resized_bbox_image inside self.display_img at the location of the bounding boxes
        # ---------------------------------------------------------lab3---------------------------------------------------

        bbox_image_path = os.path.join(
            self.current_track_df.loc[current_indx, "bbox_image_path"]
        )
        self.bbox_image = cv2.imread(bbox_image_path)
        (y_upper_left, x_upper_left, y_lower_right, x_lower_right) = bbox
        counts = np.round(
            np.mean(
                self.routine_map[x_upper_left:x_lower_right, y_upper_left:y_lower_right]
            ),
            0,
        )
        cv2.putText(
            self.display_img,
            f"ID: {track_id}, Obj: {object_name} {confidence}, counts: {counts}",
            (bbox[0], bbox[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
        )
        self.display_img[bbox[1] : bbox[3], bbox[0] : bbox[2], :] = cv2.resize(
            self.bbox_image, [bbox[2] - bbox[0], bbox[3] - bbox[1]]
        )

        cv2.imshow("image", self.display_img)
        cv2.setWindowTitle("image", self.title)
        self.key_pressed = cv2.waitKey(33)
        # print(self.key_pressed)


def main():
    parser = argparse.ArgumentParser(
        description="Tagging system for YOLO Anomaly Detection."
    )
    parser.add_argument(
        "--run_id", type=str, required=True, help="Run ID to load data from."
    )
    args = parser.parse_args()

    config = get_config(args.run_id)

    TaggingSystem(
        csv_file=str(config.tracked_objects_csv),
        background_image=str(config.background_image_path),
        routine_map_file_name=str(config.routine_map_path),
    )
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
