# YOLO-Anomaly-Detector

This project is a real-time anomaly detection system that uses the YOLO (You Only Look Once) object detection model to learn normal patterns of movement in a video stream and then identify deviations from that routine. It's designed to spot both spatial anomalies (objects in unusual locations) and behavioral anomalies (unusual speed or direction of movement).

## Problem Statement

Traditional video surveillance often relies on human operators, which is prone to errors due to fatigue and the sheer volume of data. Automating anomaly detection can significantly improve efficiency and accuracy in identifying unusual events, ensuring quicker responses and better security. This project aims to provide a robust and extensible framework for real-time anomaly detection in video streams.

## How It Works

The system operates in two main phases:

1.  **Learning Phase (`learn`):**
    *   The system processes a "routine" video that shows normal activity.
    *   It uses YOLOv8 to track all objects and records their positions, speeds, and movement vectors over time.
    *   From this data, it builds key models:
        *   A **Routine Map (`routine_map.pkl`):** A heatmap that represents the most frequently occupied areas in the frame.
        *   A **Routine Flow (`routine_flow.npy`):** A histogram of movement angles representing the dominant directions of movement.
        *   **Speed Statistics (`normal_speed_stats.npy`):** Median and Median Absolute Deviation (MAD) of normal object speeds.

2.  **Detection Phase (`detect`):**
    *   The system analyzes a new video stream, again tracking all objects with YOLO.
    *   For each object, it calculates an anomaly score based on three factors:
        *   **Spatial Anomaly:** How infrequently does the object appear in its current location according to the Routine Map?
        *   **Directional Anomaly:** Does the object's movement direction significantly deviate from the Routine Flow?
        *   **Speed Anomaly:** Is the object moving significantly faster than the normal speeds observed during the learning phase?
    *   If an object's combined score exceeds a certain threshold, it is flagged as an anomaly, and the output video highlights it.

3.  **Evaluation Phase (`eval`):**
    *   Compares the system's anomaly detections against a provided ground truth dataset.
    *   Generates Precision-Recall (PR) and Receiver Operating Characteristic (ROC) curves to quantitatively assess performance.

## Approach Diagram (Conceptual)

```
[Input Video] -- (YOLOv8 Object Tracking) --> [Object Trajectories]
                                                     |
                                                     V
                                         +-----------------------+
                                         |     LEARNING PHASE    |
                                         +-----------------------+
                                          |
                                          V
                          +-------------------------------+
                          |  Routine Map (Spatial)        |
                          |  Routine Flow (Directional)   |
                          |  Speed Statistics (Behavioral)|
                          +-------------------------------+
                                          |
                                          V
[New Video Stream] -- (YOLOv8 Object Tracking) --> [Current Object States]
                                                     |
                                                     V
                                         +-----------------------+
                                         |    DETECTION PHASE    |
                                         +-----------------------+
                                          |
                                          V
                          +-------------------------------+
                          |  Anomaly Scoring (Spatial)    |
                          |  Anomaly Scoring (Directional)|
                          |  Anomaly Scoring (Speed)      |
                          +-------------------------------+
                                          |
                                          V
                                   [Anomaly Flags] --> [Annotated Output Video]
                                          |
                                          V
                                         +-----------------------+
                                         |   EVALUATION PHASE    |
                                         +-----------------------+
                                          |
                                          V
                                   [Performance Metrics & Plots]
```

## Project Structure

The project is organized into a clean and modular structure:

```
YOLO-Anomaly-Detector/
├── .gitignore
├── data/
│   └── raw/               # Raw video files
│       ├── routine.mp4
│       └── anomaly.mp4
├── outputs/
│   └── runs/              # Experiment runs are stored here
│       └── <run_id>/
│           ├── config.json            # Configuration for the run
│           ├── data/                  # Generated data (routine maps, tracked objects CSV)
│           │   ├── routine_map.pkl
│           │   ├── routine_flow.npy
│           │   ├── normal_speed_stats.npy
│           │   └── tracked_objects.csv
│           ├── images/                # Generated images (anomaly frames, bbox images)
│           └── videos/                # Generated videos (object detection, anomaly detection)
├── src/
│   ├── anomaly/           # Anomaly scoring logic
│   ├── io/                # Input/Output utilities
│   ├── pipeline/          # Main learning and detection pipelines
│   ├── tracking/          # Object tracking utilities
│   ├── cli.py             # Command-line interface
│   ├── config.py          # Project configuration (dataclass)
│   ├── eval.py            # Evaluation script
│   ├── logging_config.py  # Centralized logging configuration
│   └── params_links.py    # External video download links
├── tests/                 # Unit and integration tests
├── README.md              # This file
├── requirements.txt       # Python dependencies
└── pyproject.toml         # Project metadata and tool configurations (e.g., ruff)
```

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/<your-username>/YOLO-Anomaly-Detector.git
    cd YOLO-Anomaly-Detector
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

The application is controlled via the command-line interface `src/cli.py`. You can run it as a Python module.

### 1. Learn the Routine

Teach the system what "normal" looks like by providing a routine video. The script will automatically download default videos if not present.

```bash
python -m src learn --video_path data/raw/routine.mp4 --display
```

*   `--video_path`: Path to the video showing normal activity.
*   `--display`: (Optional) Show the video frames during processing.
*   `--run_id`: (Optional) Identifier for the run. Defaults to a timestamp, creating `outputs/runs/<timestamp>/`.

This command generates the routine models (`routine_map.pkl`, `routine_flow.npy`, `normal_speed_stats.npy`) and a `config.json` in the `outputs/runs/<run_id>/data/` directory.

### 2. Detect Anomalies

Run anomaly detection on a new video using the previously learned routine.

```bash
python -m src detect --video_path data/raw/anomaly.mp4 --run_id <LEARN_RUN_ID> --display
```

*   `--video_path`: Path to the video where you want to detect anomalies.
*   `--run_id`: The `run_id` from the learning phase. This tells the system which learned models to use.
*   `--display`: (Optional) Show the video frames during processing.

The resulting video, with anomalies highlighted, will be saved in `outputs/runs/<run_id>/videos/anomaly_detection.mp4`. Tracking data is saved to `outputs/runs/<run_id>/data/tracked_objects.csv`.

### 3. Evaluate Results

Evaluate the performance of the anomaly detection system against ground truth.

```bash
python -m src eval --run_id <DETECT_RUN_ID> --gt_path data/ground_truth.json
```

*   `--run_id`: The `run_id` from the detection phase whose `tracked_objects.csv` you want to evaluate.
*   `--gt_path`: Path to a JSON file containing ground truth anomaly intervals.

This command generates `precision_recall_curve.png` and `roc_curve.png` in the `outputs/runs/<run_id>/` directory.

## Running Tests

This project uses `pytest`. To run the test suite, navigate to the root directory and run:

```bash
pytest
```

### Ground Truth Format (Example: `data/ground_truth.json`)

```json
[
    {"start_frame": 10, "end_frame": 50},
    {"start_frame": 120, "end_frame": 180}
]
```
Each object in the array represents an anomaly event, specified by its `start_frame` and `end_frame` (inclusive).

## Metrics Table (Placeholder)

| Metric            | Value (Example) |
| :---------------- | :-------------- |
| Precision         | 0.85            |
| Recall            | 0.72            |
| F1-Score          | 0.78            |
| ROC AUC           | 0.91            |
| PR AUC            | 0.88            |

*Note: Actual metrics will be generated upon running the `eval` command.*

## Development & Quality Control

### Running Tests

This project uses `pytest` for testing. To run the test suite, navigate to the root directory and run:

```bash
pytest
```
Or to run a specific test:
```bash
python -m pytest tests/test_my_feature.py
```

### Linting and Formatting

The project uses `ruff` for code linting and formatting.
To check for issues:
```bash
ruff check .
```
To automatically fix issues:
```bash
ruff format .
```

## Known Limitations and Next Steps

*   **Single Class Detection:** Currently optimized for detecting "person" objects. Extending to multiple object classes would require more nuanced routine learning.
*   **Static Thresholds:** While spatial thresholds are calibrated, speed and directional thresholds are still fixed constants. Dynamic thresholding based on learned distributions would improve robustness.
*   **Simple Anomaly Aggregation:** The current system flags anomalies per-frame and per-object. More sophisticated temporal aggregation could reduce false positives and provide event-level anomaly reports.
*   **Limited Evaluation Metrics:** Currently, only PR and ROC curves are generated. Integrating metrics like False Positive Rate per minute or event-level evaluation would provide a more complete picture.
*   **No UI/Dashboard:** The project is CLI-based. A web-based UI for visualization and interaction would enhance usability.
*   **Scalability:** Processing large volumes of video data in real-time may require distributed processing solutions.