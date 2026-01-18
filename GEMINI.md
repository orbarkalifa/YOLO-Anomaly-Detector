# Gemini Code Assistant Context

This document provides context for the Gemini AI assistant to understand and effectively assist with the YOLO-Anomaly-Detector project.

## Project Overview

The YOLO-Anomaly-Detector is a real-time anomaly detection system written in Python. It uses the YOLOv8 object detection model to learn normal patterns of movement in a video stream and then identify deviations from that routine. The system can detect both spatial anomalies (objects in unusual locations) and behavioral anomalies (unusual speed or direction of movement).

The project is structured as a Python package with a modular architecture:
- `src/anomaly`: Contains the logic for anomaly scoring.
- `src/pipeline`: Holds the main pipelines for learning and detection.
- `src/tracking`: Encapsulates the object tracking functionality.
- `src/io`: Manages output directory structures.
- `src/config.py`: Centralized configuration for the project.
- `src/cli.py`: The command-line interface for the application.

The primary technologies used are Python, the `ultralytics` library for YOLOv8, `pandas` for data manipulation, and `scikit-learn` for machine learning calculations.

## Building and Running

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/orbarkalifa/YOLO-Anomaly-Detector.git
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

### Usage

The application is controlled via the command-line interface in `src/cli.py`. You can run it as a module.

#### 1. Learn the Routine

To teach the system what "normal" looks like, run the following command:

```bash
python -m src learn --video_path data/raw/routine.mp4 --display
```

*   `--video_path`: Path to the video showing normal activity.
*   `--display`: (Optional) Show the video frames during processing.

This command will generate the `routine_map.pkl`, `routine_flow.npy`, and other model files in the `outputs/data/` directory.

#### 2. Detect Anomalies

Once the routine is learned, you can run anomaly detection on a new video:

```bash
python -m src detect --video_path data/raw/anomaly.mp4 --display
```

*   `--video_path`: Path to the video where you want to detect anomalies.
*   `--display`: (Optional) Show the video frames during processing.

The resulting video, with anomalies highlighted, will be saved in `outputs/videos/anomaly_detection.mp4`.


## Development Conventions

### Testing

The project uses `pytest` for testing. To run the test suite, navigate to the root directory and run:

```bash
pytest
```