# YOLO-Anomaly-Detector

This project is a real-time anomaly detection system that uses the YOLO (You Only Look Once) object detection model to learn normal patterns of movement in a video stream and then identify deviations from that routine. It's designed to spot both spatial anomalies (objects in unusual locations) and behavioral anomalies (unusual speed or direction of movement).

## How It Works

The system operates in two main phases:

1.  **Learning Phase (`learn`):**
    *   The system processes a "routine" video that shows normal activity.
    *   It uses YOLOv8 to track all objects and records their positions, speeds, and movement vectors over time.
    *   From this data, it builds two key models:
        *   A **Routine Map (`routine_map.pkl`):** A heatmap that represents the most frequently occupied areas in the frame.
        *   A **Routine Flow (`routine_flow.npy`):** An average vector representing the dominant direction of movement.
    *   It also calculates the average speed and standard deviation of moving objects.

2.  **Detection Phase (`detect`):**
    *   The system analyzes a new video stream, again tracking all objects with YOLO.
    *   For each object, it calculates an anomaly score based on three factors:
        *   **Spatial Anomaly:** How far is the object from the high-traffic areas defined in the Routine Map?
        *   **Directional Anomaly:** Is the object moving against the dominant direction of flow defined in the Routine Flow?
        *   **Speed Anomaly:** Is the object moving significantly faster than the normal speeds observed during the learning phase?
    *   If an object's combined score exceeds a certain threshold, it is flagged as an anomaly, and the output video highlights it.

## Project Structure

The project is organized into a clean and modular structure:

```
YOLO-Anomaly-Detector/
├── .gitignore
├── data/
│   └── (Raw video files for learning and detection)
├── outputs/
│   ├── data/
│   │   ├── routine_map.pkl
│   │   ├── routine_flow.npy
│   │   └── ... (other generated data)
│   ├── images/
│   │   └── (Generated images, like anomaly frames)
│   └── videos/
│       ├── anomaly_detection.mp4
│       └── object_detection.mp4
├── src/
│   ├── __init__.py
│   ├── main.py
│   ├── params_links.py
│   └── ... (other source code)
├── tests/
│   ├── __init__.py
│   └── ... (test scripts)
├── README.md
└── requirements.txt
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

The main script `src/main.py` is controlled via command-line arguments.

### 1. Learn the Routine

First, you need to teach the system what "normal" looks like by providing a routine video. The script will automatically download a default routine video if one is not provided.

```bash
python src/main.py learn --video_path data/raw/routine.mp4 --display
```

*   `--video_path`: Path to the video showing normal activity.
*   `--display`: (Optional) Show the video frames during processing.

This command will generate the `routine_map.pkl`, `routine_flow.npy`, and other model files in the `outputs/data/` directory.

### 2. Detect Anomalies

Once the routine is learned, you can run anomaly detection on a new video. The script will automatically download a default anomaly video if one is not provided.

```bash
python src/main.py detect --video_path data/raw/anomaly.mp4 --display
```

*   `--video_path`: Path to the video where you want to detect anomalies.
*   `--display`: (Optional) Show the video frames during processing.

The resulting video, with anomalies highlighted, will be saved in `outputs/videos/anomaly_detection.mp4`.

## Running Tests

This project uses `pytest`. To run the test suite, navigate to the root directory and run:

```bash
pytest