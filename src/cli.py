import argparse
import logging
from datetime import datetime
from pathlib import Path

import gdown

from src.config import get_config  # Import get_config to construct pred_path
from src.eval import evaluate  # Import the evaluate function
from src.logging_config import configure_logging  # Import the new logging config
from src.params_links import (
    video_link_anomaly,
    video_link_routine,
    video_name_anomaly,
    video_name_routine,
)
from src.pipeline.detect import detect_anomalies
from src.pipeline.learn import learn_routine


def download_missing_video_file(video_link=None, video_path=None):
    """Downloads a file from a link if it doesn't exist."""
    video_path = Path(video_path)
    video_path.parent.mkdir(parents=True, exist_ok=True)
    if not video_path.exists():
        logging.info(f"Downloading video to {video_path}...")
        gdown.download(
            video_link, str(video_path), quiet=True
        )  # Changed to quiet=True for cleaner logs


def main():
    configure_logging()  # Configure logging at the start of main
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(
        description="YOLO Anomaly Detection from video streams."
    )
    subparsers = parser.add_subparsers(
        dest="command", required=True, help="Available commands"
    )

    # Learn subparser
    learn_subparser = subparsers.add_parser("learn", help="Learn mode")
    learn_subparser.add_argument(
        "--video_path",
        type=str,
        default=video_name_routine,
        help=f"Path to the video file. Defaults to {video_name_routine}",
    )
    learn_subparser.add_argument(
        "--display",
        action="store_true",
        help="Display video frames during processing.",
    )
    learn_subparser.add_argument(
        "--run_id",
        type=str,
        default=datetime.now().strftime("%Y%m%d_%H%M%S"),
        help="Identifier for the run. Defaults to a timestamp.",
    )

    # Detect subparser
    detect_subparser = subparsers.add_parser("detect", help="Detect mode")
    detect_subparser.add_argument(
        "--video_path",
        type=str,
        default=video_name_anomaly,
        help=f"Path to the video file. Defaults to {video_name_anomaly}",
    )
    detect_subparser.add_argument(
        "--display",
        action="store_true",
        help="Display video frames during processing.",
    )
    detect_subparser.add_argument(
        "--run_id",
        type=str,
        default=datetime.now().strftime("%Y%m%d_%H%M%S"),
        help="Identifier for the current detection run. Defaults to a timestamp.",
    )
    detect_subparser.add_argument(
        "--learn_run_id",
        type=str,
        required=True,
        help="Identifier of the run from which to load learned routine models.",
    )

    # Eval subparser
    eval_subparser = subparsers.add_parser("eval", help="Evaluation mode")
    eval_subparser.add_argument(
        "--run_id",
        type=str,
        required=True,
        help="Identifier for the run to evaluate (this is the detection run ID).",
    )
    eval_subparser.add_argument(
        "--gt_path",
        type=str,
        required=True,
        help="Path to the ground truth CSV file.",
    )

    args = parser.parse_args()

    # Download required videos if not present
    logger.info("Checking for required video files...")
    download_missing_video_file(
        video_link=video_link_routine, video_path=video_name_routine
    )
    download_missing_video_file(
        video_link=video_link_anomaly, video_path=video_name_anomaly
    )
    logger.info("Video files are ready.")

    if args.command == "learn":
        logger.info(
            f"Starting routine learning from: {args.video_path} with run ID: {args.run_id}"
        )
        learn_routine(
            video_path=args.video_path, display=args.display, run_id=args.run_id
        )
        logger.info("Routine learning complete.")
    elif args.command == "detect":
        logger.info(
            f"Starting anomaly detection on: {args.video_path} with run ID: {args.run_id}, using learned models from run ID: {args.learn_run_id}"
        )
        detect_anomalies(
            video_path=args.video_path,
            display=args.display,
            run_id=args.run_id,
            learn_run_id=args.learn_run_id,
        )
        logger.info("Anomaly detection complete.")
    elif args.command == "eval":
        logger.info(
            f"Starting evaluation for run ID: {args.run_id} with ground truth from: {args.gt_path}"
        )
        # Call the evaluation function here
        config = get_config(run_id=args.run_id)
        pred_path = config.tracked_objects_csv
        evaluate(gt_path=args.gt_path, pred_path=pred_path, output_dir=config.run_dir)
        logger.info("Evaluation complete.")
