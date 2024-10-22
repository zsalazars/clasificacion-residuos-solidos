import os
import cv2
import random
import warnings
import argparse
import logging
import numpy as np
from typing import List, Tuple
from models import SCRFD, ArcFace
from utils.helpers import compute_similarity, draw_bbox_info

warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser(description="Face Detection and Recognition")
    parser.add_argument(
        "--det-weight",
        type=str,
        default="./weights/det_10g.onnx",
        help="Path to detection model"
    )
    parser.add_argument(
        "--rec-weight",
        type=str,
        default="./weights/w600k_r50.onnx",
        help="Path to recognition model"
    )
    parser.add_argument(
        "--similarity-thresh",
        type=float,
        default=0.4,
        help="Similarity threshold between faces"
    )
    parser.add_argument(
        "--confidence-thresh",
        type=float,
        default=0.5,
        help="Confidence threshold for face detection"
    )
    parser.add_argument(
        "--faces-dir",
        type=str,
        default="./facesog",
        help="Path to faces stored directory"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Video file or video camera source. i.e 0 - webcam"
    )
    parser.add_argument(
        "--max-num",
        type=int,
        default=0,
        help="Maximum number of face detections from a frame"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level"
    )
    return parser.parse_args()

def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), None),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

def build_targets(detector, recognizer, params) -> List[Tuple[np.ndarray, str]]:
    """
    Build targets using face detection and recognition.

    Args:
        detector (SCRFD): Face detector model.
        recognizer (ArcFace): Face recognizer model.
        params (argparse.Namespace): Command line arguments.

    Returns:
        List[Tuple[np.ndarray, str]]: A list of tuples containing feature vectors and corresponding image names.
    """
    targets = []
    for filename in os.listdir(params.faces_dir):
        name = filename[:-4]
        image_path = os.path.join(params.faces_dir, filename)

        image = cv2.imread(image_path)
        bboxes, kpss = detector.detect(image, max_num=1)

        if len(kpss) == 0:
            logging.warning(f"No face detected in {image_path}. Skipping...")
            continue

        embedding = recognizer(image, kpss[0])
        targets.append((embedding, name))

    return targets

def frame_processor(
    frame: np.ndarray,
    detector: SCRFD,
    recognizer: ArcFace,
    targets: List[Tuple[np.ndarray, str]],
    colors: dict,
    params
) -> np.ndarray:
    """
    Process a video frame for face detection and recognition.

    Args:
        frame (np.ndarray): The video frame.
        detector (SCRFD): Face detector model.
        recognizer (ArcFace): Face recognizer model.
        targets (List[Tuple[np.ndarray, str]]): List of target feature vectors and names.
        colors (dict): Dictionary of colors for drawing bounding boxes.
        params (argparse.Namespace): Command line arguments.

    Returns:
        np.ndarray: The processed video frame.
    """
    bboxes, kpss = detector.detect(frame, params.max_num)

    for bbox, kps in zip(bboxes, kpss):
        *bbox, conf_score = bbox.astype(np.int32)
        embedding = recognizer(frame, kps)

        max_similarity = 0
        best_match_name = "Unknown"
        for target, name in targets:
            similarity = compute_similarity(target, embedding)
            if similarity > max_similarity and similarity > params.similarity_thresh:
                max_similarity = similarity
                best_match_name = name

        # Color for 'Unknown' and the label
        if best_match_name == "Unknown":
            color = (255, 0, 0)  # Red color for unknown
            label = "Desconocido"
        else:
            color = colors[best_match_name]
            label = f"{best_match_name} ({max_similarity:.2f})"

        draw_bbox_info(frame, bbox, similarity=max_similarity, name=label, color=color)

    return frame

def main(params):
    setup_logging(params.log_level)

    detector = SCRFD(params.det_weight, input_size=(640, 640), conf_thres=params.confidence_thresh)
    recognizer = ArcFace(params.rec_weight)

    targets = build_targets(detector, recognizer, params)
    colors = {name: (random.randint(0, 256), random.randint(0, 256), random.randint(0, 256)) for _, name in targets}

    cap = cv2.VideoCapture(params.source)
    if not cap.isOpened():
        raise Exception("Could not open video or webcam")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = frame_processor(frame, detector, recognizer, targets, colors, params)
        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    args = parse_args()
    if args.source.isdigit():
        args.source = int(args.source)
    main(args)
