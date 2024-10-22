import os
import cv2
import random
import warnings
import argparse
import logging
import numpy as np
import mss
import pygetwindow as gw

import onnxruntime
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
        default="./faces",
        help="Path to faces stored directory"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="chrome",
        help="Specify 'chrome' for Chrome window capture, or 'full_screen' for full screen capture"
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

        if best_match_name == "Unknown":
            color = (255, 0, 0)
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

    if params.source == "chrome":
        windows = gw.getWindowsWithTitle("Google Chrome")
        if not windows:
            raise Exception("Google Chrome window not found")
        
        window = windows[0]
        monitor = {
            "top": window.top,
            "left": window.left,
            "width": window.width,
            "height": window.height
        }
        
        with mss.mss() as sct:
            while True:
                screen = sct.grab(monitor)
                frame = np.array(screen)
                
                # Convert from RGB to BGR
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                frame = frame_processor(frame, detector, recognizer, targets, colors, params)
                cv2.imshow("Frame", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    elif params.source == "full_screen":
        with mss.mss() as sct:
            monitor = sct.monitors[1]  # Use the primary monitor
            while True:
                screen = sct.grab(monitor)
                frame = np.array(screen)

                # Convert from RGB to BGR
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                frame = frame_processor(frame, detector, recognizer, targets, colors, params)
                cv2.imshow("Frame", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    else:
        raise Exception("Invalid source specified. Use 'chrome' or 'full_screen'.")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    args = parse_args()
    main(args)
