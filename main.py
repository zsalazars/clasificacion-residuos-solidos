import os
import cv2
import random
import warnings
import argparse
import logging
import numpy as np
import datetime
import time
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


from typing import List, Tuple, Dict
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
        default="http://192.168.1.5:5000/video_feed",
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
    parser.add_argument(
        "--log-file",
        type=str,
        default="recognition_log.txt",
        help="Path to the log file"
    )
    return parser.parse_args()

def setup_logging(level: str, log_file: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), None),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, mode='a')
        ]
    )

def log_recognition(name: str, similarity: float) -> None:
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logging.info(f"{timestamp} - Recognized: {name} with similarity {similarity:.2f}")

from tqdm import tqdm

def build_targets(detector, recognizer, faces_dir: str) -> Dict[str, np.ndarray]:
    targets = {}

    def process_person(person_path: str, person_name: str) -> Tuple[str, np.ndarray]:
        embeddings = []
        image_files = [os.path.join(person_path, filename) for filename in os.listdir(person_path) if filename.lower().endswith(('.jpg', '.png', '.jpeg'))]
        for image_path in tqdm(image_files, desc=f"Processing {person_name}", unit="image"):
            image = cv2.imread(image_path)
            bboxes, kpss = detector.detect(image, max_num=1)
            if len(kpss) == 0:
                logging.warning(f"No face detected in {image_path}. Skipping...")
                continue
            embedding = recognizer(image, kpss[0])
            embeddings.append(embedding)
        if embeddings:
            mean_embedding = np.mean(embeddings, axis=0)
            return person_name, mean_embedding
        return None

    person_dirs = [os.path.join(faces_dir, d) for d in os.listdir(faces_dir) if os.path.isdir(os.path.join(faces_dir, d))]
    with ThreadPoolExecutor() as executor:
        futures = []
        for person_path in person_dirs:
            person_name = os.path.basename(person_path)
            futures.append(executor.submit(process_person, person_path, person_name))
        for future in as_completed(futures):
            result = future.result()
            if result:
                name, embedding = result
                targets[name] = embedding

    return targets


def prompt_user_for_confirmation(name: str, similarity: float) -> bool:
    while True:
        black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(black_frame, 'Confirming identity...', (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow("Frame", black_frame)
        user_input = input(f"Recognized as {name} with similarity {similarity:.2f}. Is this correct? (yes/no): ").strip().lower()
        if user_input in ['yes', 'no']:
            cv2.destroyWindow("Frame")
            return user_input == 'yes'
        print("Invalid input. Please enter 'yes' or 'no'.")

def save_recognized_face(frame: np.ndarray, name: str, faces_dir: str) -> None:
    person_folder = os.path.join(faces_dir, name)
    if not os.path.exists(person_folder):
        os.makedirs(person_folder)
    filename = f"recognized_face_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
    filepath = os.path.join(person_folder, filename)
    cv2.imwrite(filepath, frame)

def frame_processor(
    frame: np.ndarray,
    detector: SCRFD,
    recognizer: ArcFace,
    targets: Dict[str, np.ndarray],
    colors: Dict[str, Tuple[int, int, int]],
    params,
    recognized_faces: set,
    last_reset_time: float
) -> Tuple[np.ndarray, float]:
    current_time = time.time()
    
    if current_time - last_reset_time > 60:
        recognized_faces.clear()
        last_reset_time = current_time

    bboxes, kpss = detector.detect(frame, params.max_num)

    for bbox, kps in zip(bboxes, kpss):
        *bbox, conf_score = bbox.astype(np.int32)
        embedding = recognizer(frame, kps)

        max_similarity = 0
        best_match_name = "Unknown"
        for name, target in targets.items():
            similarity = compute_similarity(target, embedding)
            if similarity > max_similarity and similarity > params.similarity_thresh:
                max_similarity = similarity
                best_match_name = name

        if best_match_name != "Unknown" and best_match_name not in recognized_faces:
            if prompt_user_for_confirmation(best_match_name, max_similarity):
                log_recognition(best_match_name, max_similarity)
                recognized_faces.add(best_match_name)
                save_recognized_face(frame, best_match_name, params.faces_dir)

        if best_match_name == "Unknown":
            color = (255, 0, 0)  # Red color for unknown
            label = "Desconocido"
        else:
            color = colors[best_match_name]
            label = f"{best_match_name} ({max_similarity:.2f})"

        draw_bbox_info(frame, bbox, similarity=max_similarity, name=label, color=color)

    return frame, last_reset_time

def main(params):
    setup_logging(params.log_level, params.log_file)

    detector = SCRFD(params.det_weight, input_size=(640, 640), conf_thres=params.confidence_thresh)
    recognizer = ArcFace(params.rec_weight)

    targets = build_targets(detector, recognizer, params.faces_dir)
    colors = {name: (random.randint(0, 256), random.randint(0, 256), random.randint(0, 256)) for name in targets}

    cap = cv2.VideoCapture(params.source)
    if not cap.isOpened():
        raise Exception("Could not open video or webcam")

    recognized_faces = set()
    last_reset_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame, last_reset_time = frame_processor(frame, detector, recognizer, targets, colors, params, recognized_faces, last_reset_time)
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
