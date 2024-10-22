import os
import cv2
import random
import warnings
import argparse
import logging
import numpy as np
import pyttsx3
import time
import threading
import requests  # Para las peticiones HTTP
import keyboard  # Para la simulación con teclado
from typing import List, Tuple
from models import SCRFD, ArcFace
from utils.helpers import compute_similarity, draw_bbox_info

warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser(description="Face Detection and Recognition")
    parser.add_argument("--det-weight", type=str, default="./weights/det_10g.onnx", help="Path to detection model")
    parser.add_argument("--rec-weight", type=str, default="./weights/w600k_r50.onnx", help="Path to recognition model")
    parser.add_argument("--similarity-thresh", type=float, default=0.4, help="Similarity threshold between faces")
    parser.add_argument("--confidence-thresh", type=float, default=0.5, help="Confidence threshold for face detection")
    parser.add_argument("--faces-dir", type=str, default="./facesog", help="Path to faces stored directory")
    parser.add_argument("--source", type=str, default="http://192.168.0.6:5000/video_feed", help="Video file or video camera source")
    parser.add_argument("--max-num", type=int, default=0, help="Maximum number of face detections from a frame")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    return parser.parse_args()

def setup_logging(level: str) -> None:
    logging.basicConfig(level=getattr(logging, level.upper(), None), format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

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

def say_message(text: str) -> None:
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def process_residue():
    while True:
        # Esperar por la entrada del teclado (1 o 2)
        if keyboard.is_pressed('1'):
            # Enviar señal para cartón
            requests.get("http://192.168.0.6:6500/led/carton/on")
            say_message("Has introducido cartón. Por favor deposítalo en la sección indicada.")
            time.sleep(2)  # Pausa para que el LED se quede encendido
            requests.get("http://192.168.0.6:6500/led/carton/off")
        
        elif keyboard.is_pressed('2'):
            # Enviar señal para plástico
            requests.get("http://192.168.0.6:6500/led/plastico/on")
            say_message("Has introducido plástico. Por favor deposítalo en la sección indicada.")
            time.sleep(2)  # Pausa para que el LED se quede encendido
            requests.get("http://192.168.0.6:6500/led/plastico/off")

        time.sleep(0.1)  # Para no consumir demasiados recursos

def say_greeting(name: str) -> None:
    engine = pyttsx3.init()

    # Enviar solicitud para encender el LED en la Raspberry Pi

    greeting_part_1 = f"Hola {name}, soy el prototipo número 1."
    engine.say(greeting_part_1)
    engine.runAndWait()  # Esperar a que termine de decir esta parte
    
    time.sleep(2)  # Pausa de 2 segundos

    greeting_part_2 = "Te mostraré como puedes botar tu residuo, introduce tu residuo en la compuerta."
    engine.say(greeting_part_2)
    engine.runAndWait()

    # Enviar solicitud para apagar el LED en la Raspberry Pi

def threaded_greeting(name: str) -> None:
    threading.Thread(target=say_greeting, args=(name,)).start()

def frame_processor(frame: np.ndarray, detector: SCRFD, recognizer: ArcFace, targets: List[Tuple[np.ndarray, str]], colors: dict, params, last_recognized_name: str, greeting_active: dict) -> Tuple[np.ndarray, str]:
    bboxes, kpss = detector.detect(frame, params.max_num)

    current_recognized_name = "Unknown"
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
            color = (255, 0, 0)  # Rojo para desconocido
            label = "Desconocido"
        else:
            color = colors[best_match_name]
            label = f"{best_match_name} ({max_similarity:.2f})"
            current_recognized_name = best_match_name

        draw_bbox_info(frame, bbox, similarity=max_similarity, name=label, color=color)

    # Verificar si se debe reproducir el saludo
    if current_recognized_name != last_recognized_name:
        if current_recognized_name != "Unknown" and not greeting_active.get(current_recognized_name, False):
            threaded_greeting(current_recognized_name)
            greeting_active[current_recognized_name] = True  # Marcar como saludo activo

    return frame, current_recognized_name

def main(params):
    setup_logging(params.log_level)

    # Crear un hilo para manejar la entrada del teclado (simulación)
    threading.Thread(target=process_residue, daemon=True).start()

    detector = SCRFD(params.det_weight, input_size=(640, 640), conf_thres=params.confidence_thresh)
    recognizer = ArcFace(params.rec_weight)

    targets = build_targets(detector, recognizer, params)
    colors = {name: (random.randint(0, 256), random.randint(0, 256), random.randint(0, 256)) for _, name in targets}

    cap = cv2.VideoCapture(params.source)
    if not cap.isOpened():
        raise Exception("Could not open video or webcam")

    last_recognized_name = "Unknown"
    greeting_active = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame, current_recognized_name = frame_processor(frame, detector, recognizer, targets, colors, params, last_recognized_name, greeting_active)
        cv2.imshow("Frame", frame)

        # Reiniciar la sesión si se reconoce una persona diferente
        if current_recognized_name != last_recognized_name and current_recognized_name != "Unknown":
            if last_recognized_name != "Unknown":
                greeting_active[last_recognized_name] = False  # Restablecer saludo anterior
            last_recognized_name = current_recognized_name

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    args = parse_args()
    if args.source.isdigit():
        args.source = int(args.source)
    main(args)
