import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import cv2
import random
import warnings
import argparse
import logging
import numpy as np
import pyttsx3
import time
import threading
import requests
from typing import List, Tuple
from tensorflow.keras.preprocessing import image  # type: ignore
import keras.applications.xception as xception  # type: ignore
import tensorflow as tf
from models import SCRFD, ArcFace
from utils.helpers import compute_similarity, draw_bbox_info

warnings.filterwarnings("ignore")

# Constants
IMAGE_WIDTH, IMAGE_HEIGHT = 320, 320
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)

# Categories
categories = {0: 'battery', 1: 'biological', 2: 'brown-glass', 3: 'cardboard',
              4: 'clothes', 5: 'green-glass', 6: 'metal', 7: 'paper',
              8: 'plastic', 9: 'shoes', 10: 'trash', 11: 'white-glass'}

# Define Xception preprocessing
def xception_preprocessing(img):
    return xception.preprocess_input(img)

# Load the classification model
model = tf.keras.models.load_model('model_res.h5', custom_objects={'xception_preprocessing': xception_preprocessing})

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=IMAGE_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_image(img_path, model):
    processed_img = preprocess_image(img_path)
    predictions = model.predict(processed_img)
    return predictions

# Modifica la sección donde clasificas el residuo para incluir el mensaje de voz
def classify_waste(img_path, model):
    predictions = predict_image(img_path, model)
    predicted_class = np.argmax(predictions, axis=1)[0]
    predicted_category = categories[predicted_class]
    say_classification(predicted_category)  # Llamar a la función de voz
    return predicted_category, predictions[0]

def say_classification(category: str) -> None:
    engine = pyttsx3.init()
    # Mensajes predefinidos para cada tipo de residuo
    disposal_instructions = {
        "battery": "Por favor, deseche las baterías en el contenedor de residuos peligrosos.",
        "biological": "Deseche los residuos biológicos en el tacho designado para desechos biológicos.",
        "brown-glass": "El vidrio marrón debe ir en el tacho de reciclaje de vidrio.",
        "cardboard": "Por favor, coloque el cartón en el tacho de reciclaje de papel y cartón.",
        "clothes": "La ropa usada debe ir en el contenedor de donaciones o reciclaje de textiles.",
        "green-glass": "El vidrio verde debe ir en el tacho de reciclaje de vidrio.",
        "metal": "Por favor, coloque los metales en el tacho de reciclaje de metales.",
        "paper": "El papel debe ir en el contenedor de reciclaje de papel.",
        "plastic": "Por favor, coloque el plástico en el tacho de reciclaje de plásticos.",
        "shoes": "Los zapatos usados deben ir en el contenedor de donaciones o reciclaje de textiles.",
        "trash": "Esto es basura general y debe ir en el contenedor de residuos no reciclables.",
        "white-glass": "El vidrio blanco debe ir en el tacho de reciclaje de vidrio."
    }

    # Generar el mensaje según la categoría reconocida
    message = disposal_instructions.get(category, "Por favor, deseche este residuo correctamente.")
    engine.say(message)
    engine.runAndWait()

def parse_args():
    parser = argparse.ArgumentParser(description="Face Detection, Recognition, and Waste Classification")
    parser.add_argument("--det-weight", type=str, default="./weights/det_10g.onnx", help="Path to detection model")
    parser.add_argument("--rec-weight", type=str, default="./weights/w600k_r50.onnx", help="Path to recognition model")
    parser.add_argument("--similarity-thresh", type=float, default=0.4, help="Similarity threshold between faces")
    parser.add_argument("--confidence-thresh", type=float, default=0.5, help="Confidence threshold for face detection")
    parser.add_argument("--faces-dir", type=str, default="./facesog", help="Path to faces stored directory")
    parser.add_argument("--source", type=str, default="http://192.168.1.8:5000/video_feed", help="Video source")
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

def say_greeting(name: str) -> None:
    engine = pyttsx3.init()
    greeting_part_1 = f"Hola {name}, soy el prototipo número 1."
    engine.say(greeting_part_1)
    engine.runAndWait()
    time.sleep(2)
    greeting_part_2 = "Te mostraré como puedes botar tu residuo, introduce tu residuo en la compuerta."
    engine.say(greeting_part_2)
    engine.runAndWait()

def threaded_greeting(name: str) -> None:
    threading.Thread(target=say_greeting, args=(name,)).start()

def frame_processor(frame, detector, recognizer, targets, colors, params, last_recognized_name, greeting_active) -> Tuple[np.ndarray, str]:
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
            color = (255, 0, 0)
            label = "Desconocido"
        else:
            color = colors[best_match_name]
            label = f"{best_match_name} ({max_similarity:.2f})"
            current_recognized_name = best_match_name
        draw_bbox_info(frame, bbox, similarity=max_similarity, name=label, color=color)
    if current_recognized_name != last_recognized_name:
        if current_recognized_name != "Unknown" and not greeting_active.get(current_recognized_name, False):
            threaded_greeting(current_recognized_name)
            greeting_active[current_recognized_name] = True
    return frame, current_recognized_name

def main(params):
    setup_logging(params.log_level)
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

        # Show frame and classify waste if key '2' is pressed
        frame, current_recognized_name = frame_processor(frame, detector, recognizer, targets, colors, params, last_recognized_name, greeting_active)
        cv2.imshow("Frame", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('2'):
            response = requests.get("http://192.168.1.8:5000/take_photo")
            if response.status_code == 200:
                desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
                photo_filename = os.path.join(desktop_path, f'foto_{int(time.time())}.jpg')
                with open(photo_filename, 'wb') as f:
                    f.write(response.content)
                predicted_category, probabilities = classify_waste(photo_filename, model)
                print(f'Classified as: {predicted_category}')
            else:
                print("Error taking photo")
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    args = parse_args()
    if args.source.isdigit():
        args.source = int(args.source)
    main(args)
