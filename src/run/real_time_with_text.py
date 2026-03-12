import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import time

# === Konfiguration ===
TFLITE_PATH = "models/asl_mobilenetv2_unfreezed.tflite"
IMG_SIZE = 224
STABLE_DURATION = 1.0  # Sekunden bis Zeichen gültig ist

CLASS_NAMES = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L",
    "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "delete", "space"
]


# === TFLite Modell vorbereiten ===
interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# === MediaPipe vorbereiten ===
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

# === Kamera starten ===
cap = cv2.VideoCapture(0)
print("[INFO] Starte Live-Vorhersage... ESC zum Beenden")

def preprocess_image(image):
    return np.expand_dims(image.astype(np.float32), axis=0)

# Zustände
current_prediction = ""
text_output = ""
last_confirmed_label = None
label_start_time = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(image_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            h, w, _ = frame.shape
            x_min, y_min = w, h
            x_max = y_max = 0

            for lm in hand_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x)
                y_max = max(y_max, y)

            # Großzügiges Quadrat um die Hand
            margin = 100
            x_min = max(0, x_min - margin)
            y_min = max(0, y_min - margin)
            x_max = min(w, x_max + margin)
            y_max = min(h, y_max + margin)

            roi_w = x_max - x_min
            roi_h = y_max - y_min
            roi_size = max(roi_w, roi_h)

            cx = x_min + roi_w // 2
            cy = y_min + roi_h // 2
            half_size = roi_size // 2

            q_x_min = max(0, cx - half_size)
            q_y_min = max(0, cy - half_size)
            q_x_max = min(w, cx + half_size)
            q_y_max = min(h, cy + half_size)

            roi = frame[q_y_min:q_y_max, q_x_min:q_x_max]
            if roi.size == 0 or roi.shape[0] < 10 or roi.shape[1] < 10:
                continue

            roi_resized = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
            roi_flipped = cv2.flip(roi_resized, 1)

            # Vorhersage
            input_tensor = preprocess_image(roi_flipped)
            interpreter.set_tensor(input_details[0]['index'], input_tensor)
            interpreter.invoke()
            output = interpreter.get_tensor(output_details[0]['index'])

            pred_idx = np.argmax(output)
            confidence = output[0][pred_idx]
            predicted_label = CLASS_NAMES[pred_idx]
            current_prediction = f'{predicted_label} ({confidence:.2f})'

            # Stabilitätsprüfung
            current_time = time.time()
            if predicted_label == last_confirmed_label:
                if current_time - label_start_time >= STABLE_DURATION:
                    if predicted_label == "space":
                        text_output += " "
                    elif predicted_label == "delete":
                        text_output = text_output[:-1]
                    else:
                        text_output += predicted_label
                    last_confirmed_label = None
                    label_start_time = None
            else:
                last_confirmed_label = predicted_label
                label_start_time = current_time
    else:
        current_prediction = ""
        last_confirmed_label = None
        label_start_time = None

    # === Anzeige: Vorhersage oben + Text unten (mit "_" für space) ===
    if current_prediction:
        cv2.putText(frame, current_prediction, (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (0, 255, 0), 3)

    display_text = text_output.replace(" ", "_")
    cv2.putText(frame, display_text, (10, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    cv2.imshow('ASL Live-Vorhersage (ESC zum Beenden)', frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
