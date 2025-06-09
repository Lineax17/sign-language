import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf

# === Konfiguration ===
TFLITE_PATH = "models/alexnet_tuned.tflite"
CLASS_FILE = "data/class_name.txt"
IMG_SIZE = 224

# === Klassen laden ===
with open(CLASS_FILE, 'r') as f:
    CLASS_NAMES = [line.strip() for line in f]

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
    resized = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    normalized = resized.astype(np.float32) / 255.0
    return np.expand_dims(normalized, axis=0)

# Aktuelle Vorhersage (wird nur angezeigt wenn Hand erkannt)
current_prediction = ""

while True:
    ret, frame = cap.read()
    if not ret:
        break

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(image_rgb)

    hand_detected = False

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            hand_detected = True  # Flag setzen

            h, w, _ = frame.shape
            x_min, y_min = w, h
            x_max = y_max = 0

            for lm in hand_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x)
                y_max = max(y_max, y)

            generous_margin = 150
            x_min = max(0, x_min - generous_margin)
            y_min = max(0, y_min - generous_margin)
            x_max = min(w, x_max + generous_margin)
            y_max = min(h, y_max + generous_margin)

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

            # === Modellvorhersage ===
            input_tensor = preprocess_image(roi)
            interpreter.set_tensor(input_details[0]['index'], input_tensor)
            interpreter.invoke()
            output = interpreter.get_tensor(output_details[0]['index'])

            pred_idx = np.argmax(output)
            confidence = output[0][pred_idx]
            current_prediction = f'{CLASS_NAMES[pred_idx]} ({confidence:.2f})'

    else:
        # Wenn keine Hand erkannt: Vorhersage ausblenden
        current_prediction = ""

    # === Anzeige der aktuellen Vorhersage ===
    if current_prediction:
        cv2.putText(frame, current_prediction, (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (0, 255, 0), 3)

    cv2.imshow('ASL Live-Vorhersage (ESC zum Beenden)', frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
