import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp

# === Klassen laden ===
with open('data/class_name.txt', 'r') as f:
    CLASS_NAMES = [line.strip() for line in f]

# === TFLite Modell vorbereiten ===
TFLITE_PATH = 'models/sign_language_mobilenetv2.tflite'
IMG_SIZE = 224

interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# === MediaPipe Hand-Modul ===
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

# === Kamera starten ===
cap = cv2.VideoCapture(0)

def preprocess_hand_roi(hand_roi):
    resized = cv2.resize(hand_roi, (IMG_SIZE, IMG_SIZE))
    normalized = resized.astype(np.float32) / 255.0
    input_tensor = np.expand_dims(normalized, axis=0)
    return input_tensor

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

            # Bounding Box berechnen
            for lm in hand_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x)
                y_max = max(y_max, y)

            # Sicherheitsrand
            margin = 20
            x_min = max(0, x_min - margin)
            y_min = max(0, y_min - margin)
            x_max = min(w, x_max + margin)
            y_max = min(h, y_max + margin)

            # ROI ausschneiden
            hand_roi = frame[y_min:y_max, x_min:x_max]

            if hand_roi.size == 0:
                continue  # Verhindert Fehler bei leeren Ausschnitten

            input_tensor = preprocess_hand_roi(hand_roi)

            # Modellvorhersage
            interpreter.set_tensor(input_details[0]['index'], input_tensor)
            interpreter.invoke()
            output = interpreter.get_tensor(output_details[0]['index'])
            pred_idx = np.argmax(output)
            confidence = output[0][pred_idx]
            label = f'{CLASS_NAMES[pred_idx]} ({confidence:.2f})'

            # Bounding Box und Label anzeigen
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow('ASL mit TFLite + MediaPipe', frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC zum Beenden
        break

cap.release()
cv2.destroyAllWindows()
