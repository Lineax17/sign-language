import cv2
import numpy as np
import tensorflow as tf

# Pfad zum TFLite-Modell
TFLITE_PATH = 'models/sign_language_mobilenetv2.tflite'
IMG_SIZE = 224  # Muss zum Training passen

# TFLite-Interpreter laden
interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Kamera öffnen
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Bild zuschneiden und skalieren
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    # Inference
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    pred = np.argmax(output)

    # Ergebnis anzeigen (hier nur Index, passe für deine Labels an)
    cv2.putText(frame, f'Klasse: {pred}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow('TFLite Echtzeit', frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC zum Beenden
        break

cap.release()
cv2.destroyAllWindows()