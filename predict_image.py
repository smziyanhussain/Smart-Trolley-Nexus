import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import os
import time

model = load_model('product_recognizer_smallset.h5')

dataset_path = 'dataset/'
class_names = sorted(os.listdir(dataset_path))

def predict_from_frame(frame):
    img = cv2.resize(frame, (224, 224))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0) / 255.0
    predictions = model.predict(x)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions) * 100
    return predicted_class, confidence

cap = cv2.VideoCapture(0)

threshold = 75
cooldown_seconds = 3
last_detection_time = time.time()
last_detected = None
cart = []

print("Show product to camera. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    predicted_class, confidence = predict_from_frame(frame)

    current_time = time.time()

    if confidence >= threshold:
        if (predicted_class != last_detected) or (current_time - last_detection_time > cooldown_seconds):
            if predicted_class not in cart:
                cart.append(predicted_class)
                print(f"Added to cart: {predicted_class}")
            last_detected = predicted_class
            last_detection_time = current_time

    if confidence >= threshold:
        label = f"{predicted_class}: {confidence:.2f}%"
        cv2.putText(frame, label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    y_offset = 60
    cv2.putText(frame, "Cart:", (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    for item in cart:
        y_offset += 30
        cv2.putText(frame, f"- {item}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow('Product Detector', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("\nFinal Cart:")
for item in cart:
    print(f" - {item}")
