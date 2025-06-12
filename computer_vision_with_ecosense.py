import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO

# Load YOLOv8
yolo_model = YOLO("yolov8n.pt")

# Load TFLite Interpreter
interpreter = tf.lite.Interpreter(model_path="model_with_metadata.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Label dan saran
class_labels = ["Organik", "Anorganik"]
suggestions = {
    "Organik": "Buang ke tempat sampah organik",
    "Anorganik": "Pisahkan untuk didaur ulang",
}


def predict_from_image_tflite(image):
    try:
        # Preprocessing
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (150, 150))
        image = image.astype(np.float32) / 255.0
        image = np.expand_dims(image, axis=0)

        # Inferensi
        interpreter.set_tensor(input_details[0]["index"], image)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]["index"])[0]

        # Interpretasi hasil (sigmoid satu neuron)
        predicted_index = int(predictions[0] < 0.5)
        predicted_label = class_labels[predicted_index]
        confidence = (
            float(1 - predictions[0]) * 100
            if predicted_index == 1
            else float(predictions[0]) * 100
        )
        suggestion = suggestions[predicted_label]

        return predicted_label, confidence, suggestion
    except Exception as e:
        print("Error klasifikasi:", str(e))
        return None, None, None


# Buka kamera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Deteksi sampah dengan YOLO
    results = yolo_model(frame)[0]

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        # Klasifikasi dengan TFLite
        label, confidence, suggestion = predict_from_image_tflite(crop)
        if label is None:
            continue

        # Kotak dan label
        color = (0, 255, 0) if label == "Organik" else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            f"{label} ({confidence:.1f}%)",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
        )

    cv2.imshow("Deteksi dan Klasifikasi Sampah (TFLite)", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
