import cv2
import numpy as np
from tensorflow.keras.models import load_model
from ultralytics import YOLO


# Inisialisasi model
yolo_model = YOLO("yolov8n.pt")
klasifikasi_model = load_model("klasifikasi_sampah1.keras")
labels = ["Organik", "Anorganik"]

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Deteksi objek
    results = yolo_model(frame)[0]
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        # Klasifikasi
        img_resized = cv2.resize(crop, (224, 224))
        img_array = np.expand_dims(img_resized / 255.0, axis=0)
        pred = klasifikasi_model.predict(img_array)[0]
        class_id = int(pred > 0.5)
        label = labels[class_id]
        color = (0, 255, 0) if class_id == 0 else (0, 0, 255)

        # Tambah box dan label
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2
        )

    cv2.imshow("YOLO + Klasifikasi Sampah", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
