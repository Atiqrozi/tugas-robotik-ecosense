import cv2
import numpy as np
from tensorflow.keras.models import load_model
from ultralytics import YOLO

# Load model deteksi dan klasifikasi
yolo_model = YOLO("yolov8n.pt")
klasifikasi_model = load_model("klasifikasi_sampah1.keras")
class_labels = ["Organik", "Anorganik"]
suggestions = {
    "Organik": "Buang ke tempat sampah organik",
    "Anorganik": "Pisahkan untuk didaur ulang",
}


# Fungsi prediksi klasifikasi yang sudah diperbaiki
def predict_from_image(image):
    try:
        # Preprocessing - sama seperti di API yang berhasil
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (150, 150))
        image = image.astype(np.float32) / 255.0
        image = np.expand_dims(image, axis=0)

        # Prediksi
        predictions = klasifikasi_model.predict(image, verbose=0)[0]
        print(f"Prediction raw: {predictions}")  # Debugging

        # Interpretasi output - menggunakan threshold 0.5 seperti di API
        predicted_class_index = int(
            predictions[0] > 0.5
        )  # 0 for Organik, 1 for Anorganik
        predicted_label = class_labels[predicted_class_index]

        # Hitung confidence score seperti di API
        confidence = (
            float(predictions[0])
            if predicted_class_index == 1
            else float(1 - predictions[0])
        )
        confidence = confidence * 100

        suggestion = suggestions[predicted_label]

        return predicted_label, confidence, suggestion

    except Exception as e:
        print("Error klasifikasi:", str(e))
        return None, None, None


# Kamera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Deteksi objek dengan YOLO
    results = yolo_model(frame)[0]

    for box in results.boxes:
        # Skip jika bukan objek yang relevan (optional)
        # if box.cls not in [relevan_class_ids]: continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        crop = frame[y1:y2, x1:x2]

        # Skip jika crop kosong
        if crop.size == 0:
            continue

        # Klasifikasi
        label, confidence, suggestion = predict_from_image(crop)
        if label is None:
            continue

        # Warna box
        color = (0, 255, 0) if label == "Organik" else (0, 0, 255)

        # Tampilkan hasil
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

        # Tampilkan suggestion (optional)
        cv2.putText(
            frame,
            suggestion,
            (x1, y2 + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
        )

    cv2.imshow("Deteksi dan Klasifikasi Sampah", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
