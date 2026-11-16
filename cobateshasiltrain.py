import cv2
import time
from collections import Counter
from ultralytics import YOLO

# 1. Muat model Anda
MODEL_PATH = r"C:\Users\User\oamp2026\oamp-2026\MODEL\onnx\baseline-52-4rd.onnx" # Ganti dengan path model Anda
model = YOLO(MODEL_PATH)

# 2. Buka webcam
cap = cv2.VideoCapture(4)

if not cap.isOpened():
    print("Error: Tidak dapat mengakses kamera.")
    exit()

# Variabel untuk menghitung FPS
prev_time = time.time()
fps = 0

# 3. Loop untuk memproses setiap frame
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Tidak dapat membaca frame.")
        break

    # Hitung waktu mulai inference
    inference_start = time.time()

    # 4. Jalankan inferensi
    # imgsz=640 untuk kecepatan real-time (bisa diubah ke 1280 jika GPU kuat)
    results = model.predict(frame, imgsz=640, conf=0.5, verbose=False)

    # Hitung waktu selesai inference
    inference_end = time.time()
    latency_ms = (inference_end - inference_start) * 1000

    # Hitung FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    # 5. Ekstraksi Data Metrik
    boxes = results[0].boxes
    total_objects = len(boxes)
    
    # Hitung distribusi kelas di frame ini
    class_counts = Counter()
    for box in boxes:
        cls = int(box.cls[0].item())
        class_counts[model.names[cls]] += 1

    # 6. Gambar kotak pembatas dan label (Confidence Rate)
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = box.conf[0].item()
        cls = int(box.cls[0].item())
        label = f"{model.names[cls]} {conf:.2f}"
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 7. Buat Dashboard Overlay di pojok kiri atas
    # Latar belakang hitam transparan
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (300, 150), (0, 0, 0), -1)
    alpha = 0.6
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    # Tulis metrik ke layar
    y_offset = 30
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    y_offset += 25
    cv2.putText(frame, f"Latency: {latency_ms:.1f} ms", (20, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    y_offset += 25
    cv2.putText(frame, f"Objects: {total_objects}", (20, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    # Tampilkan distribusi kelas (maksimal 3 baris agar tidak kepanjangan)
    y_offset += 25
    for cls_name, count in class_counts.most_common(3):
        cv2.putText(frame, f"  {cls_name}: {count}", (20, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y_offset += 20

    # 8. Tampilkan frame
    cv2.imshow("YOLO Inference Dashboard", frame)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 9. Bersihkan
cap.release()
cv2.destroyAllWindows()