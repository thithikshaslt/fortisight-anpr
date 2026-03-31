import cv2
import json
import os
import numpy as np
import csv
import time
import psutil
from fast_alpr import ALPR

# =========================
# 📊 Benchmark Class
# =========================
class Benchmark:
    def __init__(self, name="Pipeline"):
        self.name = name
        self.start_time = None
        self.frame_count = 0
        self.total_frame_time = 0
        self.cpu_readings = []
        self.mem_readings = []
        self.alpr_calls = 0
        self.process = psutil.Process(os.getpid())

    def start(self):
        self.start_time = time.time()

    def start_frame(self):
        self.frame_start = time.time()

    def end_frame(self):
        frame_time = time.time() - self.frame_start
        self.total_frame_time += frame_time
        self.frame_count += 1

        self.cpu_readings.append(psutil.cpu_percent())
        mem = self.process.memory_info().rss / (1024 * 1024)
        self.mem_readings.append(mem)

    def log_alpr_call(self):
        self.alpr_calls += 1

    def results(self):
        total_time = time.time() - self.start_time
        avg_fps = self.frame_count / total_time if total_time > 0 else 0
        avg_frame_time = self.total_frame_time / self.frame_count if self.frame_count else 0
        avg_cpu = sum(self.cpu_readings) / len(self.cpu_readings) if self.cpu_readings else 0
        avg_mem = sum(self.mem_readings) / len(self.mem_readings) if self.mem_readings else 0

        print("\n==============================")
        print(f"📊 Benchmark: {self.name}")
        print("==============================")
        print(f"Total Frames     : {self.frame_count}")
        print(f"Total Time (s)   : {total_time:.2f}")
        print(f"Average FPS      : {avg_fps:.2f}")
        print(f"Avg Frame Time   : {avg_frame_time:.4f} s")
        print(f"Avg CPU Usage    : {avg_cpu:.2f} %")
        print(f"Avg Memory Usage : {avg_mem:.2f} MB")
        print(f"ALPR Calls       : {self.alpr_calls}")
        print("==============================\n")


# =========================
# ⚙️ Config
# =========================
ROI_FILE = "roi_polygon.json"
LOG_FILE = "alpr_log.csv"
VIDEO_PATH = "sample.mp4"

# =========================
# 🚀 Initialize
# =========================
alpr = ALPR(
    detector_model="yolo-v9-t-384-license-plate-end2end",
    ocr_model="cct-xs-v2-global-model",
)

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("❌ Failed to open video")
    exit()

fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter("output.mp4",
                      cv2.VideoWriter_fourcc(*"mp4v"),
                      fps,
                      (width, height))

# =========================
# 📦 CSV Setup
# =========================
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "plate", "det_conf", "ocr_conf"])

# =========================
# 🖱️ ROI Functions
# =========================
drawing_points = []

def mouse_callback(event, x, y, flags, param):
    global drawing_points
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing_points.append((x, y))

def draw_polygon(frame):
    temp = frame.copy()
    for point in drawing_points:
        cv2.circle(temp, point, 5, (0, 0, 255), -1)
    if len(drawing_points) > 1:
        cv2.polylines(temp, [np.array(drawing_points)], False, (255, 0, 0), 2)
    return temp

def get_polygon_roi(frame):
    global drawing_points
    drawing_points = []

    cv2.namedWindow("Draw ROI")
    cv2.setMouseCallback("Draw ROI", mouse_callback)

    print("Click points → Press 's' to save")

    while True:
        temp = draw_polygon(frame)
        cv2.imshow("Draw ROI", temp)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s') and len(drawing_points) >= 3:
            break
        elif key == ord('c'):
            drawing_points = []

    cv2.destroyWindow("Draw ROI")

    with open(ROI_FILE, "w") as f:
        json.dump(drawing_points, f)

    return np.array(drawing_points, dtype=np.int32)

def load_polygon_roi():
    with open(ROI_FILE, "r") as f:
        return np.array(json.load(f), dtype=np.int32)

# =========================
# 📦 Load ROI
# =========================
ret, frame = cap.read()
if not ret:
    print("❌ Failed to read first frame")
    exit()

polygon = load_polygon_roi() if os.path.exists(ROI_FILE) else get_polygon_roi(frame)
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# =========================
# 📊 Benchmark Start
# =========================
bench = Benchmark("ROI Only ALPR")
bench.start()

# =========================
# 🎥 Main Loop
# =========================
frame_number = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_number += 1
    bench.start_frame()

    # ROI mask
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 255)
    roi_frame = cv2.bitwise_and(frame, frame, mask=mask)

    # ALPR
    results = alpr.predict(roi_frame)
    bench.log_alpr_call()

    # Draw ROI
    cv2.polylines(frame, [polygon], True, (255, 0, 0), 2)

    for result in results:
        bbox = result.detection.bounding_box
        x1, y1, x2, y2 = bbox.x1, bbox.y1, bbox.x2, bbox.y2

        plate = result.ocr.text
        det_conf = result.detection.confidence
        ocr_conf = max(result.ocr.confidence) if result.ocr.confidence else 0

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, plate, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # CSV log
        with open(LOG_FILE, mode='a', newline='') as f:
            csv.writer(f).writerow([frame_number, plate, det_conf, ocr_conf])

    cv2.putText(frame, f"Frame: {frame_number}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("ALPR ROI", frame)
    out.write(frame)

    bench.end_frame()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# =========================
# 🧹 Cleanup + Results
# =========================
cap.release()
out.release()
cv2.destroyAllWindows()

bench.results()