

# import cv2
# import json
# import os
# import numpy as np
# from fast_alpr import ALPR

# ROI_FILE = "roi_polygon.json"

# # Initialize ALPR
# alpr = ALPR(
#     detector_model="yolo-v9-t-384-license-plate-end2end",
#     ocr_model="cct-xs-v2-global-model",
# )

# video_path = "input.mp4"
# cap = cv2.VideoCapture(video_path)

# # ---------------------------
# # 🖱️ Polygon Drawing
# # ---------------------------
# drawing_points = []
# drawing_done = False

# def mouse_callback(event, x, y, flags, param):
#     global drawing_points

#     if event == cv2.EVENT_LBUTTONDOWN:
#         drawing_points.append((x, y))


# def draw_polygon(frame):
#     temp = frame.copy()

#     # Draw points
#     for point in drawing_points:
#         cv2.circle(temp, point, 5, (0, 0, 255), -1)

#     # Draw lines
#     if len(drawing_points) > 1:
#         cv2.polylines(temp, [np.array(drawing_points)], False, (255, 0, 0), 2)

#     return temp


# def get_polygon_roi(frame):
#     global drawing_points
#     drawing_points = []

#     cv2.namedWindow("Draw Polygon ROI")
#     cv2.setMouseCallback("Draw Polygon ROI", mouse_callback)

#     print("Click to add points. Press 's' to save, 'c' to clear.")

#     while True:
#         temp = draw_polygon(frame)
#         cv2.imshow("Draw Polygon ROI", temp)

#         key = cv2.waitKey(1) & 0xFF

#         if key == ord('s') and len(drawing_points) >= 3:
#             break
#         elif key == ord('c'):
#             drawing_points = []

#     cv2.destroyWindow("Draw Polygon ROI")

#     # Save polygon
#     with open(ROI_FILE, "w") as f:
#         json.dump(drawing_points, f)

#     print("Saved polygon ROI:", drawing_points)
#     return np.array(drawing_points, dtype=np.int32)


# def load_polygon_roi():
#     with open(ROI_FILE, "r") as f:
#         points = json.load(f)
#     print("Loaded polygon ROI:", points)
#     return np.array(points, dtype=np.int32)


# # ---------------------------
# # 📦 Load or Draw ROI
# # ---------------------------
# ret, frame = cap.read()
# if not ret:
#     print("Failed to read video")
#     exit()

# if os.path.exists(ROI_FILE):
#     polygon = load_polygon_roi()
# else:
#     polygon = get_polygon_roi(frame)

# # Reset video
# cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# # ---------------------------
# # 🎥 Processing Loop
# # ---------------------------
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Create mask
#     mask = np.zeros(frame.shape[:2], dtype=np.uint8)
#     cv2.fillPoly(mask, [polygon], 255)

#     # Apply mask
#     roi_frame = cv2.bitwise_and(frame, frame, mask=mask)

#     # Run ALPR
#     results = alpr.predict(roi_frame)

#     # Draw polygon
#     cv2.polylines(frame, [polygon], True, (255, 0, 0), 2)

#     for result in results:
#         bbox = result.detection.bounding_box
#         x1, y1, x2, y2 = bbox.x1, bbox.y1, bbox.x2, bbox.y2
#         plate_text = result.ocr.text

#         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#         cv2.putText(frame, plate_text, (x1, y1 - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

#     cv2.imshow("ALPR Polygon ROI", frame)

#     key = cv2.waitKey(1) & 0xFF

#     # Redraw polygon
#     if key == ord('r'):
#         polygon = get_polygon_roi(frame)

#     # Quit
#     if key == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()


import cv2
import json
import os
import numpy as np
import csv
from fast_alpr import ALPR

ROI_FILE = "roi_polygon.json"
LOG_FILE = "alpr_log.csv"

# Initialize ALPR
alpr = ALPR(
    detector_model="yolo-v9-t-384-license-plate-end2end",
    ocr_model="cct-xs-v2-global-model",
)

video_path = "sample.mp4"
cap = cv2.VideoCapture(video_path)


fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define output video writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # codec
out = cv2.VideoWriter("output.mp4", fourcc, fps, (width, height))


# ---------------------------
# 📦 CSV Logger Setup
# ---------------------------
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "plate", "det_conf", "ocr_conf"])

# ---------------------------
# 🖱️ Polygon ROI Functions
# ---------------------------
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

    cv2.namedWindow("Draw Polygon ROI")
    cv2.setMouseCallback("Draw Polygon ROI", mouse_callback)

    print("Click to add points. Press 's' to save, 'c' to clear.")

    while True:
        temp = draw_polygon(frame)
        cv2.imshow("Draw Polygon ROI", temp)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s') and len(drawing_points) >= 3:
            break
        elif key == ord('c'):
            drawing_points = []

    cv2.destroyWindow("Draw Polygon ROI")

    with open(ROI_FILE, "w") as f:
        json.dump(drawing_points, f)

    return np.array(drawing_points, dtype=np.int32)

def load_polygon_roi():
    with open(ROI_FILE, "r") as f:
        points = json.load(f)
    return np.array(points, dtype=np.int32)

# ---------------------------
# 📦 Load or Draw ROI
# ---------------------------
ret, frame = cap.read()
if not ret:
    print("Failed to read video")
    exit()

if os.path.exists(ROI_FILE):
    polygon = load_polygon_roi()
else:
    polygon = get_polygon_roi(frame)

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# ---------------------------
# 🎥 Processing Loop
# ---------------------------
frame_number = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_number += 1

    # Create mask
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 255)
    roi_frame = cv2.bitwise_and(frame, frame, mask=mask)

    # Run ALPR
    results = alpr.predict(roi_frame)

    # Draw polygon
    cv2.polylines(frame, [polygon], True, (255, 0, 0), 2)

    for result in results:
        bbox = result.detection.bounding_box
        x1, y1, x2, y2 = bbox.x1, bbox.y1, bbox.x2, bbox.y2

        plate_text = result.ocr.text
        det_conf = result.detection.confidence
        ocr_conf = max(result.ocr.confidence) if result.ocr.confidence else 0

        # Draw detection
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{plate_text} ({det_conf:.2f})"
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # 📝 Log to CSV
        with open(LOG_FILE, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([frame_number, plate_text, det_conf, ocr_conf])

    # 📊 Draw frame number on screen
    cv2.putText(frame, f"Frame: {frame_number}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("ALPR Polygon ROI", frame)

    # ✅ Save frame to output video
    out.write(frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('r'):
        polygon = get_polygon_roi(frame)

    if key == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()