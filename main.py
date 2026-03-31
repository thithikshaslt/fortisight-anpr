import cv2
import json
import os
import numpy as np
import csv
import logging
from collections import Counter
from fast_alpr import ALPR

ROI_FILE = "roi_polygon.json"
LOG_FILE = "alpr_log.csv"
DEBUG_LOG_FILE = "debug.log"

# ---------------------------
# Logging Setup
# ---------------------------
logging.basicConfig(
    filename=DEBUG_LOG_FILE,
    filemode="w",   # overwrite every run; use "a" if you want append
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

logger.info("===== ALPR SCRIPT STARTED =====")

# ---------------------------
# Initialize ALPR
# ---------------------------
alpr = ALPR(
    detector_model="yolo-v9-t-384-license-plate-end2end",
    ocr_model="cct-xs-v2-global-model",
)

video_path = "sample.mp4"
cap = cv2.VideoCapture(video_path)

fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

logger.info(f"Video loaded: {video_path}")
logger.info(f"Video properties -> fps={fps}, width={width}, height={height}")

# Output video writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("output.mp4", fourcc, fps, (width, height))
logger.info("Output video writer initialized: output.mp4")

# ---------------------------
# CSV Logger Setup
# ---------------------------
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "track_id",
            "start_frame",
            "end_frame",
            "final_plate",
            "num_votes"
        ])
    logger.info(f"CSV file created with header: {LOG_FILE}")
else:
    logger.info(f"CSV file already exists: {LOG_FILE}")

# ---------------------------
# Polygon ROI Functions
# ---------------------------
drawing_points = []

def mouse_callback(event, x, y, flags, param):
    global drawing_points
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing_points.append((x, y))
        logger.debug(f"ROI point added: {(x, y)}")

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
    logger.info("ROI drawing started")

    while True:
        temp = draw_polygon(frame)
        cv2.imshow("Draw Polygon ROI", temp)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("s") and len(drawing_points) >= 3:
            logger.info(f"ROI saved with points: {drawing_points}")
            break
        elif key == ord("c"):
            drawing_points = []
            logger.info("ROI points cleared")

    cv2.destroyWindow("Draw Polygon ROI")

    with open(ROI_FILE, "w") as f:
        json.dump(drawing_points, f)

    logger.info(f"ROI written to file: {ROI_FILE}")
    return np.array(drawing_points, dtype=np.int32)

def load_polygon_roi():
    with open(ROI_FILE, "r") as f:
        points = json.load(f)
    logger.info(f"ROI loaded from file: {ROI_FILE} | points={points}")
    return np.array(points, dtype=np.int32)

# ---------------------------
# Tracking + Voting Helpers
# ---------------------------
def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter_area = inter_w * inter_h

    areaA = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    areaB = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])

    union = areaA + areaB - inter_area
    if union == 0:
        return 0.0

    return inter_area / union

def character_majority_vote(texts, track_id=None):
    """
    Returns:
        final_text, debug_info
    """
    cleaned = [t.strip().upper() for t in texts if t and t.strip()]

    debug_info = {
        "raw_texts": texts,
        "cleaned_texts": cleaned,
        "selected_length": None,
        "filtered_texts": [],
        "position_votes": []
    }

    logger.debug(f"[Track {track_id}] Starting character_majority_vote")
    logger.debug(f"[Track {track_id}] Raw OCR texts: {texts}")
    logger.debug(f"[Track {track_id}] Cleaned OCR texts: {cleaned}")

    if not cleaned:
        logger.debug(f"[Track {track_id}] No valid cleaned texts found")
        return "", debug_info

    length_counts = Counter(len(t) for t in cleaned)
    target_length = length_counts.most_common(1)[0][0]
    filtered = [t for t in cleaned if len(t) == target_length]

    debug_info["selected_length"] = target_length
    debug_info["filtered_texts"] = filtered

    logger.debug(f"[Track {track_id}] Length counts: {dict(length_counts)}")
    logger.debug(f"[Track {track_id}] Selected target length: {target_length}")
    logger.debug(f"[Track {track_id}] Filtered texts used for voting: {filtered}")

    final_chars = []

    for i in range(target_length):
        chars_at_pos = [t[i] for t in filtered]
        counts = Counter(chars_at_pos)
        selected_char = counts.most_common(1)[0][0]

        debug_info["position_votes"].append({
            "position": i,
            "chars": chars_at_pos,
            "counts": dict(counts),
            "selected": selected_char
        })

        logger.debug(
            f"[Track {track_id}] Position {i}: chars={chars_at_pos} | "
            f"counts={dict(counts)} | selected='{selected_char}'"
        )

        final_chars.append(selected_char)

    final_text = "".join(final_chars)
    logger.debug(f"[Track {track_id}] Final voted plate: {final_text}")

    return final_text, debug_info

def finalize_track(track_id, track_data):
    logger.info(f"[Track {track_id}] Finalizing track")
    logger.info(
        f"[Track {track_id}] Track summary before finalize -> "
        f"start_frame={track_data['start_frame']}, "
        f"last_seen={track_data['last_seen']}, "
        f"texts={track_data['texts']}"
    )

    final_plate, debug_info = character_majority_vote(track_data["texts"], track_id=track_id)

    logger.info(f"[Track {track_id}] Voting debug info: {debug_info}")

    if final_plate:
        with open(LOG_FILE, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                track_id,
                track_data["start_frame"],
                track_data["last_seen"],
                final_plate,
                len(track_data["texts"])
            ])

        logger.info(
            f"[Track {track_id}] FINAL CSV WRITE -> "
            f"start_frame={track_data['start_frame']}, "
            f"end_frame={track_data['last_seen']}, "
            f"final_plate={final_plate}, "
            f"num_votes={len(track_data['texts'])}"
        )

        print(
            f"[FINALIZED] Track {track_id} | "
            f"Frames {track_data['start_frame']}->{track_data['last_seen']} | "
            f"Plate: {final_plate} | Votes: {len(track_data['texts'])}"
        )
    else:
        logger.warning(f"[Track {track_id}] Final plate is empty; not writing to CSV")

# ---------------------------
# Load or Draw ROI
# ---------------------------
ret, frame = cap.read()
if not ret:
    logger.error("Failed to read first frame from video")
    print("Failed to read video")
    exit()

if os.path.exists(ROI_FILE):
    polygon = load_polygon_roi()
else:
    polygon = get_polygon_roi(frame)

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
logger.info("Video reset to frame 0 after ROI selection")

# ---------------------------
# Tracking Config
# ---------------------------
tracks = {}
next_track_id = 1

IOU_THRESHOLD = 0.2
MAX_MISSED_FRAMES = 20
MIN_FRAMES_TO_FINALIZE = 10

logger.info(
    f"Tracking config -> IOU_THRESHOLD={IOU_THRESHOLD}, "
    f"MAX_MISSED_FRAMES={MAX_MISSED_FRAMES}, "
    f"MIN_FRAMES_TO_FINALIZE={MIN_FRAMES_TO_FINALIZE}"
)

# ---------------------------
# Processing Loop
# ---------------------------
frame_number = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        logger.info("End of video reached or failed to read frame")
        break

    frame_number += 1
    logger.debug(f"===== FRAME {frame_number} START =====")

    # Create ROI mask
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 255)
    roi_frame = cv2.bitwise_and(frame, frame, mask=mask)

    # Run ALPR
    results = alpr.predict(roi_frame)
    logger.debug(f"Frame {frame_number}: Number of ALPR results = {len(results)}")

    # Draw polygon
    cv2.polylines(frame, [polygon], True, (255, 0, 0), 2)

    matched_track_ids = set()

    for det_idx, result in enumerate(results):
        bbox = result.detection.bounding_box
        x1, y1, x2, y2 = int(bbox.x1), int(bbox.y1), int(bbox.x2), int(bbox.y2)

        plate_text = result.ocr.text.strip().upper() if result.ocr.text else ""
        det_conf = result.detection.confidence
        ocr_conf = max(result.ocr.confidence) if result.ocr.confidence else 0

        current_box = (x1, y1, x2, y2)

        logger.debug(
            f"Frame {frame_number} | Detection {det_idx} -> "
            f"bbox={current_box}, plate_text='{plate_text}', "
            f"det_conf={det_conf:.4f}, ocr_conf={ocr_conf:.4f}"
        )

        # Match detection to existing track using IoU
        best_track_id = None
        best_iou = 0.0

        for track_id, track_data in tracks.items():
            iou = compute_iou(current_box, track_data["bbox"])
            logger.debug(
                f"Frame {frame_number} | Detection {det_idx} vs Track {track_id} -> "
                f"existing_bbox={track_data['bbox']} | IoU={iou:.4f}"
            )
            
            # 1. IoU matching
            if iou > best_iou and iou >= IOU_THRESHOLD:
                best_iou = iou
                best_track_id = track_id
            
            # 2. Plate-based re-identification (if plate is readable)
            if plate_text:
                current_vote_text, _ = character_majority_vote(track_data["texts"], track_id=track_id)
                if plate_text == current_vote_text:
                    logger.info(f"Frame {frame_number} | Plate Match: '{plate_text}' re-identifies Track {track_id}")
                    best_track_id = track_id
                    break # Prioritize exact plate match

        if best_track_id is None:
            best_track_id = next_track_id
            tracks[best_track_id] = {
                "bbox": current_box,
                "texts": [],
                "last_seen": frame_number,
                "start_frame": frame_number,
            }
            next_track_id += 1

            logger.info(
                f"Frame {frame_number} | Detection {det_idx} -> "
                f"Created NEW track {best_track_id} with bbox={current_box}"
            )
        else:
            logger.info(
                f"Frame {frame_number} | Detection {det_idx} -> "
                f"Matched to existing track {best_track_id} with best_iou={best_iou:.4f}"
            )

        # Update track
        old_bbox = tracks[best_track_id]["bbox"]
        tracks[best_track_id]["bbox"] = current_box
        tracks[best_track_id]["last_seen"] = frame_number

        logger.debug(
            f"Frame {frame_number} | Track {best_track_id} updated -> "
            f"old_bbox={old_bbox}, new_bbox={current_box}, last_seen={frame_number}"
        )

        if plate_text:
            tracks[best_track_id]["texts"].append(plate_text)
            logger.info(
                f"Frame {frame_number} | Track {best_track_id} appended OCR text='{plate_text}' | "
                f"all_texts={tracks[best_track_id]['texts']}"
            )
        else:
            logger.warning(
                f"Frame {frame_number} | Track {best_track_id} got empty OCR text; not appended"
            )

        matched_track_ids.add(best_track_id)

        # Live display
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"ID {best_track_id} | {plate_text} | D:{det_conf:.2f} O:{ocr_conf:.2f}"
        cv2.putText(
            frame,
            label,
            (x1, max(30, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

        current_vote, _ = character_majority_vote(tracks[best_track_id]["texts"], track_id=best_track_id)
        logger.debug(
            f"Frame {frame_number} | Track {best_track_id} current live voted plate='{current_vote}'"
        )

        cv2.putText(
            frame,
            f"Vote: {current_vote}",
            (x1, min(height - 10, y2 + 25)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )

    # Finalize stale tracks
    tracks_to_delete = []
    for track_id, track_data in tracks.items():
        missed = frame_number - track_data["last_seen"]

        logger.debug(
            f"Frame {frame_number} | Track {track_id} status -> "
            f"last_seen={track_data['last_seen']}, missed={missed}, "
            f"text_count={len(track_data['texts'])}"
        )

        if missed > MAX_MISSED_FRAMES:
            logger.info(
                f"Frame {frame_number} | Track {track_id} exceeded MAX_MISSED_FRAMES "
                f"with missed={missed}"
            )

            if len(track_data["texts"]) >= MIN_FRAMES_TO_FINALIZE:
                logger.info(
                    f"Frame {frame_number} | Track {track_id} eligible for finalization "
                    f"(text_count={len(track_data['texts'])})"
                )
                finalize_track(track_id, track_data)
            else:
                logger.warning(
                    f"Frame {frame_number} | Track {track_id} NOT finalized because "
                    f"text_count={len(track_data['texts'])} < MIN_FRAMES_TO_FINALIZE={MIN_FRAMES_TO_FINALIZE}"
                )

            tracks_to_delete.append(track_id)

    for track_id in tracks_to_delete:
        logger.info(f"Frame {frame_number} | Deleting track {track_id}")
        del tracks[track_id]

    # Draw frame number
    cv2.putText(
        frame,
        f"Frame: {frame_number}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2,
    )

    cv2.imshow("ALPR Polygon ROI", frame)
    out.write(frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("r"):
        logger.info(f"Frame {frame_number} | User requested ROI redraw")
        polygon = get_polygon_roi(frame)

    if key == ord("q"):
        logger.info(f"Frame {frame_number} | User pressed q, stopping")
        break

# Finalize remaining tracks at end
logger.info("Finalizing remaining active tracks at end of video")

for track_id, track_data in tracks.items():
    if len(track_data["texts"]) >= MIN_FRAMES_TO_FINALIZE:
        finalize_track(track_id, track_data)
    else:
        logger.warning(
            f"End of video | Track {track_id} not finalized because "
            f"text_count={len(track_data['texts'])} < MIN_FRAMES_TO_FINALIZE={MIN_FRAMES_TO_FINALIZE}"
        )

cap.release()
out.release()
cv2.destroyAllWindows()

# ---------------------------
# 📦 Post-Processing: Merge Tracks by Plate
# ---------------------------
logger.info("Starting post-processing: Merge tracks by similar plates")

if os.path.exists(LOG_FILE):
    merged_data = {}
    try:
        with open(LOG_FILE, mode="r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                plate = row["final_plate"]
                if not plate:
                    continue
                if plate not in merged_data:
                    merged_data[plate] = {
                        "start_frame": int(row["start_frame"]),
                        "end_frame": int(row["end_frame"]),
                        "num_votes": int(row["num_votes"])
                    }
                else:
                    merged_data[plate]["start_frame"] = min(merged_data[plate]["start_frame"], int(row["start_frame"]))
                    merged_data[plate]["end_frame"] = max(merged_data[plate]["end_frame"], int(row["end_frame"]))
                    merged_data[plate]["num_votes"] += int(row["num_votes"])

        # Overwrite CSV with merged results
        with open(LOG_FILE, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["track_id", "start_frame", "end_frame", "final_plate", "num_votes"])
            for i, (plate, data) in enumerate(merged_data.items(), 1):
                writer.writerow([i, data["start_frame"], data["end_frame"], plate, data["num_votes"]])
                logger.info(f"Merged track {i} for plate {plate}: {data['start_frame']}->{data['end_frame']} ({data['num_votes']} votes)")

        print(f"Post-processing complete. Merged results written to {LOG_FILE}")
    except Exception as e:
        logger.error(f"Error during post-processing: {e}")
        print(f"Post-processing failed: {e}")
else:
    logger.warning("Log file not found; skipping merge post-processing")

logger.info("===== ALPR SCRIPT FINISHED =====")