import os
import cv2
import json
import time
import redis
import numpy as np
import csv
import logging
from datetime import datetime, timezone
from collections import Counter
import requests
from io import BytesIO
from fast_alpr import ALPR

# ---------------------------
# Config & Paths
# ---------------------------
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
BACKEND_URL = os.getenv("BACKEND_URL", "http://host.docker.internal:3000")
ROI_FILE = "roi_polygon.json"
LOG_FILE = "alpr_log.csv"
DEBUG_LOG_FILE = "debug.log"

# ---------------------------
# Logging Setup
# ---------------------------
logging.basicConfig(
    filename=DEBUG_LOG_FILE,
    filemode="a",
    level=logging.DEBUG, # Changed to DEBUG for full visibility
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# ---------------------------
# Initialize ALPR
# ---------------------------
# We use CPU for the detector because CoreML has a bug on macOS 
# where it crashes on frames with 0 detections. 
# We keep CoreML available for OCR as it is more stable there.
alpr = ALPR(
    detector_model="yolo-v9-t-384-license-plate-end2end",
    ocr_model="cct-xs-v2-global-model",
    detector_providers=['CPUExecutionProvider']
)

# ---------------------------
# Tracking Config
# ---------------------------
IOU_THRESHOLD = 0.2
MAX_MISSED_FRAMES = 15
MIN_FRAMES_TO_FINALIZE = 5


# ---------------------------
# State
# ---------------------------
tracks = {}
next_track_id = 1
frame_counters = {} # cam_name -> local_frame_count

# ---------------------------
# Helpers (Ported from main.py)
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
    return inter_area / union if union > 0 else 0.0

def character_majority_vote(texts, track_id=None):
    cleaned = [t.strip().upper() for t in texts if t and t.strip()]
    if not cleaned: return "", {}

    length_counts = Counter(len(t) for t in cleaned)
    target_length = length_counts.most_common(1)[0][0]
    filtered = [t for t in cleaned if len(t) == target_length]

    final_chars = []
    for i in range(target_length):
        chars_at_pos = [t[i] for t in filtered]
        counts = Counter(chars_at_pos)
        selected_char = counts.most_common(1)[0][0]
        final_chars.append(selected_char)

    res = "".join(final_chars)
    # Hardcode fix for OCR commonly misreading "KA" as "AA"
    if res.startswith("AA"):
        res = "KA" + res[2:]

    return res, {"votes": len(cleaned)}

def finalize_track(track_id, track_data, r_conn):
    final_plate, stats = character_majority_vote(track_data["texts"], track_id=track_id)
    if final_plate:
        # Write to local CSV as requested
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        with open(LOG_FILE, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp,
                track_id,
                track_data["start_frame"],
                track_data["last_seen"],
                final_plate,
                len(track_data["texts"])
            ])
        
        # --- POST TO BACKEND API ---
        api_image_url = None
        try:
            if "best_crop" in track_data and track_data["best_crop"] is not None:
                # Encode image to JPEG
                _, buffer = cv2.imencode('.jpg', track_data["best_crop"])
                img_io = BytesIO(buffer)

                # Metadata for API
                payload = {
                    "cameraName": track_data.get("cam_name", "unknown"),
                    "plate": final_plate,
                    "timestamp": datetime.now(timezone.utc).isoformat().replace('+00:00', '') + "Z"
                }
                
                # Send Multipart POST
                api_url = f"{BACKEND_URL}/api/anpr/detections"
                files = {'image': ('plate.jpg', img_io, 'image/jpeg')}
                response = requests.post(api_url, data=payload, files=files, timeout=5)
                
                if response.status_code == 201:
                    logger.info(f"Successfully posted detection for {final_plate} to Backend")
                    try:
                        res_data = response.json()
                        api_image_url = res_data.get("imageUrl")
                    except Exception:
                        pass
                else:
                    logger.warning(f"Failed to post AI results: {response.status_code} - {response.text}")
        except Exception as api_err:
            logger.error(f"Error calling Backend API for ANPR: {api_err}")

        # Broadcast to System
        event_payload = {
            "event": "plate_recognized",
            "camera": track_data.get("cam_name", "unknown"),
            "timestamp": datetime.now(timezone.utc).isoformat().replace('+00:00', '') + "Z",
            "plate": final_plate,
            "track_id": track_id,
            "imageUrl": api_image_url
        }
        r_conn.publish("detection_events", json.dumps(event_payload))
        logger.info(f"Finalized Track {track_id}: {final_plate}")


def load_polygon_roi():
    if os.path.exists(ROI_FILE):
        try:
            with open(ROI_FILE, "r") as f:
                points = json.load(f)
            return np.array(points, dtype=np.int32)
        except:
            return None
    return None

# ---------------------------
# Main Loop
# ---------------------------
def main():
    logger.info("Starting Real-time ANPR Engine...")
    r = redis.from_url(REDIS_URL)

    # Initialize CSV header if missing
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "track_id", "start_frame", "end_frame", "final_plate", "num_votes"])

    polygon = load_polygon_roi()

    print("Polygon loaded successfully")

    while True:
        # Blocking pop for multiple cameras
        result = r.brpop("anpr_queue", timeout=0)
        if not result: continue

        _, packet = result
        if b"|SPLIT|" not in packet: continue

        try:
            metadata_bytes, img_bytes = packet.split(b"|SPLIT|", 1)
            metadata = json.loads(metadata_bytes.decode('utf-8'))
            cam_name = metadata.get("cam_name", "unknown")
            
            # Local frame numbering for this session
            frame_counters[cam_name] = frame_counters.get(cam_name, 0) + 1
            frame_num = frame_counters[cam_name]

            logger.debug(f"Processing Frame {frame_num} for Camera {cam_name}")

            # Decode Frame
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None:
                logger.warning(f"Failed to decode frame {frame_num} from camera {cam_name}")
                continue

            # Apply ROI if exists
            if polygon is not None:
                mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                cv2.fillPoly(mask, [polygon], 255)
                roi_frame = cv2.bitwise_and(frame, frame, mask=mask)
                
                # Check for ROI validity
                non_zero_pixels = cv2.countNonZero(mask)
                logger.debug(f"ROI applied. Total visible pixels: {non_zero_pixels} / {mask.size}")
            else:
                roi_frame = frame

            # Setup visualization frame
            display_frame = frame.copy()
            if polygon is not None:
                cv2.polylines(display_frame, [polygon], True, (0, 255, 255), 2)

            # Predict
            logger.debug(f"Sending frame {frame_num} to ALPR detector...")
            results = []
            try:
                results = alpr.predict(roi_frame)
                logger.debug(results)
                logger.debug(f"Detector returned {len(results)} results")
            except Exception as predict_err:
                logger.error(f"Prediction crash (possible model/EP anomaly): {predict_err}")
            
            if results:
                # Process detections
                current_detections = []
                matched_track_ids = set()

                for result in results:
                    bbox = result.detection.bounding_box
                    x1, y1, x2, y2 = int(bbox.x1), int(bbox.y1), int(bbox.x2), int(bbox.y2)
                    plate_text = result.ocr.text.strip().upper() if result.ocr.text else ""
                    current_box = (x1, y1, x2, y2)

                    # Visualization: Draw on display_frame
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(display_frame, plate_text, (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    # Match
                    best_track_id = None
                    best_iou = 0.0

                    # 1. First priority: Match by Exact Plate Text (Re-identification)
                    if plate_text:
                        for tid, tdata in tracks.items():
                            if tdata.get("cam_name") != cam_name: continue
                            vote_text, _ = character_majority_vote(tdata["texts"], track_id=tid)
                            if vote_text and plate_text == vote_text:
                                best_track_id = tid
                                break

                    # 2. Second priority: Match by IOU (Location tracking)
                    if best_track_id is None:
                        for tid, tdata in tracks.items():
                            if tdata.get("cam_name") != cam_name: continue
                            iou = compute_iou(current_box, tdata["bbox"])
                            if iou > best_iou and iou >= IOU_THRESHOLD:
                                # Anti-Merging Filter: If this track already holds a confident plate, 
                                # and the current frame sees a dramatically different plate, 
                                # DO NOT merge them! They are separate cars occupying the same physical space.
                                vote_text, _ = character_majority_vote(tdata["texts"], track_id=tid)
                                if plate_text and vote_text and len(tdata["texts"]) >= 3:
                                    matches = sum(1 for a, b in zip(plate_text, vote_text) if a == b)
                                    max_len = max(len(plate_text), len(vote_text))
                                    if max_len > 0 and (matches / max_len) < 0.4:
                                        continue # Differs by >60%, skip merging
                                
                                best_iou = iou
                                best_track_id = tid

                    global next_track_id
                    if best_track_id is None:
                        best_track_id = next_track_id
                        tracks[best_track_id] = {
                            "bbox": current_box, "texts": [], "last_seen": frame_num,
                            "start_frame": frame_num, "cam_name": cam_name,
                            "best_crop": None, "max_crop_area": 0
                        }
                        next_track_id += 1
                    
                    tracks[best_track_id]["bbox"] = current_box
                    tracks[best_track_id]["last_seen"] = frame_num
                    if plate_text:
                        tracks[best_track_id]["texts"].append(plate_text)
                        
                    # Maintain the BEST CROP (highest resolution/cleanest)
                    crop_w = x2 - x1
                    crop_h = y2 - y1
                    area = crop_w * crop_h
                    if area > tracks[best_track_id].get("max_crop_area", 0):
                        # Ensure crop stays within frame bounds
                        h, w = frame.shape[:2]
                        pad = 10
                        cy1, cy2 = max(0, y1-pad), min(h, y2+pad)
                        cx1, cx2 = max(0, x1-pad), min(w, x2+pad)
                        
                        tracks[best_track_id]["best_crop"] = frame[cy1:cy2, cx1:cx2].copy()
                        tracks[best_track_id]["max_crop_area"] = area

                    matched_track_ids.add(best_track_id)

                    # Add to live detections
                    current_vote, _ = character_majority_vote(tracks[best_track_id]["texts"], track_id=best_track_id)
                    current_detections.append({
                        "bbox": [x1, y1, x2, y2],
                        "plate": current_vote or plate_text,
                        "track_id": best_track_id
                    })

                # Save live results to Redis
                r.setex(f"latest_anpr_detections:{cam_name}", 5, json.dumps(current_detections))
            else:
                # Still update Redis with empty results to keep dashboard fresh
                r.setex(f"latest_anpr_detections:{cam_name}", 5, json.dumps([]))

            # Finalize stale tracks (Outside the if results block)
            to_del = []
            for tid, tdata in tracks.items():
                if tdata.get("cam_name") != cam_name: continue
                if frame_num - tdata["last_seen"] > MAX_MISSED_FRAMES:
                    if len(tdata["texts"]) >= MIN_FRAMES_TO_FINALIZE:
                        finalize_track(tid, tdata, r)
                    to_del.append(tid)
            for tid in to_del: del tracks[tid]

            # Headless mode — no display window

        except Exception as e:
            logger.error(f"Error in ANPR loop: {e}")
            
    cv2.destroyAllWindows()

if __name__ == "__main__":
    time.sleep(2)  # Wait for Redis to be ready
    main()
