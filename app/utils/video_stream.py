import cv2
import time
import csv
import os
import numpy as np
from datetime import datetime

def process_frame(frame, model, conf_threshold=0.4, colors=None, csv_file="detections.csv"):
    """
    Process a single frame, detect objects, draw boxes and log to CSV.
    Returns the processed frame and whether a chair was detected.
    """
    start = time.time()
    chair_detected = False
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Convert frame to RGB and run detection
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model.track(source=frame_rgb, conf=conf_threshold, persist=True)[0]

    # Overlay masks if present
    if results.masks:
        for mask in results.masks.data:
            mask_np = (mask.cpu().numpy() * 255).astype("uint8")
            mask_rgb = cv2.merge([mask_np] * 3)
            frame = cv2.addWeighted(frame, 1.0, mask_rgb, 0.3, 0)

    # Log and draw boxes
    if results.boxes:
        with open(csv_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = f"{model.names[cls_id]} {conf:.2f}"
                color = colors[cls_id] if colors else (0, 255, 0)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, max(10, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                writer.writerow([now_str, model.names[cls_id], round(conf, 2), x1, y1, x2, y2])
                #if model.names[cls_id].lower() == "chair":
                #    chair_detected = True

    return frame       #, chair_detected
