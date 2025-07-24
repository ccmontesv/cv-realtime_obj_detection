# app/utils/logger.py
import os
import csv
from datetime import datetime

def init_csv(filepath):
    if not os.path.exists(filepath):
        with open(filepath, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["datetime", "class", "confidence", "x1", "y1", "x2", "y2"])

def log_detection(filepath, label, conf, box):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    x1, y1, x2, y2 = box
    with open(filepath, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([now, label, conf, x1, y1, x2, y2])
