# 🧠 YOLOv8 Streamlit App for Real-Time Object Detection, Segmentation & Pose Estimation

This project is a real-time computer vision web app built with **Streamlit** and **Ultralytics YOLOv8** models. It allows users to run **object detection**, **segmentation**, or **pose estimation** on live camera feeds through a simple web interface.

---

##  Features

-  Real-time video processing from your **webcam**
-  Model selection: **Detection**, **Segmentation**, or **Pose Estimation**
-  Adjustable confidence threshold via sidebar
-  Visual overlays with labels and FPS
-  Logging of detections to a CSV file (optional)
-  Start/Stop button to control processing

## Project structure



├── app/                        # Main application logic
│   ├── __init__.py             # Makes app a package
│   ├── main.py                 # Streamlit entry point
│   ├── detection.py            # run_detection() function
│   └── utils/                  # Utility modules
│       ├── model_loader.py     # Loads YOLOv8 models
│       ├── video_stream.py     # Frame processing logic
│       ├── logger.py           # Logging detections to CSV
│       └── __init__.py 
│
├── data/                       # (Optional) Input/output files, if used
│
├── models/                     # YOLOv8 model weights (.pt files)
│
├── test/                       # Testing scripts or unit tests (ignored by Git)
│
├── requirements.txt            # List of Python dependencies
├── README.md                   # Project documentation
├── .gitignore                  # Git ignore rules
└── .ipynb_checkpoints/         # Auto-saved Jupyter checkpoints (ignored)


---

## Installation

1. **Clone this repository**:

   ```bash
   git clone https://github.com/ccmontesv/cv-realtime_obj_detection.git
   ```

2. **Create a virtual environment** (recommended):

   ```bash
   python -m venv CV
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Download YOLOv8 models** (if not already in `models/`):

   You can get pre-trained models from [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics).

   Place the models here:
   - `models/yolov8n.pt` (Detection)
   - `models/yolov8n-seg.pt` (Segmentation)
   - `models/yolov8n-pose.pt` (Pose estimation)

---

##  Running the App

```bash
streamlit run main.py
```

---

## Camera Configuration

- **Camera Index**: Change the index if you have multiple webcams.
  - `0` = default camera (usually your laptop's webcam)
  - `1`, `2`, ... = external cameras
  - I tested the app with DroidCam Client installed in the server the code runs and the iOS app and it worked properly

Adjust this in the sidebar inside the app.

---

##  Output

- All detections (if logging is enabled) are saved to `detections.csv` with:
  - Timestamp
  - Detected object class
  - Confidence score
  - Bounding box coordinates (optional)

---

