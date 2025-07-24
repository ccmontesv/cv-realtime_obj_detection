import streamlit as st
import cv2
import time
from datetime import datetime
import numpy as np
from utils.model_loader import load_model
from utils.video_stream import process_frame
from utils.logger import init_csv, log_detection

def run_detection():
    # Sidebar options
    st.sidebar.title("YOLOv8 Streamlit App")
    model_option = st.sidebar.selectbox("Select Model Type", ["Detection", "Segmentation", "Pose Estimation"])
    model_paths = {
        "Detection": "models/yolov8n.pt",
        "Segmentation": "models/yolov8n-seg.pt",
        "Pose Estimation": "models/yolov8n-pose.pt"
    }
    model_path = model_paths[model_option]

    camera_index = st.sidebar.number_input("Camera Index", min_value=0, max_value=4, value=0)
    conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.4)

    start_btn = st.sidebar.button("Start Detection")
    stop_btn = st.sidebar.button("Stop Detection")

    # Session control
    if 'run' not in st.session_state:
        st.session_state.run = False
    if start_btn:
        st.session_state.run = True
    if stop_btn:
        st.session_state.run = False

    # Load model and color palette
    model = load_model(model_path)
    COLORS = {cls: [np.random.randint(0, 255) for _ in range(3)] for cls in model.names}

    # Streamlit video frame container
    stframe = st.empty()

    # Start detection loop
    if st.session_state.run:
        cap = cv2.VideoCapture(camera_index)
        if not cap or not cap.isOpened():
            st.error("Could not open camera.")
            return

        csv_file = "detections.csv"
        init_csv(csv_file)

        while st.session_state.run:
            start = time.time()
            ret, frame = cap.read()
            if not ret:
                break

            processed_frame = process_frame(frame, model, conf_threshold, COLORS)

            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            #for det in detections:
            #    log_detection(csv_file, now_str, *det)

            fps = 1 / (time.time() - start)
            cv2.putText(processed_frame, f"FPS: {fps:.1f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

            frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            frame_rgb = cv2.resize(frame_rgb, (640, 480))

            stframe.image(frame_rgb, channels="RGB")  #, use_container_width=True)

        cap.release()
        cv2.destroyAllWindows()

