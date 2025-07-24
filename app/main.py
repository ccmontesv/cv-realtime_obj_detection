# app/main.py
import streamlit as st
from detection import run_detection

st.set_page_config(layout="wide", page_title="YOLOv8 Streamlit App")

run_detection()
