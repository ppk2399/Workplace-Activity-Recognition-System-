# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import pandas as pd

st.set_page_config(page_title="YOLOv11 Object Detection", page_icon="🤖", layout="centered")


st.title("Object Detection with YOLOv11")

st.markdown("Upload an image and let the YOLOv11 model detect objects in it.")


st.markdown("""

    <style>

        .stButton>button {

            background-color: #008CBA;

            color: white;

            font-size: 16px;

            border-radius: 10px;

            padding: 10px;

        }

        .stButton>button:hover {

            background-color: #006F8E;

        }

    </style>

""", unsafe_allow_html=True)


uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
   

    image = Image.open(uploaded_file)

    
  

    st.image(image, caption="Uploaded Image", use_container_width=True)

    
    

    model = YOLO(r"C:\Users\Pavan Paradesi\Downloads\Yolov8n\best .pt")



    

    

    results = model(image)  

    

    

    st.image(results[0].plot(), caption="Detected Objects", use_container_width=True)

    

    

    detections = results[0].boxes  

    if len(detections) > 0:

        

        detection_data = {

            'Class': detections.cls.cpu().numpy(),

            'Confidence': detections.conf.cpu().numpy(),

            'X_min': detections.xyxy[:, 0].cpu().numpy(),  

            'Y_min': detections.xyxy[:, 1].cpu().numpy(),  

            'X_max': detections.xyxy[:, 2].cpu().numpy(),

            'Y_max': detections.xyxy[:, 3].cpu().numpy(),  

        }

        

        

        df = pd.DataFrame(detection_data)

        st.subheader("Detection Results:")

        st.write(df)  # Display the dataframe

    else:

        st.write("No objects detected.")