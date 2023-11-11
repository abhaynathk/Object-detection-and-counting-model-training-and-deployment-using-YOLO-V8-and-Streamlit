# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 21:33:51 2023

@author: Abhay
"""

#installing the required libraries
#pip install torch
#pip install ultralytics
#pip install streamlit
import PIL
import streamlit as st
from ultralytics import YOLO
model_path="best.pt"
max_detections = 1000

#setting page layout
st.set_page_config(
    page_title="Object Counting",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
    )


#creating sidebar
with st.sidebar:
    st.header=("Image_config")
    #adding file uploader to side bar for selecting images
    source_img=st.file_uploader(
        "Upload the Image", type=("jpg","png","jpeg","bmp","webp"))
    #model options
    confidence=float(st.slider(
        "select model confidence", 25, 100, 40)) / 100
#creating main page heading
st.title(":red[Object Detection]")
st.caption("Upload a photo .")
st.caption('Then click the [Detect Objects] button and check the result.')
# Creating two columns on the main page
col1, col2 = st.columns(2)
# Adding image to the first column if image is uploaded
with col1:
    if source_img:
        # Opening the uploaded image
        uploaded_image = PIL.Image.open(source_img)
        # Adding the uploaded image to the page with a caption
        st.image(source_img,
                 caption="Uploaded Image",
                 use_column_width=True
                 )
try:
    model = YOLO(model_path)
except Exception as ex:
    st.error(
        f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)
if st.sidebar.button('Detect Objects'):
    res = model.predict(uploaded_image,
                        conf=confidence, max_det=1000,
                        )
    boxes = res[0].boxes
    res_plotted = res[0].plot()[:, :, ::-1]
    with col2:
        st.image(res_plotted,
                 caption='Detected Image',
                 use_column_width=True 
                 )
        boxes = res[0].boxes
        num_boxes = len(boxes)
        if num_boxes == 0:
            st.error("No objects detected in the image.")
        else:
            st.write(f"Number of detected objects: {num_boxes}")  # Display the total count
            # Create a dictionary to count each class separately
            class_counts = {}
            class_name_map = {
                0: 'Class 1' ,
                1: 'Class 2' ,
                2: 'Class 3' ,
                
            }
            for box in boxes:
                # Access the last element of the tensor to get the class label
                class_label = int(box.cls)  # Convert to integer
                class_name = class_name_map.get(class_label , "Unknown")
                if class_name not in class_counts:
                    class_counts[class_name] = 1
                else:
                    class_counts[class_name] += 1
            st.write("Count of each classes detected:")
            for class_name, count in class_counts.items():
                st.write(f":green[{class_name}:] :- {count}")
        
        try:
            with st.expander("Detection Results"):
                for box in boxes:
                    st.write(box.xywh)
        except Exception as ex:
            st.write("No image is uploaded yet!")
    
        
