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
                0: 'C 32 2.5' ,
                1: 'C 38 2.9' ,
                2: 'C 48 2.9' ,
                3: 'R 20 40 1.9' ,
                4: 'R 25 75 1.9' ,
                5: 'R 48 96 2.0' ,
                6: 'R 48 96 2.9' ,
                7: 'R 60 40 1.9' ,
                8: 'R 80 40 1.2' ,
                9: 'R 96 48 2.0' ,
                10: 'R 96 48 2.9' ,
                11: 'S 20 20 1.2' ,
                12: 'S 20 20 1.5' ,
                13: 'S 20 20 1.9' ,
                14: 'S 25 25 1.9' ,
                15: 'S 25 25 2.5' ,
                16: 'S 38 38 1.9' ,
                17: 'S 40 40 2.5' ,
                18: 'S 50 50 1.5' ,
                19: 'S 50 50 1.9' ,
                20: 'S 50 50 4.0' ,
                21: 'S 60 60 2.0' ,
                22: 'S 72 72 4.0' ,
                23: 'S 72 72 4.8' ,
                
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
    
        