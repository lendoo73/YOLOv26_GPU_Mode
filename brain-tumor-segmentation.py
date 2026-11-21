import streamlit as st
import numpy as np
import cv2
from ultralytics import YOLO
import argparse

# ========================
# PAGE CONFIGURATION
# ========================
# Configure the Streamlit page title, icon, and layout
st.set_page_config(page_title="Brain Tumor AI Dashboard", page_icon="🧠", layout="wide")

# ========================
# ARGUMENT PARSER FOR MODEL PATH    
# ========================
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="yolo26s_brain_tumor.pt")
args, _ = parser.parse_known_args()

model_path = args.model

# ========================
# LOAD YOLO MODEL
# ========================
# Cache the model so it is loaded only once and reused
@st.cache_resource
def load_model(model_path):
    # Load trained YOLO model for brain tumor segmentation
    return YOLO(model_path)

# Initialize the model
model = load_model(model_path)

# ========================
# CUSTOM DARK MODE STYLING
# ========================
# Apply custom CSS to make the dashboard look modern with dark mode
st.markdown("""
<style>
body { background-color: #121212; color: #e0e0e0; }
.title { font-size: 42px; font-weight: 700; color: #ffffff; }
.subtitle { font-size: 18px; color: #b0b0b0; margin-bottom: 30px; }

.diagnosis-normal { 
background: #064e3b; 
padding: 15px; 
margin-bottom: 15px; 
border-radius: 15px; 
color: #a7f3d0; 
font-weight: 700; 
text-align: center; 
}

.diagnosis-tumor { 
background: #7f1d1d; 
margin-bottom: 15px; 
padding: 15px; 
border-radius: 15px; 
color: #fecaca; 
font-weight: 700; 
text-align: center; 
}

.metric-card { 
background: #2c2c2c; 
border-radius: 15px; 
padding: 20px; 
text-align: center; 
margin-bottom: 15px; 
box-shadow: 0px 6px 15px rgba(0,0,0,0.5); 
color: #e0e0e0; 
}
</style>
""", unsafe_allow_html=True)

# ========================
# HEADER TITLE
# ========================
# Display the main dashboard title
st.markdown('<div class="title">🧠 Brain Tumor Segmentation System</div>', unsafe_allow_html=True)

# ========================
# MAIN LAYOUT (LEFT = INPUT, RIGHT = RESULT)
# ========================
# Create two main columns
left_col, right_col = st.columns([1, 1.5])

# ========================
# LEFT COLUMN: IMAGE UPLOAD
# ========================
with left_col:
    # Upload section title
    st.markdown("<h2 style='font-size:32px;'>📤 Upload MRI Image</h2>", unsafe_allow_html=True)

    # Upload MRI image (JPG / PNG / JPEG)
    uploaded_file = st.file_uploader("Select MRI image", type=["jpg","png","jpeg"])

    # Confidence threshold slider for detection filtering
    conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.25)

    st.markdown('</div>', unsafe_allow_html=True)

# ========================
# RIGHT COLUMN: SEGMENTATION RESULT
# ========================
with right_col:
    # Result section title
    st.markdown("<h2 style='font-size:32px;'>📊 Segmentation Result</h2>", unsafe_allow_html=True)    

    if uploaded_file is not None:

        # ===== READ IMAGE USING OPENCV =====
        # Convert uploaded file into numpy array
        file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)

        # Decode image into OpenCV format (BGR)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)        

        # Run YOLO prediction
        with st.spinner("Analyzing MRI image..."):
            results = model.predict(source=image, conf=conf_threshold)

            # Draw segmentation results on image
            annotated = results[0].plot(line_width=4)

            # Extract segmentation masks and bounding boxes
            masks = results[0].masks
            boxes = results[0].boxes

        detected_classes = []        

        # ===== CHECK IF TUMOR MASK EXISTS =====
        if masks is not None and len(masks.xy) > 0:            

            # Loop through each detected mask
            for i, mask_obj in enumerate(masks.xy):                

                # Get class ID from bounding box
                cls_id = int(boxes[i].cls[0])

                # Convert class ID to class name
                detected_classes.append(model.names[cls_id])

            # Remove duplicate classes
            detected_classes = list(set(detected_classes))

            # ===== CREATE NESTED LAYOUT (IMAGE LEFT, INFO RIGHT) =====
            img_col, info_col = st.columns([2,1])

            # Display segmented MRI image
            with img_col:
                st.image(annotated)

            # Display diagnostic information
            with info_col:

                # Tumor detection message
                st.markdown('<div class="diagnosis-tumor" style="font-size: 1.5em;">Tumor Detected</div>', unsafe_allow_html=True)

                # Display number of tumor regions
                st.markdown(f'<div class="metric-card"><h3 style="font-size: 1.5em;">{len(detected_classes)}</h3><p>Tumor Regions</p></div>', unsafe_allow_html=True)

                # Calculate average confidence score
                avg_conf = float(np.mean([float(box.conf[0]) for box in boxes]))

                # Show average model confidence
                st.markdown(f'<div class="metric-card"><h3>{avg_conf:.2f}</h3><p>Average Confidence</p></div>', unsafe_allow_html=True)                

                # ===== SHOW MEDICAL INFORMATION PER TUMOR TYPE =====
                for cls in detected_classes:

                    if cls.lower()=="glioma":
                        with st.expander("Glioma Info"):
                            st.write("A tumor originating from glial cells that can be malignant. Symptoms may include headaches, nausea, and neurological disorders.")

                    elif cls.lower()=="meningioma":
                        with st.expander("Meningioma Info"):
                            st.write("A tumor that arises from the membranes covering the brain. It is usually benign but can press on brain tissue.")

                    elif cls.lower()=="pituitary":
                        with st.expander("Pituitary Tumor Info"):
                            st.write("A tumor of the pituitary gland that can affect hormones and vision.")

        else:
            # ===== CASE: NO TUMOR DETECTED =====
            img_col, info_col = st.columns([2,1])

            # Show original image
            with img_col:
                st.image(image)

            # Display normal diagnosis message
            with info_col:
                st.markdown('<div class="diagnosis-normal" style="font-size: 1.5em;">Normal - No Tumor Detected</div>', unsafe_allow_html=True)

                st.write("No tumor was detected in the MRI image. Please consult a medical specialist for an official diagnosis.")

    else:
        # Message when no image has been uploaded yet
        st.info("Upload MRI image to see detection result.")

    st.markdown('</div>', unsafe_allow_html=True)

# ========================
# FOOTER
# ========================
# Footer section
st.markdown("""
<br><hr>
<center style="color:gray;">© 2026 Brain Tumor Segmentation System | Stunning Vision AI</center>
""", unsafe_allow_html=True)