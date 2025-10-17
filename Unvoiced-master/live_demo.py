import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from gtts import gTTS
import io

# ==========================
# App Title
# ==========================
st.title("Sign Wave")
st.write("Upload an image or use your webcam to detect sign language letters!")

# ==========================
# Load Model and Labels
# ==========================
@st.cache_resource
def load_model_and_labels():
    # Load TensorFlow 2.x model
    model = tf.keras.models.load_model("trained_model_graph.pb")  # Replace with TF2 saved model
    # Load labels
    with tf.io.gfile.GFile("training_set_labels.txt", "r") as f:
        label_lines = [line.strip() for line in f]
    return model, label_lines

model, label_lines = load_model_and_labels()

# ==========================
# Prediction Function
# ==========================
def predict_letter(img):
    # Convert uploaded image to OpenCV format
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # Crop ROI (same as original)
    roi = img[70:350, 70:350]
    roi_resized = cv2.resize(roi, (200, 200))
    
    # Normalize / expand dims
    input_data = np.expand_dims(roi_resized, axis=0) / 255.0
    
    # Predict
    predictions = model.predict(input_data)
    top_idx = np.argmax(predictions[0])
    letter = label_lines[top_idx]
    score = predictions[0][top_idx]
    return letter, score

# ==========================
# Speech Function
# ==========================
def speak_letter(letter):
    tts = gTTS(text=letter, lang="en")
    audio_bytes = io.BytesIO()
    tts.write_to_fp(audio_bytes)
    audio_bytes.seek(0)
    st.audio(audio_bytes.read(), format='audio/mp3')

# ==========================
# User Input: Upload or Webcam
# ==========================
input_method = st.radio("Select Input Method:", ["Upload Image", "Use Webcam"])

if input_method == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        img = np.array(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Uploaded Image", channels="RGB")
        
        letter, score = predict_letter(img)
        st.write(f"Predicted Letter: **{letter.upper()}**  |  Confidence: {score:.2f}")
        speak_letter(letter.upper())

elif input_method == "Use Webcam":
    cam_image = st.camera_input("Take a picture")
    if cam_image:
        img = np.array(bytearray(cam_image.read()), dtype=np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Captured Image", channels="RGB")
        
        letter, score = predict_letter(img)
        st.write(f"Predicted Letter: **{letter.upper()}**  |  Confidence: {score:.2f}")
        speak_letter(letter.upper())

# ==========================
# Current Word Display
# ==========================
# You can extend this by storing the letters in session state
