import streamlit as st
from PIL import Image, ImageOps
import numpy as np
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="MNIST Digit Recognizer", page_icon="🔢")
st.title("MNIST Digit Recognizer 🔢")
st.write("Draw a digit or upload an image to classify (0-9).")

@st.cache_resource
def load_model():
    model = None
    return model

def predict(model, image):
    prediction = 0
    confidence = 0.99
    return prediction, confidence

model = load_model()

option = st.radio("Choose input method:", ("Draw a Digit", "Upload an Image"))

if option == "Draw a Digit":
    st.write("Draw a single digit in the box below:")
    
    canvas_result = st_canvas(
        fill_color="#000000",
        stroke_width=15,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )
    
    if st.button("Predict Digit"):
        if canvas_result.image_data is not None:
            img_array = canvas_result.image_data
            image = Image.fromarray(img_array.astype('uint8'), 'RGBA')
            image = image.convert('L')
            
            if model is not None:
                prediction, confidence = predict(model, image)
                st.success(f"### Predicted Digit: {prediction}")
                st.write(f"Confidence: {confidence:.2%}")
            else:
                st.warning("Model is not loaded. Please update the load_model function.")

elif option == "Upload an Image":
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('L')
        
        img_array = np.array(image)
        if img_array[0, 0] > 127:
            image = ImageOps.invert(image)
        
        st.image(image, width=280)
        
        if st.button("Predict Digit"):
            if model is not None:
                prediction, confidence = predict(model, image)
                st.success(f"### Predicted Digit: {prediction}")
                st.write(f"Confidence: {confidence:.2%}")
            else:
                st.warning("Model is not loaded. Please update the load_model function.")
