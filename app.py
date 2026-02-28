import streamlit as st
from PIL import Image, ImageOps
import numpy as np
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="MNIST Digit Recognizer", page_icon="🔢")

st.title("MNIST Digit Recognizer 🔢")
st.write("Draw a digit or upload an image to classify (0-9).")

st.info("Note: The model is currently disabled as requested. The app is set up for UI only.")

# Placeholder function for prediction
def predict(image):
    # This is where the model inference would happen
    # Returning dummy values for now
    return 0, 0.99

option = st.radio("Choose input method:", ("Draw a Digit", "Upload an Image"))

if option == "Draw a Digit":
    st.write("Draw a single digit in the box below:")
    
    # Create a canvas component
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
            # Get the image data from canvas
            img_array = canvas_result.image_data
            image = Image.fromarray(img_array.astype('uint8'), 'RGBA')
            # Convert to grayscale
            image = image.convert('L')
            
            # Predict (using placeholder)
            prediction, confidence = predict(image)
            st.success(f"### Predicted Digit: {prediction}")
            st.write(f"Confidence: {confidence:.2%}")
            
elif option == "Upload an Image":
    uploaded_file = st.file_uploader("Choose an image (preferably square, digit centered)", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('L')
        
        # Simple heuristic: if the image has a white background (edges are bright), invert it
        # MNIST expects white digits on black background
        img_array = np.array(image)
        if img_array[0, 0] > 127: # if top-left pixel is bright
            image = ImageOps.invert(image)
        
        st.image(image, caption='Processed Uploaded Image', width=280)
        
        if st.button("Predict Digit"):
            prediction, confidence = predict(image) # using placeholder
            st.success(f"### Predicted Digit: {prediction}")
            st.write(f"Confidence: {confidence:.2%}")
