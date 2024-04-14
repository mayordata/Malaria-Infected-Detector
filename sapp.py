import streamlit as st
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('malaria_dectective_model.keras')

def preprocess_data(image_data):
    img = image.load_img(image_data, target_size=(130,130,3))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def main():
    st.title("Malaria Infected Detector")
    st.write('The app will help the doctor with quick diagnosis of malaria infected patient')
    st.image('th.jpeg')
    st.write("Introducing our malaria detector: a powerful tool for quick and accurate diagnosis. Users can upload images for instant assessment, guiding them to seek timely medical help. This improves hospital workflow, allowing doctors to focus on other diagnoses.")
    st.divider()

    # User Prompt
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
        if st.button('Predict'):
            image = preprocess_data(uploaded_file)
            prediction = model.predict(image)
            predicted_class = prediction > 0.5
     
            if predicted_class > 0.5:
                st.success('The patient is not infected')
            else:
                st.error('The patient is infected')
    else:
        st.warning('Please upload an image first.')

if __name__ == "__main__":
    main()