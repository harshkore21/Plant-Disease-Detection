import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Load all the models and define disease names
models = {
    'Tomato': load_model('models/Tomato_model_new_subset.h5'),
    'Tea': load_model('models/Tea_model_new_subset (1).h5'),
    'Sugarcane': load_model('models/Sugarcane_model_new_subset.h5'),
    'Strawberry': load_model('models/Strawberry_model_new_subset.h5'),
    'Soyabean': load_model('models/Soyabean_model_new_subset.h5'),
    'Rice': load_model('models/Rice_model_new_subset.h5'),
    'Potato': load_model('models/Potato_model_new_subset.h5'),
    'Pepper_Bell': load_model('models/Pepper_Bell_model_new_subset.h5'),
    'Peach': load_model('models/Peach_model_new_subset.h5'),
    'Mango': load_model('models/Mango_model_new_subset.h5'),
    'Lemon': load_model('models/Lemon_model_new_subset.h5'),
    'Jamun': load_model('models/Jamun_model_new_subset.h5'),
    'Grape': load_model('models/Grape_model_new_subset.h5'),
    'Cassava': load_model('models/Cassava_model_new_subset.h5'),
    'Apple': load_model('models/Apple_model_new_subset.h5')
}

disease_names = {
    'Tomato': ['Tomato__leaf_mold', 'Tomato__mosaic_virus', 'Tomato__spider_mites_(two_spotted_spider_mite)', 'Tomato__late_blight', 'Tomato__septoria_leaf_spot','Tomato__target_spot','Tomato__yellow_leaf_curl_virus','Tomato__early_blight','Tomato__healthy','Tomato__bacterial_spot'],  # Replace with actual disease names
    'Tea': ['Tea__red_leaf_spot', 'Tea__healthy', 'Tea__brown_blight', 'Tea__bird_eye_spot', 'Tea__anthracnose', 'Tea__algal_leaf'],
    'Sugarcane': ['Sugarcane__rust', 'Sugarcane__red_stripe', 'Sugarcane__red_rot', 'Sugarcane__healthy', 'Sugarcane__bacterial_blight'],
    'Strawberry': ['Strawberry___leaf_scorch', 'Strawberry__healthy'] ,
    'Soyabean': ['Soybean__southern_blight', 'Soybean__rust', 'Soybean__powdery_mildew', 'Soybean__mosaic_virus', 'Soybean__healthy','Soybean__downy_mildew','Soybean__diabrotica_speciosa','Soybean__caterpillar','Soybean__bacterial_blight'],
    'Rice': ['Rice__neck_blast', 'Rice__leaf_blast', 'Rice__hispa', 'Rice__healthy', 'Rice__brown_spot'],
    'Potato': ['Potato__late_blight', 'Potato__healthy', 'Potato__early_blight'],
    # Add other models' disease names here
}

def preprocess_image(image):
    img = Image.open(image)
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0  # Normalize image to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

def predict_disease(model, image_array):
    prediction = model.predict(image_array)
    predicted_class_index = np.argmax(prediction)
    return predicted_class_index

def main():
    st.title('Plant Disease Prediction')

    # Add an interesting background
    st.markdown(
        """
        <style>
            body {
                background-color: #7FFFD4;  /* Background color for the main part */
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Add some animations or decorations
    st.sidebar.markdown("## Welcome to Plant Disease Prediction App")
    st.sidebar.markdown("Please select the plant and upload the image to predict the disease.")
    st.sidebar.text("")  # Add some spacing
    st.sidebar.text("")  # Add some spacing
    st.sidebar.text("")  # Add some spacing
    st.sidebar.text("")  # Add some spacing
    st.sidebar.text("")  # Add some spacing
    st.sidebar.markdown("---")  # Add a horizontal line

    # Select the plant
    selected_plant = st.sidebar.selectbox('Select Plant:', list(models.keys()), key='select_plant')

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type="jpg", key='uploaded_file')

    if uploaded_file is not None:
        try:
            # Preprocess the image
            img_array = preprocess_image(uploaded_file)

            # Perform prediction using the selected model
            if selected_plant in models:
                predicted_class_index = predict_disease(models[selected_plant], img_array)
                predicted_class = disease_names[selected_plant][predicted_class_index]
                st.success(f'The predicted disease for {selected_plant} is {predicted_class}')
            else:
                st.error('Selected plant model not found')
        except Exception as e:
            st.error(f"Error processing image: {e}")

if __name__ == '__main__':
    main()
