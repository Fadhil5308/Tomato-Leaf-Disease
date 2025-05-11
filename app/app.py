import streamlit as st
import numpy as np
from PIL import Image
from tensorflow import keras

# Load model with caching to avoid repeated loading
@st.cache_resource
def load_model():
    return keras.models.load_model('tomato_disease.h5')

model = load_model()

# Define prevention measures
disease_prevention = {
    "Tomato_Bacterial_spot": [
        "Prevent bacterial spot by using disease-free seeds.",
        "Implement crop rotation to reduce the disease's prevalence.",
        "Apply copper-based fungicides to control the disease."
    ],
    "Tomato_Early_blight": [
        "Prevent early blight by practicing good garden hygiene.",
        "Ensure proper watering to avoid splashing soil onto the leaves.",
        "Apply fungicides as needed to control the disease."
    ],
    "Tomato_Late_blight": [
        "Prevent late blight by providing good air circulation in your garden or greenhouse.",
        "Avoid overhead watering, as wet leaves can encourage the disease.",
        "Apply fungicides when necessary to manage the disease."
    ],
    "Tomato_Leaf_Mold": [
        "Prevent leaf mold by ensuring good air circulation and spacing between plants.",
        "Avoid wetting the leaves when watering, and water the soil instead.",
        "Apply fungicides if the disease is present and worsening."
    ],
    "Tomato_Septoria_leaf_spot": [
        "Prevent Septoria leaf spot by maintaining good garden hygiene.",
        "Avoid overhead watering to keep the leaves dry.",
        "Apply fungicides if the disease becomes a problem."
    ],
    "Tomato_Spider_mites_Two_spotted_spider_mite": [
        "Prevent spider mite infestations by regularly inspecting your plants for signs of infestation.",
        "Increase humidity in the growing area to discourage mites.",
        "Use insecticidal soap or neem oil to control mites if necessary."
    ],
    "Tomato__Target_Spot": [
        "Prevent target spot by ensuring good air circulation and avoiding overcrowding of plants.",
        "Water at the base of the plants, keeping the leaves dry.",
        "Apply fungicides as needed to control the disease."
    ],
    "Tomato__Tomato_YellowLeaf__Curl_Virus": [
        "Prevent Tomato Yellow Leaf Curl Virus by using virus-free tomato plants.",
        "Control whiteflies, which transmit the virus, with insecticides.",
        "Remove and destroy infected plants to prevent the spread of the disease."
    ],
    "Tomato__Tomato_mosaic_virus": [
        "Prevent tomato mosaic virus by using virus-free seeds and disease-resistant tomato varieties.",
        "Control aphids, which transmit the virus, with insecticides.",
        "Remove and destroy infected plants to prevent further spread."
    ],
    "Tomato_healthy": [
        "If your tomato plant is healthy, continue to monitor for pests and diseases regularly.",
        "Follow good gardening practices, including proper watering, fertilization, and maintenance."
    ]
}

# Function to process image
def read_image(image):
    img = Image.open(image)
    img = img.resize((256, 256))
    img = np.array(img)
    if img.shape[-1] != 3:  # Ensure 3 channels (RGB)
        img = np.stack((img,) * 3, axis=-1)
    return img

# Streamlit UI
st.title("Tomato Disease Detection App")
st.write("Upload an image of a tomato leaf, and the app will predict the disease and provide prevention measures.")

uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Process the image
    with st.spinner("Processing image..."):
        try:
            img = read_image(uploaded_file)
            img = np.expand_dims(img, axis=0)  # Add batch dimension

            # Prediction
            prediction = model.predict(img)
            pred = np.argmax(prediction)
            confidence = float(np.max(prediction[0]))

            class_labels = [
                'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight', 'Tomato_Leaf_Mold',
                'Tomato_Septoria_leaf_spot', 'Tomato_Spider_mites_Two_spotted_spider_mite',
                'Tomato__Target_Spot', 'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus',
                'Tomato_healthy'
            ]
            predicted_class = class_labels[pred]

            # Display results
            st.success(f"Prediction: {predicted_class}")
            st.write(f"Confidence: {confidence:.2%}")

            # Display prevention measures
            prevention_measures = disease_prevention.get(predicted_class, ['No prevention measures available.'])
            st.subheader("Prevention Measures:")
            for measure in prevention_measures:
                st.write(f"- {measure}")
        except Exception as e:
            st.error(f"Error during prediction: {e}"
