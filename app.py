import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image

st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="centered"
)

st.title("🌿 Plant Disease Detection")
st.write("Upload a leaf image to detect the plant and its disease.")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "plant_disease_efficientnetb0_final.keras"
    )

@st.cache_data
def load_classes():
    with open("class_names.json", "r") as f:
        return json.load(f)

model = load_model()
class_names = load_classes()

uploaded_file = st.file_uploader(
    "Upload a leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Leaf Image",
        use_container_width=True
    )

    image = image.resize((224, 224))
    image_array = np.array(image)
    image_array = np.expand_dims(image_array, axis=0)

    predictions = model.predict(image_array, verbose=0)[0]

    predicted_index = np.argmax(predictions)
    confidence = predictions[predicted_index] * 100

    predicted_class = class_names[predicted_index]

    plant, disease = predicted_class.split("___")

    st.success(f"Plant: {plant}")
    st.success(f"Disease: {disease}")
    st.info(f"Confidence: {confidence:.2f}%")

    st.subheader("Top Predictions")

    top_indices = np.argsort(predictions)[-5:][::-1]

    for index in top_indices:
        name = class_names[index].split("___")
        st.write(
            f"{name[0]} - {name[1]}: "
            f"{predictions[index] * 100:.2f}%"
        )
