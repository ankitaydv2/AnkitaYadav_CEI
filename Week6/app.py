import numpy as np
import streamlit as st
from PIL import Image, ImageOps
import tensorflow as tf
from tensorflow.keras import layers, models

st.set_page_config(page_title="MNIST Denoising Autoencoder", page_icon="✍️", layout="wide")

MODEL_PATH = "models/mnist_denoising_autoencoder.keras"


def build_autoencoder():
    input_img = layers.Input(shape=(28, 28, 1))
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(input_img)
    x = layers.MaxPooling2D((2, 2), padding="same")(x)
    x = layers.Conv2D(16, (3, 3), activation="relu", padding="same")(x)
    encoded = layers.MaxPooling2D((2, 2), padding="same")(x)
    x = layers.Conv2D(16, (3, 3), activation="relu", padding="same")(encoded)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = layers.UpSampling2D((2, 2))(x)
    decoded = layers.Conv2D(1, (3, 3), activation="sigmoid", padding="same")(x)
    model = models.Model(input_img, decoded)
    model.compile(optimizer="adam", loss="binary_crossentropy")
    return model


@st.cache_resource
def load_or_train_model():
    try:
        return tf.keras.models.load_model(MODEL_PATH), "Loaded trained model from models folder."
    except Exception:
        # Fallback keeps the Streamlit app usable even before uploading the saved model.
        (x_train, _), _ = tf.keras.datasets.mnist.load_data()
        x_train = x_train[:6000].astype("float32") / 255.0
        x_train = x_train.reshape(-1, 28, 28, 1)
        noisy = np.clip(x_train + 0.45 * np.random.normal(size=x_train.shape), 0.0, 1.0)
        model = build_autoencoder()
        model.fit(noisy, x_train, epochs=2, batch_size=128, verbose=0)
        return model, "Demo model trained quickly because saved model was not found."


def prepare_uploaded_image(uploaded_file):
    image = Image.open(uploaded_file).convert("L")
    image = ImageOps.invert(image)
    image = image.resize((28, 28))
    arr = np.array(image).astype("float32") / 255.0
    arr = arr.reshape(1, 28, 28, 1)
    return arr


def add_noise(image_array, noise_factor):
    noisy = image_array + noise_factor * np.random.normal(size=image_array.shape)
    return np.clip(noisy, 0.0, 1.0)


st.title("MNIST Image Denoising using Autoencoder")
st.write("Upload a handwritten digit image or test the model on MNIST samples. The model takes a noisy digit and reconstructs a cleaner version.")

model, status = load_or_train_model()
st.sidebar.success(status)
noise_factor = st.sidebar.slider("Noise level", 0.10, 0.80, 0.45, 0.05)

option = st.radio("Choose input type", ["MNIST sample", "Upload digit image"], horizontal=True)

if option == "MNIST sample":
    (_, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
    x_test = x_test.astype("float32") / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1)
    idx = st.slider("Select sample index", 0, 9999, 7)
    clean = x_test[idx:idx + 1]
else:
    uploaded_file = st.file_uploader("Upload a simple handwritten digit image", type=["png", "jpg", "jpeg"])
    if uploaded_file is None:
        st.info("Upload an image to continue.")
        st.stop()
    clean = prepare_uploaded_image(uploaded_file)

noisy = add_noise(clean, noise_factor)
denoised = model.predict(noisy, verbose=0)

col1, col2, col3 = st.columns(3)
with col1:
    st.subheader("Original")
    st.image(clean.reshape(28, 28), width=220, clamp=True)
with col2:
    st.subheader("Noisy Input")
    st.image(noisy.reshape(28, 28), width=220, clamp=True)
with col3:
    st.subheader("Denoised Output")
    st.image(denoised.reshape(28, 28), width=220, clamp=True)

st.markdown("### Project Summary")
st.write("This app demonstrates a convolutional autoencoder trained for image denoising. The encoder learns compressed digit features and the decoder reconstructs the clean digit image from the noisy input.")
