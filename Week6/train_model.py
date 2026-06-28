"""
Week 6 Assessment
MNIST Image Denoising using Convolutional Autoencoder

Input  : noisy handwritten digit image
Output : reconstructed clean digit image
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"
MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


def load_mnist_data():
    """Load MNIST from Kaggle CSV if present; otherwise use TensorFlow MNIST."""
    kaggle_train = DATA_DIR / "mnist_train.csv"
    kaggle_test = DATA_DIR / "mnist_test.csv"
    alt_train = DATA_DIR / "train.csv"
    alt_test = DATA_DIR / "test.csv"

    if kaggle_train.exists() and kaggle_test.exists():
        train_df = pd.read_csv(kaggle_train)
        test_df = pd.read_csv(kaggle_test)
        x_train = train_df.iloc[:, 1:].values.reshape(-1, 28, 28, 1)
        x_test = test_df.iloc[:, 1:].values.reshape(-1, 28, 28, 1)
        source = "Kaggle CSV: mnist_train.csv and mnist_test.csv"
    elif alt_train.exists():
        train_df = pd.read_csv(alt_train)
        if "label" in train_df.columns:
            x_train = train_df.drop(columns=["label"]).values.reshape(-1, 28, 28, 1)
        else:
            x_train = train_df.iloc[:, 1:].values.reshape(-1, 28, 28, 1)
        if alt_test.exists():
            test_df = pd.read_csv(alt_test)
            if "label" in test_df.columns:
                x_test = test_df.drop(columns=["label"]).values.reshape(-1, 28, 28, 1)
            else:
                x_test = test_df.values.reshape(-1, 28, 28, 1)
        else:
            x_test = x_train[-10000:]
            x_train = x_train[:-10000]
        source = "Kaggle CSV: train.csv/test.csv"
    else:
        (x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
        x_train = x_train.reshape(-1, 28, 28, 1)
        x_test = x_test.reshape(-1, 28, 28, 1)
        source = "TensorFlow built-in MNIST"

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    return x_train, x_test, source


def add_noise(images, noise_factor=0.45):
    noisy = images + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=images.shape)
    return np.clip(noisy, 0.0, 1.0)


def build_autoencoder():
    input_img = layers.Input(shape=(28, 28, 1))

    # Encoder
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(input_img)
    x = layers.MaxPooling2D((2, 2), padding="same")(x)
    x = layers.Conv2D(16, (3, 3), activation="relu", padding="same")(x)
    encoded = layers.MaxPooling2D((2, 2), padding="same")(x)

    # Decoder
    x = layers.Conv2D(16, (3, 3), activation="relu", padding="same")(encoded)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = layers.UpSampling2D((2, 2))(x)
    decoded = layers.Conv2D(1, (3, 3), activation="sigmoid", padding="same")(x)

    autoencoder = models.Model(input_img, decoded, name="mnist_denoising_autoencoder")
    autoencoder.compile(optimizer="adam", loss="binary_crossentropy")
    return autoencoder


def save_loss_plot(history):
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Autoencoder Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Binary Crossentropy Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "training_loss.png", dpi=150)
    plt.close()


def save_result_grid(clean, noisy, denoised, n=10):
    plt.figure(figsize=(18, 5))
    for i in range(n):
        plt.subplot(3, n, i + 1)
        plt.imshow(clean[i].reshape(28, 28), cmap="gray")
        plt.title("Original", fontsize=9)
        plt.axis("off")

        plt.subplot(3, n, i + 1 + n)
        plt.imshow(noisy[i].reshape(28, 28), cmap="gray")
        plt.title("Noisy", fontsize=9)
        plt.axis("off")

        plt.subplot(3, n, i + 1 + 2 * n)
        plt.imshow(denoised[i].reshape(28, 28), cmap="gray")
        plt.title("Denoised", fontsize=9)
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "denoising_results.png", dpi=150)
    plt.close()


def main():
    x_train, x_test, source = load_mnist_data()
    print(f"Dataset source: {source}")
    print("Training shape:", x_train.shape)
    print("Testing shape :", x_test.shape)

    x_train_noisy = add_noise(x_train)
    x_test_noisy = add_noise(x_test)

    autoencoder = build_autoencoder()
    autoencoder.summary()

    history = autoencoder.fit(
        x_train_noisy,
        x_train,
        epochs=8,
        batch_size=128,
        shuffle=True,
        validation_data=(x_test_noisy, x_test),
    )

    model_path = MODEL_DIR / "mnist_denoising_autoencoder.keras"
    autoencoder.save(model_path)

    denoised = autoencoder.predict(x_test_noisy[:10])
    save_loss_plot(history)
    save_result_grid(x_test[:10], x_test_noisy[:10], denoised, n=10)

    print("Project completed successfully.")
    print(f"Saved model: {model_path}")
    print("Saved outputs: outputs/training_loss.png and outputs/denoising_results.png")


if __name__ == "__main__":
    main()
