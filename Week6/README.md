# 🧠 Week 6: MNIST Denoising Autoencoder

## 📌 Project Overview

This project focuses on building a **Convolutional Denoising Autoencoder** using **TensorFlow** and **Keras**. The objective is to remove random Gaussian noise from handwritten digit images in the MNIST dataset and reconstruct clean, high-quality images.

A **Streamlit web application** has been developed to provide an interactive interface where users can visualize the denoising process and compare noisy and reconstructed images.

---

# 🎯 Objectives

- Understand the concept of Autoencoders.
- Learn image denoising using Deep Learning.
- Train a Convolutional Neural Network to reconstruct clean images.
- Deploy the trained model using Streamlit.
- Visualize the effectiveness of image reconstruction.

---

# 📂 Project Structure

```
Week6_MNIST_Denoising_Autoencoder_Final/
│
├── app.py
├── train_model.py
├── mnist_denoising_autoencoder.ipynb
├── requirements.txt
├── README.md
├── project_explanation.txt
├── .gitignore
│
├── models/
│   └── mnist_denoising_autoencoder.keras
│
└── outputs/
    └── sample_predictions.png
```

---

# 🛠️ Technologies Used

| Technology | Purpose |
|------------|---------|
| Python | Programming Language |
| TensorFlow | Deep Learning Framework |
| Keras | Neural Network API |
| NumPy | Numerical Operations |
| Matplotlib | Data Visualization |
| Streamlit | Interactive Web Application |

---

# 📊 Dataset Information

**Dataset:** MNIST Handwritten Digits

- Total Training Images: **60,000**
- Total Testing Images: **10,000**
- Image Size: **28 × 28 Pixels**
- Color Mode: Grayscale
- Number of Classes: **10 (Digits 0–9)**

To train the model, Gaussian noise is added to the original images. The autoencoder learns to reconstruct the original clean image from the noisy version.

---

# 🧠 Model Architecture

The model follows an **Encoder-Decoder** architecture.

### Encoder
- Input Layer
- Convolution Layer (32 Filters)
- Max Pooling
- Convolution Layer (16 Filters)
- Max Pooling

The encoder compresses the noisy image into a compact latent representation.

### Decoder
- Convolution Layer
- Up Sampling
- Convolution Layer
- Up Sampling
- Output Layer with Sigmoid Activation

The decoder reconstructs the original image from the compressed representation.

---

# ⚙️ Installation

### Clone the Repository

```bash
git clone https://github.com/ankitaydv2/AnkitaYadav_CEI.git
```

### Navigate to the Project

```bash
cd Week6_MNIST_Denoising_Autoencoder_Final
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🚀 Running the Project

### Train the Model

```bash
python train_model.py
```

This will train the denoising autoencoder and save the trained model in the `models` directory.

---

### Launch the Streamlit Application

```bash
streamlit run app.py
```

The application will open in your default browser, allowing you to interact with the trained model.

---

# 📈 Expected Output

The application demonstrates:

- Original MNIST Image
- Noisy Image
- Denoised (Reconstructed) Image

The reconstructed output should closely resemble the original handwritten digit while effectively removing the added noise.

---

# 💡 Key Learning Outcomes

Through this project, the following concepts were explored:

- Image Denoising
- Autoencoders
- Convolutional Neural Networks (CNNs)
- Encoder-Decoder Architecture
- TensorFlow & Keras
- Streamlit Deployment
- Model Saving and Loading
- Image Reconstruction

---

# 🔮 Future Enhancements

- Support for custom image uploads.
- Train on larger and more complex datasets.
- Experiment with different types of image noise.
- Improve reconstruction quality using deeper architectures.
- Deploy the application on Streamlit Community Cloud.

---

# 👩‍💻 Author

**Ankita Yadav**

B.Tech – Computer Science Engineering

GitHub: https://github.com/ankitaydv2

---

# ⭐ Acknowledgement

This project was developed as part of the **Week 6 Assignment** to understand and implement **Image Denoising using Convolutional Autoencoders**. It demonstrates practical applications of Deep Learning in image restoration and reconstruction.
