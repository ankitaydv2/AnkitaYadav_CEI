# Week 6 - MNIST Image Denoising using Autoencoder

## Project Overview
This project builds a **Convolutional Denoising Autoencoder** for MNIST handwritten digit images. The model receives a noisy digit image as input and learns to reconstruct the clean digit image as output.

A Streamlit web app is also included so the model can be tested interactively using MNIST samples or uploaded digit images.

## Dataset
Dataset used: **MNIST handwritten digit dataset**

Resource: Kaggle MNIST Dataset  
`https://www.kaggle.com/datasets/awsaf49/mnist-dataset`

The code supports two options:

1. Kaggle CSV files placed inside the `data/` folder:
   - `data/mnist_train.csv`
   - `data/mnist_test.csv`

2. TensorFlow built-in MNIST loader if CSV files are not present.

This keeps the project easy to run while still staying aligned with the MNIST dataset requirement.

## Why Autoencoder?
An autoencoder is useful when the goal is to reconstruct data. For denoising, the input is a corrupted/noisy image and the target output is the original clean image.

## Model Architecture
The project uses a simple convolutional autoencoder:

### Encoder
- Conv2D layer
- MaxPooling2D layer
- Conv2D layer
- MaxPooling2D layer

### Decoder
- Conv2D layer
- UpSampling2D layer
- Conv2D layer
- UpSampling2D layer
- Final Conv2D layer with sigmoid activation

## Workflow
1. Load MNIST dataset
2. Normalize pixel values between 0 and 1
3. Add Gaussian noise to the images
4. Train the autoencoder
5. Save the trained model
6. Generate output visualizations
7. Deploy the trained model using Streamlit

## Project Structure
```text
Week6_MNIST_Denoising_Autoencoder/
│
├── app.py
├── train_model.py
├── mnist_denoising_autoencoder.ipynb
├── requirements.txt
├── README.md
├── project_explanation.txt
│
├── data/
│   └── optional Kaggle CSV files
│
├── models/
│   └── mnist_denoising_autoencoder.keras
│
└── outputs/
    ├── training_loss.png
    └── denoising_results.png
```

## How to Run Locally
Install dependencies:

```bash
pip install -r requirements.txt
```

Train the model:

```bash
python train_model.py
```

Run the Streamlit app:

```bash
streamlit run app.py
```

## Streamlit Deployment
1. Push this folder to GitHub.
2. Go to Streamlit Cloud.
3. Click **New app**.
4. Select your GitHub repository.
5. Set the main file path as:

```text
Week6_MNIST_Denoising_Autoencoder/app.py
```

6. Click **Deploy**.

## Expected Output
The output image contains three rows:

- Original clean digit
- Noisy digit
- Denoised digit reconstructed by the model

## Output Images : 
<img width="2880" height="1704" alt="image" src="https://github.com/user-attachments/assets/deb1ab94-a9a4-4068-8801-ffcedbb82a79" />
<img width="2880" height="1704" alt="image" src="https://github.com/user-attachments/assets/f9ca254c-45cb-4e21-8c8e-e83e2248ea5a" />



## Conclusion
The convolutional autoencoder learns important digit features from noisy images and reconstructs cleaner handwritten digits. This shows how autoencoders can be used for image denoising and image reconstruction tasks.
