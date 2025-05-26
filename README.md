
# FashionGAN: Generative Adversarial Network for Fashion-MNIST

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)  
![TensorFlow](https://img.shields.io/badge/Built%20with-TensorFlow-brightgreen)  
![Python](https://img.shields.io/badge/Python-3.8+-blue)

FashionGAN is a deep learning project that demonstrates how Generative Adversarial Networks (GANs) can learn to generate realistic images of fashion items, trained on the Fashion-MNIST dataset using TensorFlow 2.x. It includes both a training pipeline and a lightweight Flask web application for interactive image generation.

---

## 📚 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
  - [Training the Model](#1-train-the-model)
  - [Running the Web App](#2-run-the-flask-app)
- [Output](#output)
- [Performance Tips](#performance-tips)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## 🧠 Overview

FashionGAN uses a GAN model to synthesize grayscale 28x28 images of clothing items. It includes a modular implementation of both generator and discriminator models, and supports model training, checkpointing, image generation, and a Flask-based web interface to interactively view results.

---

## ✨ Features

- TensorFlow 2.x implementation
- Modular generator and discriminator models
- Custom training loop for flexibility
- Intermediate image generation during training
- Model saving and image visualization
- Lightweight Flask web app for image display
- Easy to extend and modify

---

## 🏗 Architecture

### Generator
- Fully connected + Reshape
- Batch Normalization
- Conv2DTranspose layers
- Tanh activation on output

### Discriminator
- Convolutional layers with LeakyReLU
- Dropout for regularization
- Sigmoid activation for binary classification

---

## 📂 Dataset

- **Fashion-MNIST**
- 60,000 training images, 10 fashion categories
- Grayscale 28x28 images
- Loaded using `tensorflow_datasets`

```python
import tensorflow_datasets as tfds
ds = tfds.load('fashion_mnist', split='train')
```

---

## 🛠 Installation

### Requirements

- Python 3.8+
- TensorFlow 2.8+
- Flask
- NumPy
- Matplotlib
- Pillow
- TensorFlow Datasets

### Install Dependencies

```bash
pip install tensorflow flask numpy matplotlib pillow tensorflow-datasets
```

---

## 🚀 Usage

### 1. Train the Model

Train the GAN model using the notebook or a Python script:

```bash
# In Jupyter
jupyter notebook fashion.ipynb

# Or as a script
python fashion_gan.py
```

Optional: For quick prototyping:

```python
ds = tfds.load('fashion_mnist', split='train[:10%]')
gan.fit(ds, epochs=5)
```

### 2. Run the Flask App

After training the model (you should have `generator.h5`), launch the web app:

```bash
python app.py
```

Then open a browser and go to: `http://127.0.0.1:5000/`  
Click the "Generate New Image" button to get new results from the generator.

---

## 🖼 Output

- `images/`: Saved images from training (optional)
- `generator.h5`: Trained generator model
- `discriminator.h5`: Trained discriminator model
- `static/generated.png`: Last generated image from Flask app
- `loss_plot.png`: Training loss plot

Example Flask app output:

![Generated Samples](static/generated.png)

---

## ⏱ Performance Tips

- Enable GPU for faster training (TensorFlow auto-detects CUDA)
- Use TensorBoard to monitor training:
  ```python
  %tensorboard --logdir logs/
  ```
- Use small data splits and epochs to iterate faster when experimenting

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---


## 👤 Author

**Brian Kimutai Siele**  
📧 Email: [korosbrian574@gmail.com](mailto:korosbrian574@gmail.com)  
📌 Location: Nairobi, Kenya
