# sign-language
# 🤟 Real-Time ASL Alphabet Recognition (A-Z)

## 📌 Project Overview

This project implements a high-accuracy, real-time American Sign Language (ASL) fingerspelling detection system. It leverages **Google's MediaPipe Hands** library to extract 3D skeleton keypoints from the user's hand gestures and uses a custom-trained **Deep Neural Network (DNN) model (Keras/TensorFlow)** for classification.

The system achieves **99.00% accuracy** on the static ASL alphabet (A-Z) and provides immediate, on-screen prediction via a webcam feed.

## ✨ Features

* **Real-Time Detection:** Live classification of all 26 ASL alphabet signs.
* **Feature Engineering:** Utilizes 126 normalized 3D hand keypoints (2 hands x 21 landmarks x 3 coordinates) extracted by MediaPipe. 
* **High Accuracy:** Model trained using Keras achieves **99.00%** test accuracy.
* **Temporal Smoothing:** Implements a simple **majority vote buffer** to stabilize predictions and reduce flickering between similar signs (like 'M' and 'N').

---

## 🚀 Getting Started

Follow these steps to set up and run the real-time detector on your local machine.

### 📋 Prerequisites

Ensure you have **Python 3.8+** installed.

### 💾 Installation

1.  **Clone the Repository (If on GitHub):**
    ```bash
    git clone [YOUR_REPOSITORY_URL]
    cd [YOUR_REPOSITORY_NAME]
    ```

2.  **Install Dependencies:**
    ```bash
    pip install opencv-python mediapipe tensorflow numpy
    ```

3.  **Model Setup:**
    * **Download the Trained Model:** The trained model file, `best_static_asl_model.h5`, must be downloaded from the training environment (Google Drive) and placed directly into the project's root directory.

---

## 💡 Usage

The primary file for real-time demonstration is `real_time_predictor.py`.

### 1. Run the Detector

Open your terminal in the project directory and execute the script:

```bash
python real_time_predictor.py
