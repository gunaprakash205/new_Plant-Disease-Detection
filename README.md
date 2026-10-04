# 🌿 Plant Disease Detection

A deep learning-based web application that detects plant diseases from leaf images using **EfficientNetB0** and **Streamlit**.

## 🚀 Features

* 🌱 Detects diseases from leaf images
* 🍎 Supports **Apple**
* 🌽 Supports **Corn (Maize)**
* 🍇 Supports **Grape**
* 🥔 Supports **Potato**
* 🍅 Supports **Tomato**
* 🧠 Uses EfficientNetB0 transfer learning
* 📊 25 plant disease/healthy classes
* 🌐 Streamlit web interface
* 📷 Upload JPG, JPEG, or PNG images
* 🔝 Displays top predictions with confidence scores

## 📊 Model Performance

The model was trained on a custom dataset containing **12,000 images**.

| Dataset    | Images |
| ---------- | -----: |
| Training   |  8,400 |
| Validation |  1,800 |
| Testing    |  1,800 |
| Total      | 12,000 |

### Test Accuracy

**95.67%**

The model uses **EfficientNetB0** pretrained on ImageNet with transfer learning and fine-tuning.

## 🧠 Model

* Architecture: **EfficientNetB0**
* Input Size: **224 × 224**
* Optimizer: **Adam**
* Loss: **Sparse Categorical Crossentropy**
* Output: **25 classes**
* Framework: **TensorFlow / Keras**

## 🛠️ Technologies Used

* Python
* TensorFlow / Keras
* NumPy
* Pillow
* Streamlit
* Git & Git LFS

## 📁 Project Structure

```text
new_Plant-Disease-Detection/
│
├── app.py
├── plant_disease_efficientnetb0_final.keras
├── class_names.json
├── requirements.txt
└── README.md
```

## ▶️ Run Locally

Clone the repository:

```bash
git clone https://github.com/gunaprakash205/new_Plant-Disease-Detection.git
```

Go to the project folder:

```bash
cd new_Plant-Disease-Detection
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

## 📷 How It Works

1. Upload a leaf image.
2. The image is resized to **224 × 224**.
3. EfficientNetB0 processes the image.
4. The model predicts the most likely class.
5. The application displays:

   * Plant name
   * Disease name
   * Confidence score
   * Top 5 predictions

## ⚠️ Note

The model may sometimes produce a correct prediction with relatively low confidence, especially for real-world images with different lighting, backgrounds, angles, or image quality.

The application displays the model's confidence score so users can understand the uncertainty of each prediction.

## 🌐 Deployment

The application is designed to be deployed using **Streamlit Community Cloud**.

## 👨‍💻 Author

**Guna Prakash**

B.Tech Computer Science Engineering
2027 Graduate

## 📌 Project Goal

The goal of this project is to demonstrate the practical use of **deep learning and transfer learning for automated plant disease detection** through an easy-to-use web application.
