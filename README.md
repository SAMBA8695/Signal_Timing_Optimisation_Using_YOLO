# 🚦 Intelligent Vehicle / Traffic Detection using YOLO

This project implements a **vehicle detection system** using the **YOLO (You Only Look Once)** object detection framework powered by **Ultralytics**.  
It covers **dataset preparation, class remapping, model training, and inference/testing** on images.

The project is designed to run smoothly in **Google Colab** with datasets stored on **Google Drive**.

---

## 📌 Features

- YOLO-based object detection using Ultralytics  
- Dataset class remapping for customized vehicle categories  
- Automated training configuration generation  
- Image-based inference and visualization  
- Supports batch testing on folders of images  
- Easy integration with Google Drive  

---

## 🧠 Tech Stack

- Python  
- Ultralytics YOLO  
- OpenCV  
- NumPy  
- Matplotlib  
- Google Colab  

---

## 📂 Project Structure

.
├── IFP.ipynb # Main notebook (training + testing)
├── data/
│ ├── images/
│ ├── labels/
│ └── dataset.yaml
├── runs/ # YOLO training outputs
└── README.md

yaml
Copy code

---

## 🚀 Setup & Installation

### 1️⃣ Open in Google Colab
Upload `IFP.ipynb` to Colab or open it directly from GitHub.

### 2️⃣ Mount Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```
3️⃣ Install Dependencies
```python
pip install ultralytics
```
🗂 Dataset Preparation
Supports remapping original dataset classes into a reduced/custom set

Automatically generates a YOLO-compatible dataset.yaml file

Reorganizes images and labels for training and validation

🏋️ Model Training
Uses YOLOv8 via the Ultralytics API

Training parameters (epochs, batch size, image size) are configurable

Training outputs are saved under:

runs/detect/
🔍 Model Testing & Inference
Inference can be run on:

Single images

Entire folders of images

Bounding boxes and class labels are displayed directly in Colab.

Example:

```python
test_all_images_in_folder(
    '/content/drive/MyDrive/test-vehicles',
    max_images=20
)
```
📊 Output
Detected vehicles visualized with bounding boxes

Results include:

Class name

Confidence score

Useful for traffic analysis and smart transportation systems.

💡 Use Cases
Traffic monitoring systems

Smart city applications

Vehicle counting and classification

Edge AI and computer vision projects

📜 License
This project is intended for educational and research purposes.

yaml
Copy code
