
# 🌫️ Adverse Weather Detection using Deep Learning

## 📌 Project Overview

This project focuses on detecting adverse weather conditions from road images using deep learning techniques. The system classifies images into four weather categories:

- 🌫️ Fog
- ❄️ Snow
- 🌪️ Dust
- 🌧️ Rain

The model is trained using the DAWN Weather Dataset and is designed for intelligent transportation systems, smart mobility, and road safety applications.

---

## 🎯 Problem Statement

Adverse weather conditions such as fog, snow, dust, and rain significantly affect road visibility and driving safety. Early detection of such conditions can help:

- Improve driver alert systems
- Enhance autonomous vehicle decision-making
- Reduce accidents caused by low visibility
- Assist smart city monitoring systems

---

## 📂 Dataset

**Dataset Name:** DAWN Weather Dataset  
**Source:** Kaggle  
🔗 https://www.kaggle.com/datasets/shuvoalok/dawn-dataset  

### Dataset Details:
- Weather Categories: Fog, Snow, Dust, Rain
- Real-world road scene images
- Used for multi-class image classification
- Data split into training and validation sets

---

## 🛠️ Technologies Used

- Python
- PyTorch
- Ultralytics YOLO 
- OpenCV
- NumPy
- Matplotlib
- Scikit-learn
- ONNX (for deployment optimization)

---

## 🧠 Model Architecture

The model is trained for multi-class classification of weather conditions. The architecture includes:

- Pre-trained backbone - YOLO 
- Fine-tuned final classification layer (4 classes)
- Cross-Entropy Loss
- Adam Optimizer
## Clone the repository:

```bash
git clone https://github.com/divyapatamsetti/Adverse_Weather_Detection_Using_YOLO.git
cd Adverse_Weather_Detection_Using_YOLO


