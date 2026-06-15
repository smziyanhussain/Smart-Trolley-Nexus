SmartTrolley Nexus – AI-Powered Smart Shopping Cart
📌 Project Overview

SmartTrolley Nexus is a Final Year Project (FYP) developed to automate the shopping and billing process using Artificial Intelligence and Computer Vision. The system uses a camera and machine learning techniques to identify products placed inside a shopping trolley and automatically generate a bill.

The primary objective of this project is to reduce checkout time, eliminate long queues, and provide a smart shopping experience for customers.

🎯 Features
Product recognition using Computer Vision
Automatic product billing
Real-time image capture through camera
Product database management
Add new products dynamically
Automatic price retrieval for known products
User-friendly interface
TensorFlow-based image classification
Raspberry Pi compatible implementation
🛠 Technologies Used
Python 3.7+
TensorFlow / Keras
OpenCV
NumPy
Pandas
Raspberry Pi 3
Pi Camera / USB Camera
CSV Database
📂 Project Structure
SmartTrolleyNexus/
│
├── hope.py                  # Product recognition and billing
├── capture.py               # Product image capture
├── model.h5                 # Trained AI model
├── model.tflite             # TensorFlow Lite model
├── labels.txt               # Product labels
├── product_db.csv           # Product database with prices
│
├── dataset/
│   ├── Product1/
│   ├── Product2/
│   └── ProductN/
│
├── captured_images/
│
├── requirements.txt
└── README.md
⚙️ System Workflow
Customer places a product inside the trolley.
Camera captures the product image.
TensorFlow model identifies the product.
Product information is retrieved from the database.
Product price is added to the bill.
If the product is not found:
User is prompted to add product information.
Product image and price are stored for future recognition.
Updated bill is displayed automatically.
🧠 Machine Learning Model

The project utilizes a Convolutional Neural Network (CNN) trained on product images.

Training Process
Collect product images.
Organize images into class folders.
Train CNN model using TensorFlow/Keras.
Export trained model as:
model.h5

for Raspberry Pi deployment.

💻 Installation
Clone Repository
git clone https://github.com/yourusername/SmartTrolleyNexus.git
cd SmartTrolleyNexus
Create Virtual Environment
python -m venv venv
Activate Virtual Environment
Windows
venv\Scripts\activate
Linux/Mac
source venv/bin/activate
Install Dependencies
pip install -r requirements.txt
▶ Running the Project
Step 1 – Capture Product Images
python capture.py
Step 2 – Run Product Recognition
python hope.py
📋 Requirements

Create a requirements.txt file containing:

tensorflow
opencv-python
numpy
pandas
keras
Pillow

Install using:

pip install -r requirements.txt
🔧 Hardware Requirements
Raspberry Pi 3
Raspberry Pi Camera Module or USB Camera
LCD Display (Optional)
Weight Sensor (Optional Future Enhancement)
Power Supply
SD Card (16GB+)
📈 Future Enhancements
Barcode Integration
RFID-based Product Tracking
Weight Sensor Verification
Mobile Application Integration
Online Payment Gateway
Cloud Database Synchronization
Real-time Inventory Management
👨‍💻 Team Members
SmartTrolley Nexus
Syed Muhammad Ziyan Hussain
Muhammad Umar
Muneeb ur Rehman
Department

Computer Engineering

University

Mohammad Ali Jinnah University

📖 Research Objective

The purpose of SmartTrolley Nexus is to develop an intelligent shopping cart capable of recognizing products automatically and generating bills without requiring manual barcode scanning. The system aims to improve customer experience, reduce checkout delays, and introduce AI-driven automation into retail environments.

📄 License

This project is developed for academic and educational purposes as a Final Year Project (FYP).
