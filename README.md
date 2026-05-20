# HexSoftwares_Face_Detection

# 🎯 Face Detection using OpenCV in Python

An AI-powered **Face Detection** application built using **OpenCV's Haar Cascade Classifier** as part of the **Hex Softwares AI Internship Program (Task 2)**. The app detects human faces in real-time from images or live webcam with a fully interactive dark-themed GUI.

---

## 🤖 What is this Project?

**FaceDetectAI** is a desktop application that uses Computer Vision to detect and highlight human faces in images or live video streams — deployed as a complete GUI application with a modern dark-themed interface built in Python.

---

## 💬 Features

- 🖼️ **Image Upload** — Load any JPG, PNG, BMP or WEBP image for instant face detection
- 📷 **Live Webcam** — Real-time face detection via webcam stream
- 👁️ **Eye Detection** — Optional eye detection inside each detected face
- 🎛️ **Adjustable Settings** — Scale Factor & Min Neighbours sliders for fine-tuning accuracy
- 💾 **Save Results** — Export the detected face result as an image
- 📊 **Live Stats** — Real-time face count shown on screen
- 🌙 **Dark Mode GUI** — Modern dark-themed desktop interface

---

## 🛠️ Technologies Used

- Python 3.8+
- OpenCV (cv2) — Haar Cascade Face Detection
- Tkinter — GUI Framework
- Pillow (PIL) — Image Processing & Display
- Threading — Smooth webcam streaming

---

## 🔄 How it Works

```
Input (Image / Webcam)
        ↓
Convert to Grayscale
        ↓
Histogram Equalization (better low-light performance)
        ↓
Haar Cascade Classifier scans for face patterns
        ↓
Bounding boxes drawn around detected faces
        ↓
Result displayed in GUI with live face count
```

---

## 🚀 Getting Started

**1. Clone the repository**
```bash
git clone https://github.com/Abdullahgulzar/HexSoftwares_Face_Detection.git
cd HexSoftwares_Face_Detection
```

**2. Install required libraries**
```bash
pip install opencv-python pillow
```

**3. Run the application**
```bash
python face_detection_gui.py
```

---

## ⚙️ Settings Guide

| Setting | Recommended Value | Effect |
|---|---|---|
| **Scale Factor** | 1.2 | Controls detection sensitivity |
| **Min Neighbours** | 7–8 | Higher = fewer false positives |

> 💡 **Tip:** If too many false detections appear, increase Min Neighbours to 8–9

---

## 📁 Project Structure

```
HexSoftwares_Face_Detection/
│
├── face_detection_gui.py    # Main application
├── requirements.txt         # Dependencies
└── README.md                # Documentation
```

---

---

## 👤 Author

**Shaikh Abdullah Siddique**
AI Intern — Hex Softwares Pvt. Ltd.

