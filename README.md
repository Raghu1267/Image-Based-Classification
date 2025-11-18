# 🐄 Cattle–Buffalo Image Classification Web App

This project is a **Streamlit-based web app** that classifies images as **Cattle** or **Buffalo** using a **custom-trained YOLOv8 classification model**.  
The entire application is deployed on **Vercel using Docker**, allowing smooth execution of OpenCV, PyTorch, and Ultralytics.

---

## 🚀 Live Demo
👉 **https://YOUR-VERCEL-LINK.vercel.app**

*(Replace with your actual deployed URL)*

---

## 🌟 Features

- ✔ YOLOv8 custom-trained classification model  
- ✔ Upload an image and get instant predictions  
- ✔ Confidence score for each prediction  
- ✔ Google Drive-based model loading (no file size limits)  
- ✔ Premium UI with glass effect and gradient hero section  
- ✔ Docker container deployment on Vercel  
- ✔ Mobile-responsive clean interface  

---

## 🧠 How It Works

1. User uploads an image  
2. The web app loads the YOLOv8 model (downloaded once from Google Drive)  
3. The model predicts whether the image is **Cattle** or **Buffalo**  
4. Streamlit displays the result with confidence score  

The model download happens only on first run and is cached afterwards.

---

## 🏗️ Project Structure

