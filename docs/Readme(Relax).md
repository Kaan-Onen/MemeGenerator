# 😎 The Thug Life Meme Generator 🕶️

Welcome to the ultimate **Computer Vision** project! This tool uses cutting-edge tech to automatically drop "Thug Life" assets onto faces with pinpoint precision. Whether it is a boss-level static photo or a video that triggers on a blink, this Python-powered engine has you covered. ✨

---

## 🚀 Key Features

* **🖼️ Static Image Overlay** Automatically processes images to scale and rotate assets based on the distance and angle between the eyes. No manual tweaking needed!

* **🎥 Video Magic** Real-time processing that detects when you **blink**. Close your eyes, and *boom*—the meme appears.

* **📍 Smart Anchoring** Assets are locked to specific facial landmarks, ensuring they don't just float around or look out of place.

* **🔄 Dynamic Rotation** Whether your head is tilted or you are leaning back, the assets adjust their angle using eye-coordinate math for a perfect fit.

---

## 🛠️ The Tech Stack

This project is built with **Python** and a trio of powerhouse libraries:

1. **MediaPipe:** The "eyes" of the project, used for high-fidelity face landmarking. 🧐
2. **OpenCV:** The engine for image and video manipulation. 🎬
3. **NumPy:** The "brain" handling all the coordinate geometry and matrix transformations. 🔢

---

## 📂 Project Structure

├── .venv/               # Isolated virtual   
├── assets/              # Thug life PNGs (glasses, hat, blunt) and media   
├── docs/                # Documentation and project guides  
├── models/              # The face_landmarker.task file 🤖  
├── src/                 # Core logic  
│   ├── image.py         # Script for processing static images 📸  
│   ├── video.py         # Script for video & blink detection 📹  
│   └── vision_utils.py  # Utility functions for scaling/rotation  
└── requirements.txt     # The "shopping list" of dependencies  

🎬 Getting Started
1. Install Dependencies
Get your environment ready with one command: pip install -r requirements.txt 💻  

2. Models and Assets
Drop the face_landmarker.task model into the models/ folder.

Throw your swagger (glasses, hats, etc.) and your test videos into the assets/ folder. 📂  

3. Run the Scripts  
For Photos: python src/image.py 📸  

For Videos: python src/video.py 📹
