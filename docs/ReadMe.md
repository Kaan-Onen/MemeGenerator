# Meme Generator

A professional-grade computer vision application designed to automate the overlay of memes graphical assets onto human faces. Uses the **MediaPipe FaceLandmarker** framework, the system ensures high-precision placement through real-time landmark tracking and geometric transformations.

---

## 🛠 Core Features

### Static Image Processing
The system analyzes static images to determine facial orientation. Assets are dynamically scaled and rotated based on the distance and the angle of the eyes to ensure a natural fit.

### Real-Time Video & Blink Detection
Utilizing a temporal analysis of facial landmarks, the application detects blink events in video streams. A detected blink serves as the trigger mechanism to initiate meme overlay sequence.

### Precision Anchoring & Rotation
Assets are anchored to specific facial indices provided by the MediaPipe mesh. The system calculates the precise rotation matrix using the following geometric relationship:

$$\theta = \arctan2(y_{eye\_right} - y_{eye\_left}, x_{eye\_right} - x_{eye\_left})$$

### Advanced Alpha Blending
To ensure visual fidelity, the software utilizes the alpha channel of PNG assets. This allows for transparency and professional-grade compositing over the source frames.

---

## 💻 Technical Stack

* **Language:** Python 3.9+  
* **Face Tracking:** MediaPipe FaceLandmarker  
* **Image Processing:** OpenCV (Open Source Computer Vision Library)  
* **Mathematical Operations:** NumPy (Matrix transformations and coordinate geometry)  

---

## 📂 Project Structure


├── .venv/                # Virtual environment for dependency isolation  
├── assets/               # Source assets (transparent PNGs, images, and videos)  
├── docs/                 # Technical documentation and specifications  
├── models/               # Pre-trained MediaPipe task models  
├── src/                  # Source code directory  
│   ├── image_proc.py     # Static image processing logic  
│   ├── video_proc.py     # Real-time video and blink detection logic  
│   └── vision_utils.py   # Coordinate transformation and rendering utilities  
└── requirements.txt      # List of mandatory Python dependencies  


🚀 Implementation Guide  
1. Environment Configuration  
Install the required dependencies using the Python package manager:

Bash  
pip install -r requirements.txt  

2. Model and Asset Integration  
Download the face_landmarker.task file and place it within the models/ directory.

Ensure all graphical assets (e.g., eyewear, headwear) are stored in the assets/ directory with appropriate transparency layers.

3. Execution
To process a static image:

Bash  
python src/image.py  
To initialize the video stream with blink detection:

Bash  
python src/video.py
