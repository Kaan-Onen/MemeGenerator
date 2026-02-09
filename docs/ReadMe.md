Meme Generator
A computer vision project that automatically overlays "thug life" assets onto faces using mediapipe face landmarker.
Can be used to make thug life meme to an image or video with blink detection trigger.

Features:
Static Image Overlay: Processes image and scales and rotates assets based on the distance and angle between eyes.

Video Overlay: Processes video and when user closes their eyes "thug life meme" will apper on the user.

Smart Anchoring: Assets placed based on specific landmarks on the face.

Rotation: Based on distance between eyes and angel between eyes the assets placed on face even if the face is tilted ensuring assets fit perfect.

Uses Python

Core libraries: Mediapipe for face landmarks, OpenCV for image and video manipulation and NumPy for coordinates and matrix transformations.

Project Structure:
.venv: Virtual environments
assets: Thug life photos, images and videos
docs: Documents
models: landmark models
src: .py files -> scripts for videos and images and their utils
requirements.txt: List of necessary Python dependencies.

How it Works:
The project uses FaceLandmarker to identify face points.

The overlay function in vision_utils.py performs the following:
Scaling: Resizes the asset based on the distance between the user's eyes.

Rotation: Rotates the asset to match the tilt of the eyes using cv2.getRotationMatrix2D.

Alpha Blending: Uses the PNG alpha channel to ensure seamless transparency for the assets.

Getting Started:

1- Install Dependencies:
pip install -r requirements.txt

2- Models and Assets:
Ensure the MediaPipe model face_landmarker.task is in the models/ folder.

Place your assets (glasses, hat, blunt), face images and the videos in the assets/ folder.

Run the Scripts:

For images: python image.py

For video: python video.py
