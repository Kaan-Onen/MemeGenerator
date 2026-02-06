import cv2
import mediapipe as mp
from vision_utils import overlay
from vision_utils import BaseOptions,FaceLandmarker, FaceLandmarkerOptions,VisionRunningMode

#take the assets and models
model_path: str = "../models/face_landmarker.task"
raw_frame = cv2.imread("../assets/face5.png")
hat = cv2.imread("../assets/thug_life_hat.png", cv2.IMREAD_UNCHANGED)
glasses = cv2.imread("../assets/thug_life_glasses4.png", cv2.IMREAD_UNCHANGED)
blunt = cv2.imread("../assets/thug_life_blunt.png", cv2.IMREAD_UNCHANGED)

#take the frame's original size and make the image 256x256 to get better landmark position
rf_h, rf_w = raw_frame.shape[:2]
frame = cv2.resize(raw_frame, (256, 256))

#land mark runs on image
options = FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=model_path),running_mode=VisionRunningMode.IMAGE)

#create landmark model
with FaceLandmarker.create_from_options(options) as landmarker:
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    result = landmarker.detect(mp_image)

    #take the first face
    if result.face_landmarks:
        face = result.face_landmarks[0]
        #overlays' attributes
        frame = overlay(frame, glasses, face, 168, ratio_to_eyes=1.5, anchor_point="middle")

        frame = overlay(frame, hat, face, 9, ratio_to_eyes=2.2, anchor_point="bottom_middle")

        frame = overlay(frame, blunt, face, 13, ratio_to_eyes=0.9, anchor_point="top_right")

    else:
        print("No face detected.")
#show image after the overlays
real_frame = cv2.resize(frame, (1000,1000))
cv2.imshow(f"Meme Generator", real_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()