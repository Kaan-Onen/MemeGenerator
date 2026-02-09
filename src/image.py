from vision_utils import *


# Load the image
raw_frame = cv2.imread("../assets/face2.png")
rf_h, rf_w = raw_frame.shape[:2]
frame = cv2.resize(raw_frame, (256, 256))

# Setup Landmarker
options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.IMAGE
)

# Process and Overlay
with FaceLandmarker.create_from_options(options) as landmarker:
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    result = landmarker.detect(mp_image)

    if result.face_landmarks:
        face = result.face_landmarks[0]

        # This loop applies all items in the 'outfit' list automatically
        for asset, config in get_outfit(load_assets() ,theme_name="pirate"):
            frame = overlay(frame, asset, face, config)
    else:
        print("No face detected.")

# Display Result
real_frame = cv2.resize(frame, (1000, 1000))
cv2.imshow("Meme Generator", real_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()