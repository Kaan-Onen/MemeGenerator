from vision_utils import *


model_path = "../models/face_landmarker.task"

#assets import
hat = cv2.imread("../assets/thug_life_hat.png", cv2.IMREAD_UNCHANGED)
glasses = cv2.imread("../assets/thug_life_glasses4.png", cv2.IMREAD_UNCHANGED)
blunt = cv2.imread("../assets/thug_life_blunt.png", cv2.IMREAD_UNCHANGED)

#landmarks run on video
options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO
)

#get video and video's fps
cap = cv2.VideoCapture("../assets/video3.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
delay = int(1000 / fps) if fps > 0 else 33

#create landmark model
with FaceLandmarker.create_from_options(options) as landmarker:

        while cap.isOpened():
            ret, frame = cap.read()

            if not ret: break

            #take frames in the video
            h, w, _ = frame.shape
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.face_landmarks:
                #take the first detected face
                face = result.face_landmarks[0]

                #take eyes' landmarks and compute each eyes' ear
                left_ear = get_ear(face, [362, 385, 386, 263, 374, 380], w, h)
                right_ear = get_ear(face, [33, 160, 158, 133, 153, 144], w, h)

                avg_ear = (left_ear + right_ear) / 2.0

                #treshold's for eye's ear
                status_average = "Closed" if avg_ear < 0.21 else "Open"
                status_left = "Closed" if left_ear < 0.21 else "Open"
                status_right = "Closed" if right_ear < 0.21 else "Open"

                #how to trigger trigger
                trigger = False
                if status_average == "Closed" and not trigger:
                    trigger = True
                elif status_average == "Open" and  trigger:
                    trigger = False

                #overlays' attributes
                if trigger:
                    frame = overlay(frame, glasses, face, 168, ratio_to_eyes=1.5, anchor_point="middle")

                    frame = overlay(frame, hat, face, 9, ratio_to_eyes=2.2, anchor_point="bottom_middle")

                    frame = overlay(frame, blunt, face, 13, ratio_to_eyes=0.9, anchor_point="top_right")

            #make the seen frames have a certain size and the window's label
            frame = cv2.resize(frame, (1020, 680))
            cv2.imshow('Landmark Debugger', frame)
            if cv2.waitKey(delay) & 0xFF == ord('q'):
                break

cap.release()
cv2.destroyAllWindows()