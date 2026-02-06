from vision_utils import *

options = landmarker_options("Video")
cap = cv2.VideoCapture("../assets/video1.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
delay = int(1000 / fps) if fps > 0 else 33

# State variable for the trigger (must be outside the while loop)
is_meme_active = False

with FaceLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        h, w, _ = frame.shape
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        if result.face_landmarks:
            face = result.face_landmarks[0]

            # Eye Aspect Ratio (EAR) calculation
            left_ear = get_ear(face, [362, 385, 386, 263, 374, 380], w, h)
            right_ear = get_ear(face, [33, 160, 158, 133, 153, 144], w, h)
            avg_ear = (left_ear + right_ear) / 2.0

            # Toggle the meme on when eyes close (< 0.21)
            # Toggle it off when eyes open (> 0.21)
            if avg_ear > 0.21:
                is_meme_active = True
            else:
                is_meme_active = False

            # Apply the outfit if triggered
            if is_meme_active:
                for asset, cfg in get_outfit(load_assets(), "pirate"):
                    frame = overlay(frame, asset, face, cfg)

        # UI and Display
        display_frame = cv2.resize(frame, (1020, 680))
        cv2.imshow('Meme Generator Video', display_frame)

        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()