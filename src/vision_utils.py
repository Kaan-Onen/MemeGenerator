import math
import cv2
import numpy as np
import mediapipe as mp

def overlay(overlay_image, asset, face_landmarks, anchor_idx, ratio_to_eyes=1.0, anchor_point="middle"):
    if asset is None: return overlay_image
    h, w, _ = overlay_image.shape

    l_eye, r_eye = face_landmarks[33], face_landmarks[263]
    anchor = face_landmarks[anchor_idx]

    ax, ay = int(anchor.x * w), int(anchor.y * h)
    lx, ly = l_eye.x * w, l_eye.y * h
    rx, ry = r_eye.x * w, r_eye.y * h

    eye_dist_px = math.sqrt((rx - lx) ** 2 + (ry - ly) ** 2)
    angle = math.degrees(math.atan2(ry - ly, rx - lx))

    # Resize asset
    target_w = int(eye_dist_px * ratio_to_eyes)
    ah, aw = asset.shape[:2]
    target_h = int(target_w * (ah / aw))
    resized = cv2.resize(asset, (target_w, target_h))

    if anchor_point == "top_left":
        px, py = 0, 0
    elif anchor_point == "top_middle":
        px, py = target_w // 2, 0
    elif anchor_point == "top_right":
        px, py = target_w, 0
    elif anchor_point == "middle_left":
        px, py = 0, target_h // 2
    elif anchor_point == "middle":
        px, py = target_w // 2, target_h // 2
    elif anchor_point == "middle_right":
        px, py = target_w, target_h // 2
    elif anchor_point == "bottom_left":
        px, py = 0, target_h
    elif anchor_point == "bottom_middle":
        px, py = target_w // 2, target_h
    elif anchor_point == "bottom_right":
        px, py = target_w, target_h
    else:
        px, py = target_w // 2, target_h // 2

    M = cv2.getRotationMatrix2D((target_w // 2, target_h // 2), -angle, 1.0)
    cos, sin = np.abs(M[0, 0]), np.abs(M[0, 1])
    nW, nH = int((target_h * sin) + (target_w * cos)), int((target_h * cos) + (target_w * sin))

    M[0, 2] += (nW / 2) - (target_w // 2)
    M[1, 2] += (nH / 2) - (target_h // 2)


    rotated = cv2.warpAffine(resized, M, (nW, nH), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))


    p_rotated = M @ np.array([px, py, 1])
    rot_x, rot_y = int(p_rotated[0]), int(p_rotated[1])


    x1 = ax - rot_x
    y1 = ay - rot_y
    x2, y2 = x1 + nW, y1 + nH


    img_x1, img_x2 = max(x1, 0), min(x2, w)
    img_y1, img_y2 = max(y1, 0), min(y2, h)
    asset_x1, asset_x2 = max(0, -x1), min(nW, w - x1)
    asset_y1, asset_y2 = max(0, -y1), min(nH, h - y1)

    if img_x1 >= img_x2 or img_y1 >= img_y2: return overlay_image

    overlay_part = rotated[asset_y1:asset_y2, asset_x1:asset_x2]
    roi = overlay_image[img_y1:img_y2, img_x1:img_x2]

    if overlay_part.shape[2] == 4:
        alpha = overlay_part[:, :, 3] / 255.0
        for c in range(3):
            roi[:, :, c] = (alpha * overlay_part[:, :, c] + (1 - alpha) * roi[:, :, c])

    return overlay_image

def get_ear(landmarks, eye_indices, w, h):
    # Map indices to (x, y) coordinates
    pts = []
    for idx in eye_indices:
        lm = landmarks[idx]
        pts.append((lm.x * w, lm.y * h))

    # Vertical distances
    v1 = math.dist(pts[1], pts[5])
    v2 = math.dist(pts[2], pts[4])
    # Horizontal distance
    h_dist = math.dist(pts[0], pts[3])

    ear = (v1 + v2) / (2.0 * h_dist)
    return ear

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode