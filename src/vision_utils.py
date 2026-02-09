import math
import cv2
import numpy as np
import mediapipe as mp
from types import SimpleNamespace

# MediaPipe Setup Constants
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

MODEL_PATH = "../models/face_landmarker.task"


def load_assets():
    def load_img(name):
        path = "../assets/" + name
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"Error: {name} not found at {path}")
        return img

    return SimpleNamespace(
        thug_hat=load_img("thug_life_hat.png"),
        thug_glasses=load_img("thug_life_glasses.png"),
        thug_blunt=load_img("thug_life_blunt.png"),
        pirate_hat=load_img("pirate_hat.png"),
        eye_patch=load_img("pirate_patch.png")
    )


def get_outfit(assets, theme_name="thug"):
    themes = {
        "thug": [
            (assets.thug_glasses, SimpleNamespace(index=168, ratio=1.5, point="middle")),
            (assets.thug_hat, SimpleNamespace(index=9, ratio=2.2, point="bottom_middle")),
            (assets.thug_blunt, SimpleNamespace(index=13, ratio=0.9, point="top_right"))
        ],
        "pirate": [
            (assets.pirate_hat, SimpleNamespace(index=9, ratio=2.5, point="bottom_middle")),
            (assets.eye_patch, SimpleNamespace(index=473, ratio=0.6, point="middle"))
        ]
    }
    return themes.get(theme_name.lower(), themes["thug"])


def landmarker_options(running_mode):
    running_mode = running_mode.upper()
    mode = VisionRunningMode.VIDEO if running_mode == "VIDEO" else VisionRunningMode.IMAGE

    return FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=mode
    )


# Geometry & Rendering Functions

def get_anchor_offset(width, height, point_name):
    multipliers = {
        "top_left": (0, 0), "top_middle": (0.5, 0), "top_right": (1, 0),
        "middle_left": (0, 0.5), "middle": (0.5, 0.5), "middle_right": (1, 0.5),
        "bottom_left": (0, 1), "bottom_middle": (0.5, 1), "bottom_right": (1, 1)
    }
    mult_x, mult_y = multipliers.get(point_name, (0.5, 0.5))
    return int(width * mult_x), int(height * mult_y)


def blend_onto_frame(background, overlay_part, x, y):
    bg_h, bg_w = background.shape[:2]
    ov_h, ov_w = overlay_part.shape[:2]

    screen_x1, screen_y1 = max(x, 0), max(y, 0)
    screen_x2, screen_y2 = min(x + ov_w, bg_w), min(y + ov_h, bg_h)

    asset_x1, asset_y1 = max(0, -x), max(0, -y)
    asset_x2 = asset_x1 + (screen_x2 - screen_x1)
    asset_y2 = asset_y1 + (screen_y2 - screen_y1)

    if screen_x1 < screen_x2 and screen_y1 < screen_y2:
        roi = background[screen_y1:screen_y2, screen_x1:screen_x2]
        asset_slice = overlay_part[asset_y1:asset_y2, asset_x1:asset_x2]

        if asset_slice.shape[2] == 4:
            alpha = asset_slice[:, :, 3:4] / 255.0
            color = asset_slice[:, :, :3]
            background[screen_y1:screen_y2, screen_x1:screen_x2] = (alpha * color + (1 - alpha) * roi).astype(np.uint8)
        else:
            background[screen_y1:screen_y2, screen_x1:screen_x2] = asset_slice[:, :, :3]
    return background


def overlay(image, asset, face_landmarks, config):
    if asset is None or face_landmarks is None: return image
    h, w = image.shape[:2]

    # Face Geometry
    left_eye, right_eye = face_landmarks[33], face_landmarks[263]
    anchor_landmark = face_landmarks[config.index]
    target_anchor_px = (int(anchor_landmark.x * w), int(anchor_landmark.y * h))

    dx = (right_eye.x - left_eye.x) * w
    dy = (right_eye.y - left_eye.y) * h
    eye_dist = math.sqrt(dx ** 2 + dy ** 2)
    angle = math.degrees(math.atan2(dy, dx))

    # Prep Asset
    t_w = int(eye_dist * config.ratio)
    t_h = int(t_w * (asset.shape[0] / asset.shape[1]))
    resized = cv2.resize(asset, (t_w, t_h))

    # Rotation
    center = (t_w // 2, t_h // 2)
    M = cv2.getRotationMatrix2D(center, -angle, 1.0)
    cos, sin = np.abs(M[0, 0]), np.abs(M[0, 1])
    b_w, b_h = int((t_h * sin) + (t_w * cos)), int((t_h * cos) + (t_w * sin))
    M[0, 2] += (b_w / 2) - center[0]
    M[1, 2] += (b_h / 2) - center[1]

    rotated = cv2.warpAffine(resized, M, (b_w, b_h), borderMode=cv2.BORDER_CONSTANT)

    # Final Alignment
    lx, ly = get_anchor_offset(t_w, t_h, config.point)
    rot_anchor = M @ np.array([lx, ly, 1])

    tx = target_anchor_px[0] - int(rot_anchor[0])
    ty = target_anchor_px[1] - int(rot_anchor[1])

    return blend_onto_frame(image, rotated, tx, ty)


def get_ear(landmarks, eye_indices, w, h):
    pts = [(landmarks[idx].x * w, landmarks[idx].y * h) for idx in eye_indices]
    v1 = math.dist(pts[1], pts[5])
    v2 = math.dist(pts[2], pts[4])
    h_dist = math.dist(pts[0], pts[3])
    return (v1 + v2) / (2.0 * h_dist)