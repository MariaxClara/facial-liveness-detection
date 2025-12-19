# src/features/head_pose.py
import cv2
import numpy as np

IDX = {
    "nose": 1,
    "chin": 152,
    "left_eye": 33,
    "right_eye": 263,
    "left_mouth": 61,
    "right_mouth": 291,
}

MODEL_3D = np.array([
    (0.0, 0.0, 0.0),        # nose tip
    (0.0, -63.6, -12.5),    # chin
    (-43.3, 32.7, -26.0),   # left eye corner
    (43.3, 32.7, -26.0),    # right eye corner
    (-28.9, -28.9, -24.1),  # left mouth corner
    (28.9, -28.9, -24.1),   # right mouth corner
], dtype=np.float64)


def _lm_px(landmarks, idx, frame_shape):
    h, w, _ = frame_shape
    lm = landmarks.landmark[idx]
    return (lm.x * w, lm.y * h)


def estimate_head_pose_ypr(landmarks, frame_shape):
    h, w, _ = frame_shape

    image_points = np.array([
        _lm_px(landmarks, IDX["nose"], frame_shape),
        _lm_px(landmarks, IDX["chin"], frame_shape),
        _lm_px(landmarks, IDX["left_eye"], frame_shape),
        _lm_px(landmarks, IDX["right_eye"], frame_shape),
        _lm_px(landmarks, IDX["left_mouth"], frame_shape),
        _lm_px(landmarks, IDX["right_mouth"], frame_shape),
    ], dtype=np.float64)

    focal_length = w
    center = (w / 2, h / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)

    dist_coeffs = np.zeros((4, 1))

    success, rvec, tvec = cv2.solvePnP(
        MODEL_3D, image_points, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        return None

    rmat, _ = cv2.Rodrigues(rvec)

    sy = np.sqrt(rmat[0, 0]**2 + rmat[1, 0]**2)
    singular = sy < 1e-6

    if not singular:
        pitch = np.arctan2(rmat[2, 1], rmat[2, 2])
        yaw   = np.arctan2(-rmat[2, 0], sy)
        roll  = np.arctan2(rmat[1, 0], rmat[0, 0])
    else:
        pitch = np.arctan2(-rmat[1, 2], rmat[1, 1])
        yaw   = np.arctan2(-rmat[2, 0], sy)
        roll  = 0.0

    yaw, pitch, roll = np.degrees([yaw, pitch, roll])
    return float(yaw), float(pitch), float(roll)
