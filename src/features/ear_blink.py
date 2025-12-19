# src/features/ear_blink.py

import numpy as np

LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [263, 387, 385, 362, 380, 373]

def eye_aspect_ratio(landmarks, eye_indices, image_width, image_height):
    """Calcula o EAR de um olho a partir dos landmarks normalizados do MediaPipe."""
    pts = []
    for idx in eye_indices:
        lm = landmarks.landmark[idx]
        x = lm.x * image_width
        y = lm.y * image_height
        pts.append(np.array([x, y]))

    p1, p2, p3, p4, p5, p6 = pts

    dist_v1 = np.linalg.norm(p2 - p6)
    dist_v2 = np.linalg.norm(p3 - p5)

    dist_h = np.linalg.norm(p1 - p4)

    ear = (dist_v1 + dist_v2) / (2.0 * dist_h + 1e-6)  
    return ear


class BlinkDetector:
    def __init__(self, ear_threshold, min_frames_eye_closed, blink_window_frames):
        self.ear_threshold = ear_threshold
        self.min_frames_eye_closed = min_frames_eye_closed
        self.blink_window_frames = blink_window_frames

        self.frames_eye_below_threshold = 0
        self.total_blinks = 0
        self.frame_counter = 0

    def reset_window(self):
        self.total_blinks = 0
        self.frame_counter = 0
        self.frames_eye_below_threshold = 0

    def update(self, landmarks, frame_shape):
        """Atualiza contadores a partir de um frame e retorna EAR médio."""
        h, w, _ = frame_shape

        left_ear = eye_aspect_ratio(landmarks, LEFT_EYE_IDX, w, h)
        right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE_IDX, w, h)
        ear_avg = (left_ear + right_ear) / 2.0

        self.frame_counter += 1

        if ear_avg < self.ear_threshold:
            self.frames_eye_below_threshold += 1
        else:
            if self.frames_eye_below_threshold >= self.min_frames_eye_closed:
                self.total_blinks += 1
            self.frames_eye_below_threshold = 0

        if self.frame_counter >= self.blink_window_frames:
            self.frame_counter = 0
            self.frames_eye_below_threshold = 0

        return ear_avg
