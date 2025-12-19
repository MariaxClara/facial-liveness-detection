# src/utils/video_io.py

import cv2

def open_webcam(index=0):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError("Não foi possível abrir a webcam.")
    return cap
