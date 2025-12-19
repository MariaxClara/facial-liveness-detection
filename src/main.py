# src/main.py

import cv2

import config
from detection.face_and_landmarks import FaceAndLandmarksDetector
from features.ear_blink import BlinkDetector, LEFT_EYE_IDX, RIGHT_EYE_IDX
from features.brightness import BrightnessLiveness
from liveness.rules_blink_only import BlinkOnlyLivenessRule
from liveness.rules_blink_brightness import BlinkBrightnessLivenessRule
from utils.video_io import open_webcam


def draw_text(img, text, org, color=(0, 255, 0)):
    cv2.putText(
        img,
        text,
        org,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
        cv2.LINE_AA
    )


def get_landmark_pixel(landmarks, idx, frame_shape):
    """Converte um índice de landmark do MediaPipe em coordenadas de pixel (x, y)."""
    h, w, _ = frame_shape
    lm = landmarks.landmark[idx]
    x = int(lm.x * w)
    y = int(lm.y * h)
    return x, y


def draw_face_bbox(frame, landmarks):
    h, w, _ = frame.shape

    xs = []
    ys = []
    for lm in landmarks.landmark:
        xs.append(int(lm.x * w))
        ys.append(int(lm.y * h))

    min_x, max_x = max(min(xs), 0), min(max(xs), w - 1)
    min_y, max_y = max(min(ys), 0), min(max(ys), h - 1)

    cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)


def draw_eye_points(frame, landmarks):
    for idx in LEFT_EYE_IDX + RIGHT_EYE_IDX:
        x, y = get_landmark_pixel(landmarks, idx, frame.shape)
        cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)


def main():
    cap = open_webcam(0)
    detector = FaceAndLandmarksDetector()

    # Sensores
    blink_detector = BlinkDetector(
        ear_threshold=config.EAR_THRESHOLD,
        min_frames_eye_closed=config.MIN_FRAMES_EYE_CLOSED,
        blink_window_frames=config.BLINK_WINDOW_FRAMES,
    )

    brightness_liveness = BrightnessLiveness(
        window_size=60,
        min_frames_for_decision=20,  
        var_mean_min=0.5,            
        var_std_min=0.2,  
    )

    rule_blink_only = BlinkOnlyLivenessRule(
        min_blinks_for_liveness=config.MIN_BLINKS_FOR_LIVENESS
    )

    rule_blink_bright = BlinkBrightnessLivenessRule(
        min_blinks_for_liveness=config.MIN_BLINKS_FOR_LIVENESS,
        require_brightness_live_like=True,
    )

    print("Pressione 'q' para sair.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        landmarks = detector.process_frame(frame)

        if landmarks is not None:
            # Atualiza sensores
            ear = blink_detector.update(landmarks, frame.shape)
            bright_info = brightness_liveness.update(frame, landmarks)

            # Desenho visual (rosto + olhos)
            draw_face_bbox(frame, landmarks)
            draw_eye_points(frame, landmarks)

            # --- Informações básicas na tela ---
            draw_text(frame, f"EAR: {ear:.3f}", (10, 30))
            draw_text(frame, f"Blinks: {blink_detector.total_blinks}", (10, 60))

            # Brilho: média e variância (se já tiver histórico suficiente)
            if bright_info["mean"] is not None:
                draw_text(frame, f"BrightMean: {bright_info['mean']:.1f}", (10, 90))
            if bright_info["var_mean"] is not None:
                draw_text(frame, f"VarMean: {bright_info['var_mean']:.2f}", (10, 120))

            # --- Decisão Método A: apenas blink ---
            decision_a = rule_blink_only.decide(blink_detector)

            # --- Decisão Método B: blink + brilho ---
            decision_b = rule_blink_bright.decide(blink_detector, bright_info)

            # Mostrar decisão A
            if decision_a["live"]:
                draw_text(
                    frame,
                    f"LIVENESS A (Blink): {decision_a['reason']}",
                    (10, 160),
                    color=(0, 255, 0),
                )
            else:
                draw_text(
                    frame,
                    f"LIVENESS A (Blink): {decision_a['reason']}",
                    (10, 160),
                    color=(0, 255, 255),
                )

            # Mostrar decisão B
            if decision_b["live"]:
                draw_text(
                    frame,
                    f"LIVENESS B (Blink+Bright): {decision_b['reason']}",
                    (10, 190),
                    color=(0, 255, 0),
                )
            else:
                draw_text(
                    frame,
                    f"LIVENESS B (Blink+Bright): {decision_b['reason']}",
                    (10, 190),
                    color=(0, 255, 255),
                )

        else:
            draw_text(frame, "Nenhum rosto detectado", (10, 30), color=(0, 0, 255))

        cv2.imshow("Face Liveness Demo (Blink + Brightness)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
