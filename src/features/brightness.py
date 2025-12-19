# src/features/brightness.py

import cv2
import numpy as np
from collections import deque


class BrightnessLiveness:
    """
    Classe para analisar o padrão de brilho do rosto ao longo do tempo.

    Ideia:
    - Para cada frame, recortamos a região do rosto (ROI) usando os landmarks.
    - Convertamos o ROI para escala de cinza.
    - Calculamos:
        * média do brilho (mean)
        * desvio padrão do brilho (std)
    - Mantemos um histórico (deque) desses valores nos últimos N frames.
    - Calculamos a VARIAÇÃO (variância) dessa média e desse desvio.

    - Se a variação for muito baixa -> pode ser algo muito estático (ex.: foto).
    - Se há uma variação "razoável" -> mais provável ser um rosto real (3D, micro-movimentos, textura).
    """

    def __init__(
        self,
        window_size=60,
        min_frames_for_decision=20,
        var_mean_min=3.0,
        var_std_min=1.0,
    ):

        self.window_size = window_size
        self.min_frames_for_decision = min_frames_for_decision
        self.var_mean_min = var_mean_min
        self.var_std_min = var_std_min

        self.mean_history = deque(maxlen=window_size)
        self.std_history = deque(maxlen=window_size)

    def _extract_face_roi_gray(self, frame_bgr, landmarks):
        h, w, _ = frame_bgr.shape

        xs = []
        ys = []
        for lm in landmarks.landmark:
            xs.append(int(lm.x * w))
            ys.append(int(lm.y * h))

        min_x, max_x = max(min(xs), 0), min(max(xs), w - 1)
        min_y, max_y = max(min(ys), 0), min(max(ys), h - 1)

        if max_x <= min_x or max_y <= min_y:
            return None

        roi = frame_bgr[min_y:max_y, min_x:max_x]

        if roi.size == 0:
            return None

        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        return roi_gray

    def update(self, frame_bgr, landmarks):
        roi_gray = self._extract_face_roi_gray(frame_bgr, landmarks)

        if roi_gray is None:
            return {
                "mean": None,
                "std": None,
                "var_mean": None,
                "var_std": None,
                "live_like": None,
            }

        roi_float = roi_gray.astype(np.float32)

        mean_val = float(np.mean(roi_float))
        std_val = float(np.std(roi_float))

        self.mean_history.append(mean_val)
        self.std_history.append(std_val)

        var_mean = None
        var_std = None
        live_like = None

        if len(self.mean_history) >= self.min_frames_for_decision:
            var_mean = float(np.var(self.mean_history))
            var_std = float(np.var(self.std_history))

            if (var_mean >= self.var_mean_min) or (var_std >= self.var_std_min):
                live_like = True
            else:
                live_like = False

        return {
            "mean": mean_val,
            "std": std_val,
            "var_mean": var_mean,
            "var_std": var_std,
            "live_like": live_like,
        }
