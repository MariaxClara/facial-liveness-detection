# src/liveness/rules_head_pose_challenge.py

class HeadPoseChallengeRule:

    def __init__(self, yaw_thr=15.0, center_thr=7.0, hold_frames=8, assume_selfie=False):
        self.yaw_thr = float(yaw_thr)
        self.center_thr = float(center_thr)
        self.hold_frames = int(hold_frames)

        self.assume_selfie = bool(assume_selfie)

        self.steps = [
            ("Vire para a ESQUERDA", "LEFT"),
            ("Vire para a DIREITA", "RIGHT"),
            ("Olhe para a FRENTE", "CENTER"),
        ]
        self.step_i = 0
        self._hold = 0
        self._done = False

    def reset(self):
        self.step_i = 0
        self._hold = 0
        self._done = False

    def _check(self, yaw_deg, mode):
        if yaw_deg is None:
            return False

        if mode == "CENTER":
            return abs(yaw_deg) < self.center_thr

        if not self.assume_selfie:
            if mode == "LEFT":
                return yaw_deg < -self.yaw_thr
            if mode == "RIGHT":
                return yaw_deg > self.yaw_thr
        else:
            # invertido
            if mode == "LEFT":
                return yaw_deg > self.yaw_thr
            if mode == "RIGHT":
                return yaw_deg < -self.yaw_thr

        return False

    def decide(self, yaw_deg):
        if self._done:
            return {
                "live": True,
                "reason": "Head pose challenge completo",
                "step": self.step_i,
                "ok_step": True,
                "instruction": "LIVENESS (POSE): OK",
            }

        instruction, mode = self.steps[self.step_i]
        ok_step = self._check(yaw_deg, mode)

        if ok_step:
            self._hold += 1
            if self._hold >= self.hold_frames:
                # avança etapa
                if self.step_i < len(self.steps) - 1:
                    self.step_i += 1
                else:
                    self._done = True
                self._hold = 0
        else:
            self._hold = 0

        if self._done:
            return {
                "live": True,
                "reason": "Head pose challenge completo",
                "step": self.step_i,
                "ok_step": True,
                "instruction": "LIVENESS (POSE): OK",
            }

        return {
            "live": False,
            "reason": "Siga a instruçao na tela",
            "step": self.step_i,
            "ok_step": ok_step,
            "instruction": instruction,
        }
