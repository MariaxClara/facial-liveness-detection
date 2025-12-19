# src/liveness/rules_blink_only.py

class BlinkOnlyLivenessRule:
    """
    Regra de liveness baseada APENAS em piscar.
    """

    def __init__(self, min_blinks_for_liveness=1):
        self.min_blinks_for_liveness = min_blinks_for_liveness

    def decide(self, blink_detector):
        total_blinks = blink_detector.total_blinks
        live = total_blinks >= self.min_blinks_for_liveness

        if live:
            reason = "LIVE"
        else:
            reason = "NO_BLINK"

        return {
            "live": live,
            "reason": reason,
            "total_blinks": total_blinks,
        }
