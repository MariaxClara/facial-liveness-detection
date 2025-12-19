# src/liveness/rules_blink_brightness.py

class BlinkBrightnessLivenessRule:
    """
    Regra de liveness que combina:
    - Informação geométrica (piscar, via BlinkDetector)
    - Informação fotométrica (variação de brilho, via BrightnessLiveness)

    Ideia:
    - Para ser LIVE, precisa:
        * pelo menos um blink
        * padrão de brilho compatível com rosto real (live_like = True)
    """

    def __init__(
        self,
        min_blinks_for_liveness=1,
        require_brightness_live_like=True,
    ):
        """
        :param min_blinks_for_liveness: mínimo de blinks para considerar liveness.
        :param require_brightness_live_like:
            Se True, exige que o módulo de brightness indique live_like=True.
            Se False, o brilho é só um fator auxiliar (não bloqueia).
        """
        self.min_blinks_for_liveness = min_blinks_for_liveness
        self.require_brightness_live_like = require_brightness_live_like

    def decide(self, blink_detector, brightness_info):
        
        total_blinks = blink_detector.total_blinks
        blink_live = total_blinks >= self.min_blinks_for_liveness

        brightness_live = brightness_info.get("live_like", None)
        var_mean = brightness_info.get("var_mean", None)
        var_std = brightness_info.get("var_std", None)

        if not blink_live:
            live = False
            reason = "NO_BLINK"
        else:
            if brightness_live is True:
                live = True
                reason = "LIVE_BLINK_BRIGHT"
            elif brightness_live is False:
                if self.require_brightness_live_like:
                    live = False
                    reason = "SUSPECT_BRIGHTNESS"
                else:
                    live = True
                    reason = "LIVE_BLINK_BUT_BRIGHT_SUSPECT"
            else:
                live = blink_live
                reason = "LIVE_BLINK_ONLY_BRIGHT_UNKNOWN" if blink_live else "NO_BLINK"

        return {
            "live": live,
            "reason": reason,
            "blink_live": blink_live,
            "brightness_live": brightness_live,
            "total_blinks": total_blinks,
            "var_mean": var_mean,
            "var_std": var_std,
        }
