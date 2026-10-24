"""A single dose-pair ATET; no derivative, dose averaging or group combination."""

import numpy as np

from .score import normalized_atet_score
from .weights import get_trimmed_weights


def estimate_atet_point(
    frame, dtreat, dcontrol, bandwidth, gps_mean, gps_sigma, mu_control, config
):
    """bandwidth is the rule's h; np halfwidth is sqrt(5)*h.

    Returns the raw point estimate and an ESS support flag. The caller must check
    supported before interpreting the estimate as admissible under its ESS rule.
    """
    halfwidth = bandwidth * (np.sqrt(5) if config.kernel_convention == "np" else 1.0)
    high, low, diagnostic = get_trimmed_weights(
        frame,
        dtreat,
        dcontrol,
        gps_mean,
        gps_sigma,
        halfwidth,
        config.dose,
        "ind_code",
        config.trim_threshold,
        config.density_floor,
        trim_mode=config.trim_mode,
    )
    residual = frame.Y_diff.to_numpy(float) - np.asarray(mu_control, float)
    point, influence, normalization = normalized_atet_score(residual, high, low)
    return {
        "atet": point,
        "influence": influence,
        "dtreat": dtreat,
        "dcontrol": dcontrol,
        "bandwidth": bandwidth,
        "halfwidth": halfwidth,
        "supported": bool(min(diagnostic["ess_d"], diagnostic["ess_dp"]) > config.ess),
        **diagnostic,
        **normalization,
    }
