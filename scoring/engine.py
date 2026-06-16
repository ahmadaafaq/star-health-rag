"""
scoring/engine.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Lead score engine blending logic:
  final_score = profile_score * 0.40 + whatsapp_behavior * 0.35 + call_signal * 0.25
If any signal is missing (None), its weight is distributed proportionally.
Score is clamped to [0, 100].
"""

from __future__ import annotations
import logging

logger = logging.getLogger("scoring.engine")

def calculate_blended_score(
    profile_score: float | None,
    whatsapp_behavior: float | None,
    call_signal: float | None
) -> float:
    """
    Calculate the final blended score from multiple intent signals.
    """
    w_profile = 0.40
    w_whatsapp = 0.35
    w_call = 0.25

    active_weights = []
    active_values = []

    if profile_score is not None:
        active_weights.append(w_profile)
        active_values.append(profile_score)

    if whatsapp_behavior is not None:
        active_weights.append(w_whatsapp)
        active_values.append(whatsapp_behavior)

    if call_signal is not None:
        active_weights.append(w_call)
        active_values.append(call_signal)

    if not active_weights:
        # Default safety fallback
        return 70.0

    total_weight = sum(active_weights)
    blended = sum(val * (w / total_weight) for val, w in zip(active_values, active_weights))
    
    final_score = max(0.0, min(100.0, blended))
    logger.info(
        f"Blending result: profile_score={profile_score} ({w_profile}), "
        f"whatsapp_behavior={whatsapp_behavior} ({w_whatsapp}), "
        f"call_signal={call_signal} ({w_call}) -> "
        f"Final blended score: {final_score:.2f} (weights sum={total_weight})"
    )
    return float(final_score)


def classify_lead_type(score: float) -> str:
    """
    Classify the lead based on final blended score.
    - Score >= 70 -> 'hot'
    - Score 40-69 -> 'warm'
    - Score < 40 -> 'cold'
    """
    if score >= 70:
        return "hot"
    elif score >= 40:
        return "warm"
    else:
        return "cold"
