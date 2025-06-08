"""Deterministic post‑model adjustment layer that keeps *known quirks* intact.

⚠️  This module now **falls back to sensible defaults** when *quirks.json* is
missing.  This lets you run the training script on a clean checkout without
exploding on import.
"""
import json
from pathlib import Path

_DEFAULT_Q = {
    "cents_bonus": 0.50,
    "five_day_bonus": 37.42,
    "hi_spend_threshold_per_day": 100,
    "long_trip_penalty_multiplier": 0.90,
}

_qfile = Path(__file__).with_name("quirks.json")
try:
    _Q = json.loads(_qfile.read_text()) if _qfile.exists() else _DEFAULT_Q.copy()
except json.JSONDecodeError:
    # Corrupted file?  Fail soft with defaults.
    _Q = _DEFAULT_Q.copy()


def apply_quirks(pred: float, raw: dict) -> float:
    days = raw["trip_duration_days"]
    receipts = raw["total_receipts_amount"]

    # --- 49¢ / 99¢ double‑round quirk
    if round(receipts % 1, 2) in (0.49, 0.99):
        pred += _Q["cents_bonus"]

    # --- Fixed 5‑day bonus
    if days == 5:
        pred += _Q["five_day_bonus"]

    # --- Vacation penalty: long trip + high spend per day
    if days >= 8 and receipts / days > _Q["hi_spend_threshold_per_day"]:
        pred *= _Q["long_trip_penalty_multiplier"]

    return round(pred, 2)


