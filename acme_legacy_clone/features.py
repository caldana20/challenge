"""Feature‑engineering helpers shared by train & inference."""
import numpy as np
import pandas as pd


def _efficiency_band(miles_per_day: float) -> str:
    if miles_per_day < 140:
        return "low"
    if miles_per_day < 180:
        return "mid_low"
    if miles_per_day < 220:
        return "sweet_spot"
    if miles_per_day < 260:
        return "mid_high"
    return "very_high"


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Raw aliases
    df["days"] = df["trip_duration_days"].astype(float)
    df["miles"] = df["miles_traveled"].astype(float)
    df["receipts"] = df["total_receipts_amount"].astype(float)

    # Basic combos
    df["miles_per_day"] = df["miles"] / df["days"].replace(0, np.nan)
    df["spend_per_day"] = df["receipts"] / df["days"].replace(0, np.nan)

    # Log transforms to tame skew
    df["log_miles"] = np.log1p(df["miles"])
    df["log_receipts"] = np.log1p(df["receipts"])

    # Boolean flags
    df["is_five_day_trip"] = (df["days"] == 5).astype(int)
    df["is_low_receipt_penalty"] = (
        (df["receipts"] < 50) & (df["days"] > 1)
    ).astype(int)
    df["ends_in_49_or_99"] = (
        (df["receipts"] % 1).round(2).isin([0.49, 0.99])
    ).astype(int)

    # Mileage brackets – piecewise linear tiers (0‑100, 100‑400, >400)
    df["mileage_bracket_100"] = np.clip(df["miles"], 0, 100)
    df["mileage_bracket_400"] = np.clip(df["miles"] - 100, 0, 300)
    df["mileage_bracket_hi"] = np.clip(df["miles"] - 400, 0, None)

    # Efficiency categorical encoded as one‑hot
#    df["eff_band"] = df["miles_per_day"].apply(_efficiency_band)
#    eff_dummies = pd.get_dummies(df["eff_band"], prefix="eff")
#    df = pd.concat([df, eff_dummies], axis=1)
    # House‑keep drop
#    return df.drop(columns=["eff_band","trip_duration_days",
#                            "miles_traveled","total_receipts_amount"])
    return df.drop(columns=["trip_duration_days",
                            "miles_traveled","total_receipts_amount"])
