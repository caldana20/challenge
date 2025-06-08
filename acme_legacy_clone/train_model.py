"""CLI script to fit the gradient‑boosted tree *and* learn quirk constants.

Run once to produce *model.pkl* and *quirks.json* in the package folder.
"""
import argparse
import itertools
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

from .features import add_features
from .patch import apply_quirks, _Q as DEFAULT_Q

ROOT = Path(__file__).resolve().parent

def _flatten_cases(df: pd.DataFrame) -> pd.DataFrame:
    """Return DataFrame with required columns; infer format."""
    # format A: already flat with legacy_reimbursement
    if "legacy_reimbursement" in df.columns:
        return df.copy()

    # format B: Kaggle‑style JSON with (input, expected_output)
    if {"input", "expected_output"}.issubset(df.columns):
        inputs_expanded = df["input"].apply(pd.Series)
        flat = pd.concat([
            inputs_expanded.rename(columns={
                "days": "trip_duration_days",
                "miles": "miles_traveled",
                "receipts": "total_receipts_amount",
            }),
            df["expected_output"].rename("legacy_reimbursement"),
        ], axis=1)
        return flat

    raise ValueError(
        "Dataset must contain either 'legacy_reimbursement' or ('input', 'expected_output') columns"
    )


def _learn_quirks(y_true, base_pred, rows):
    best_q = DEFAULT_Q.copy()
    best_acc = 0.0
    grids = {
        "cents_bonus": [0.3, 0.4, 0.5, 0.6],
        "five_day_bonus": [30, 35, 37.42, 40],
        "long_trip_penalty_multiplier": [0.85, 0.9, 0.95],
    }
    for cb, fb, lp in itertools.product(
        grids["cents_bonus"], grids["five_day_bonus"], grids["long_trip_penalty_multiplier"]
    ):
        trial = {
            "cents_bonus": cb,
            "five_day_bonus": fb,
            "hi_spend_threshold_per_day": DEFAULT_Q["hi_spend_threshold_per_day"],
            "long_trip_penalty_multiplier": lp,
        }
        adj = [apply_quirks(p, r) for p, r in zip(base_pred, rows.to_dict("records"))]
        acc = (np.round(adj, 2) == np.round(y_true, 2)).mean()
        if acc > best_acc:
            best_acc, best_q = acc, trial
    return best_q, best_acc




def main():
    ap = argparse.ArgumentParser(description="Train legacy clone model")
    ap.add_argument("--cases", default="public_cases.json", help="Labelled data file (JSON/CSV)")
    args = ap.parse_args()

    # Load flexible dataset
    if args.cases.endswith(".csv"):
        raw_df = pd.read_csv(args.cases)
    else:
        raw_df = pd.read_json(args.cases)
    df = _flatten_cases(raw_df)

    y = df["legacy_reimbursement"].astype(float).values
    df = df.drop(columns=["legacy_reimbursement"])
    X = add_features(df)

    idx_train, idx_val = train_test_split(np.arange(len(df)), test_size=0.15, random_state=0)

    model = lgb.LGBMRegressor(max_depth=-1, 
                              n_estimators=100, 
                              learning_rate=0.2,
                              subsample=0.8, 
                              colsample_bytree=0.8, 
                              random_state=0)
    model.fit(X.iloc[idx_train], y[idx_train])

    val_pred = model.predict(X.iloc[idx_val])
    mae = mean_absolute_error(y[idx_val], val_pred)
    print(f"MAE (val) before quirks: {mae:.2f}")

    best_q, best_acc = _learn_quirks(y[idx_val], val_pred, df.iloc[idx_val])
    print(f"Exact‑match rate after quirks: {best_acc:.2%}")

    # Save artefacts
    joblib.dump(model, ROOT / "model.pkl")
    with open(ROOT / "quirks.json", "w") as fh:
        json.dump(best_q, fh, indent=2)
    print("Saved: model.pkl  quirks.json")


if __name__ == "__main__":
    main()