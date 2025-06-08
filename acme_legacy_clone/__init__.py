from pathlib import Path
from functools import lru_cache

import joblib
import pandas as pd
from .features import add_features

# -- Delay heavy imports & file IO until first call -------------------------
_MODEL_PATH = Path(__file__).with_name("model.pkl")

__all__ = ["reimburse"]

@lru_cache()
def _load_model():
    """Lazy‑load the LightGBM model once the first time we need it.

    We defer this so that running the *training* module does **not** require the
    model artefact to exist yet (important for `python -m acme_legacy_clone.train_model`).
    """
    if not _MODEL_PATH.exists():
        raise RuntimeError(
            "Model artefact 'model.pkl' not found. "
            "Train it via `python -m acme_legacy_clone.train_model`."
        )
    return joblib.load(_MODEL_PATH)


def reimburse(days: int, miles: float, receipts: float) -> float:
    """Return the cloned legacy reimbursement amount (two decimals)."""
    # Local import avoids loading patch.py (and quirks.json) during training
    from .patch import apply_quirks

    raw = {
        "trip_duration_days": days,
        "miles_traveled": miles,
        "total_receipts_amount": receipts,
    }
    X = add_features(pd.DataFrame([raw]))
    base_pred = float(_load_model().predict(X)[0])
    return apply_quirks(base_pred, raw)