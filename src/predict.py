"""Load the saved models and score example shipments.

Run directly to see predictions on a few hand-built shipments that span the
risk spectrum:

    python src/predict.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config import (  # noqa: E402
    FEATURE_COLUMNS,
    MODEL_PATH,
    REG_MODEL_PATH,
)


def example_shipments() -> pd.DataFrame:
    """Three contrasting shipments: low, moderate and high spoilage risk."""
    rows = [
        {
            # Premium-packed vaccines flown a short hop with a top carrier.
            "label": "Low risk: air-freight vaccines, premium packaging",
            "product_category": "vaccine",
            "origin_region": "Europe",
            "destination_region": "North_America",
            "transport_mode": "air",
            "packaging_insulation_quality": 2,
            "transit_time_hours": 14.0,
            "route_distance_km": 6500.0,
            "num_coldchain_handoffs": 2,
            "ambient_temp_avg_c": 12.0,
            "ambient_temp_max_c": 18.0,
            "num_temp_excursions": 0,
            "refrigeration_failure": 0,
            "carrier_reliability_score": 0.97,
            "customs_hold_hours": 4.0,
            "shipment_month": 2,
        },
        {
            # Dairy by road reefer, standard packaging, a couple of excursions.
            "label": "Moderate risk: road-reefer dairy, standard packaging",
            "product_category": "dairy",
            "origin_region": "South_America",
            "destination_region": "North_America",
            "transport_mode": "road_reefer",
            "packaging_insulation_quality": 1,
            "transit_time_hours": 90.0,
            "route_distance_km": 5200.0,
            "num_coldchain_handoffs": 5,
            "ambient_temp_avg_c": 24.0,
            "ambient_temp_max_c": 31.0,
            "num_temp_excursions": 3,
            "refrigeration_failure": 0,
            "carrier_reliability_score": 0.80,
            "customs_hold_hours": 18.0,
            "shipment_month": 7,
        },
        {
            # Seafood on a long sea reefer leg through hot regions, a fridge
            # failure, many handoffs and basic packaging.
            "label": "High risk: sea-reefer seafood, refrigeration failure",
            "product_category": "seafood",
            "origin_region": "Southeast_Asia",
            "destination_region": "Middle_East",
            "transport_mode": "sea_reefer",
            "packaging_insulation_quality": 0,
            "transit_time_hours": 520.0,
            "route_distance_km": 9800.0,
            "num_coldchain_handoffs": 11,
            "ambient_temp_avg_c": 33.0,
            "ambient_temp_max_c": 42.0,
            "num_temp_excursions": 14,
            "refrigeration_failure": 1,
            "carrier_reliability_score": 0.55,
            "customs_hold_hours": 60.0,
            "shipment_month": 8,
        },
    ]
    return pd.DataFrame(rows)


def predict(df: pd.DataFrame) -> pd.DataFrame:
    """Attach spoilage probability and predicted shelf-life to ``df``."""
    model = joblib.load(MODEL_PATH)
    reg_model = joblib.load(REG_MODEL_PATH)

    X = df[FEATURE_COLUMNS]
    proba = model.predict_proba(X)[:, 1]
    out = df.copy()
    out["spoilage_probability"] = proba.round(3)
    out["prediction"] = ["SPOILED" if p >= 0.5 else "SAFE" for p in proba]
    # Clip the regressor to the physically meaningful 0-100 range.
    out["predicted_shelf_life"] = np.clip(reg_model.predict(X), 0.0, 100.0).round(1)
    return out


def main() -> None:
    if not MODEL_PATH.exists() or not REG_MODEL_PATH.exists():
        raise SystemExit(
            "Model artifacts not found. Run `python run.py` first to build them."
        )

    df = example_shipments()
    result = predict(df)

    print("=" * 74)
    print("ColdChain Guardian - spoilage risk predictions")
    print("=" * 74)
    for _, row in result.iterrows():
        print(f"\n{row['label']}")
        print(
            f"   product={row['product_category']:<13} "
            f"mode={row['transport_mode']:<12} "
            f"excursions={row['num_temp_excursions']:<3} "
            f"fridge_fail={row['refrigeration_failure']}"
        )
        print(
            f"   -> spoilage probability: {row['spoilage_probability']:.1%}"
            f"   verdict: {row['prediction']}"
            f"   predicted shelf-life: {row['predicted_shelf_life']}/100"
        )
    print("\n" + "=" * 74)


if __name__ == "__main__":
    main()
