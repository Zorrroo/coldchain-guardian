"""Synthetic (but domain-realistic) cold-chain shipment generator.

IMPORTANT: this is a *domain-based simulation*, not real operational data.
We build each shipment's spoilage outcome from cumulative thermal stress using
sensible logistics logic, then deliberately inject:

  * two *unobserved* drivers (hidden handling quality & sensor-calibration
    drift) that influence the outcome but are NOT exposed as features, and
  * Gaussian noise plus a few percent of label flips,

so the learning task is genuinely non-trivial and models land at realistic
performance (ROC-AUC ~0.85-0.95) rather than a leaky ~100%.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Make `from src...` imports work whether this file is run directly
# (`python src/generate_data.py`) or imported as part of the package.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config import (  # noqa: E402
    DATASET_PATH,
    MODE_PRODUCT_PENALTY,
    N_SHIPMENTS,
    PACKAGING_LEVELS,
    PACKAGING_MIX,
    PRODUCT_CATEGORIES,
    PRODUCT_MIX,
    RANDOM_SEED,
    REGIONS,
    REG_TARGET,
    SAMPLE_PATH,
    SAMPLE_SIZE,
    TARGET,
    TRANSPORT_MIX,
    TRANSPORT_MODES,
    ensure_dirs,
)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _zscore(x: np.ndarray) -> np.ndarray:
    """Standardise an array; used only to shape the latent risk score."""
    std = x.std()
    return (x - x.mean()) / std if std > 0 else x - x.mean()


# Intercept on the spoilage logit, calibrated so the base spoilage rate lands
# near ~29% (within reported cold-chain loss ranges for perishables/pharma).
SPOIL_INTERCEPT = -5.0


def generate_dataset(
    n: int = N_SHIPMENTS,
    seed: int = RANDOM_SEED,
    spoil_intercept: float = SPOIL_INTERCEPT,
) -> pd.DataFrame:
    """Return a DataFrame of ``n`` simulated shipments with both targets."""
    rng = np.random.default_rng(seed)

    # --- Categorical draws --------------------------------------------------
    products = list(PRODUCT_MIX)
    product_category = rng.choice(products, size=n, p=list(PRODUCT_MIX.values()))
    product_sensitivity = np.array(
        [PRODUCT_CATEGORIES[p] for p in product_category]
    )

    region_names = list(REGIONS)
    origin_region = rng.choice(region_names, size=n)
    destination_region = rng.choice(region_names, size=n)
    origin_climate = np.array([REGIONS[r] for r in origin_region])
    dest_climate = np.array([REGIONS[r] for r in destination_region])

    modes = list(TRANSPORT_MIX)
    transport_mode = rng.choice(modes, size=n, p=list(TRANSPORT_MIX.values()))
    mode_speed = np.array([TRANSPORT_MODES[m]["speed_kmh"] for m in transport_mode])
    mode_control = np.array([TRANSPORT_MODES[m]["control"] for m in transport_mode])
    mode_base_handoffs = np.array(
        [TRANSPORT_MODES[m]["base_handoffs"] for m in transport_mode]
    )

    packaging_names = list(PACKAGING_MIX)
    packaging_label = rng.choice(
        packaging_names, size=n, p=list(PACKAGING_MIX.values())
    )
    packaging_level = np.array([PACKAGING_LEVELS[p] for p in packaging_label])

    # --- Route geometry & timing -------------------------------------------
    # Distance depends loosely on whether origin/destination climates differ
    # (a crude proxy for geographic separation) plus a heavy-tailed component.
    base_distance = rng.gamma(shape=2.0, scale=2500.0, size=n)
    climate_gap = np.abs(origin_climate - dest_climate)
    route_distance_km = np.clip(
        base_distance + climate_gap * 180.0 + rng.normal(0, 500, n),
        300.0,
        20_000.0,
    )

    # Transit time derived from distance / mode speed, with handling overhead.
    transit_time_hours = (
        route_distance_km / mode_speed
        + rng.gamma(shape=2.0, scale=6.0, size=n)
    )
    transit_time_hours = np.clip(transit_time_hours, 2.0, 1200.0)

    # Seasonality: northern-hemisphere summer months run hotter.
    shipment_month = rng.integers(1, 13, size=n)
    seasonal_bump = 4.0 * np.sin((shipment_month - 4) / 12.0 * 2 * np.pi)

    # --- Ambient temperatures along the route ------------------------------
    route_climate = (origin_climate + dest_climate) / 2.0
    ambient_temp_avg_c = (
        route_climate + seasonal_bump + rng.normal(0, 2.5, n)
    )
    ambient_temp_max_c = (
        ambient_temp_avg_c
        + rng.gamma(shape=2.0, scale=3.0, size=n)
        + (1.0 - mode_control) * 6.0
    )

    # --- Handoffs & carrier -------------------------------------------------
    num_coldchain_handoffs = np.clip(
        rng.poisson(mode_base_handoffs + route_distance_km / 6000.0),
        0,
        20,
    ).astype(int)
    carrier_reliability_score = np.clip(rng.beta(6.0, 2.0, size=n), 0.05, 0.999)
    customs_hold_hours = np.clip(
        rng.exponential(scale=8.0, size=n)
        + (origin_region != destination_region) * rng.exponential(6.0, size=n),
        0.0,
        240.0,
    )

    # --- Refrigeration failures & temperature excursions -------------------
    # Failures grow with weak carriers, long transit and many handoffs.
    fail_logit = (
        -3.2
        + 1.8 * (1.0 - carrier_reliability_score)
        + 0.0016 * transit_time_hours
        + 0.08 * num_coldchain_handoffs
        - 0.25 * packaging_level
    )
    refrigeration_failure = rng.binomial(1, _sigmoid(fail_logit)).astype(int)

    # Excursion count: hotter ambient, weaker packaging/control, failures and
    # more handoffs all push the container out of its safe band more often.
    excursion_rate = np.clip(
        0.05 * np.clip(ambient_temp_max_c - 8.0, 0, None)
        + 0.6 * num_coldchain_handoffs * (1.0 - mode_control)
        + 3.0 * refrigeration_failure
        + 0.8 * (2 - packaging_level)
        + 0.004 * transit_time_hours,
        0.05,
        None,
    )
    num_temp_excursions = np.clip(rng.poisson(excursion_rate), 0, 60).astype(int)

    # --- Unobserved drivers (NOT exposed as features) ----------------------
    # These create irreducible uncertainty so the task can't be solved exactly.
    hidden_handling_quality = rng.normal(0, 1, n)   # good crews vs. careless
    sensor_calibration_drift = rng.normal(0, 1, n)  # mis-set thermostats

    # --- Latent thermal-stress score ---------------------------------------
    # Pre-compute standardised drivers so they can be reused in interactions.
    z_transit = _zscore(transit_time_hours)
    z_dist = _zscore(route_distance_km)
    z_amax = _zscore(ambient_temp_max_c)
    z_aavg = _zscore(ambient_temp_avg_c)
    z_exc = _zscore(num_temp_excursions.astype(float))
    z_hand = _zscore(num_coldchain_handoffs.astype(float))
    z_customs = _zscore(customs_hold_hours)

    # Linear main effects.
    main_effects = (
        0.95 * z_transit
        + 0.45 * z_dist
        + 0.85 * z_amax
        + 0.55 * z_aavg
        + 0.95 * z_exc
        + 0.60 * z_hand
        + 1.70 * refrigeration_failure
        - 0.75 * packaging_level
        - 1.30 * (carrier_reliability_score - 0.75)
        + 0.35 * z_customs
        - 1.40 * (mode_control - 0.80)
    )

    # Genuine domain interactions & threshold effects. These are NOT
    # representable by a main-effects linear model, so tree / gradient-boosting
    # models can exploit them -- which is exactly why cold-chain risk is
    # non-linear in practice:
    #   * heat compounds on long journeys,
    #   * a refrigeration failure is catastrophic once excursions pile up,
    #   * weak packaging amplifies high ambient temperatures,
    #   * there is a hard heat-stress threshold above ~35 C,
    #   * temperature-sensitive pharma faces a steep cliff once abused,
    #   * many handoffs with an unreliable carrier compound sharply.
    is_pharma = np.isin(product_category, ["vaccine", "biologic"]).astype(float)
    mode_product_penalty = np.array(
        [
            MODE_PRODUCT_PENALTY[(p, m)]
            for p, m in zip(product_category, transport_mode)
        ]
    )
    interactions = (
        1.10 * np.maximum(z_transit, 0.0) * np.maximum(z_amax, 0.0)
        + 1.40 * refrigeration_failure * np.clip(z_exc, 0.0, None)
        + 0.80 * (2 - packaging_level) * np.clip(z_amax, 0.0, None)
        + 1.20 * (ambient_temp_max_c > 35.0).astype(float)
        + 1.60
        * is_pharma
        * ((num_temp_excursions >= 2) | (refrigeration_failure == 1)).astype(float)
        + 0.90
        * np.clip(z_hand, 0.0, None)
        * (carrier_reliability_score < 0.70).astype(float)
        # Non-additive product x mode suitability (see config). Weighted
        # heavily because mode-product mismatch is a leading real-world driver
        # of cold-chain loss -- and its non-additivity is what lets tree models
        # out-predict the linear baseline.
        + 3.00 * mode_product_penalty
    )

    # Product sensitivity scales the accumulated (physical) stress; unobserved
    # drivers and pure noise are added afterwards so they stay irreducible.
    stress = (main_effects + interactions) * product_sensitivity + (
        # unobserved contributions:
        -0.70 * hidden_handling_quality
        + 0.55 * sensor_calibration_drift
        # pure noise:
        + rng.normal(0, 0.6, n)
    )

    # --- Binary spoilage label ---------------------------------------------
    # Intercept chosen so roughly a quarter to a third of shipments spoil.
    spoil_logit = spoil_intercept + 0.95 * stress
    spoil_prob = _sigmoid(spoil_logit)
    spoiled = rng.binomial(1, spoil_prob).astype(int)

    # Inject label noise: flip ~3% of labels (mis-recorded outcomes). This
    # caps achievable accuracy below ~97% and keeps the task honest.
    flip_mask = rng.random(n) < 0.03
    spoiled[flip_mask] = 1 - spoiled[flip_mask]

    # --- Bonus regression target: shelf-life remaining (0-100) -------------
    shelf_life_remaining = np.clip(
        100.0 - 17.0 * stress + rng.normal(0, 7.0, n),
        0.0,
        100.0,
    )
    # Spoiled shipments should look clearly degraded.
    shelf_life_remaining = np.where(
        spoiled == 1,
        np.clip(shelf_life_remaining - 22.0, 0.0, 100.0),
        shelf_life_remaining,
    )

    df = pd.DataFrame(
        {
            "product_category": product_category,
            "origin_region": origin_region,
            "destination_region": destination_region,
            "transport_mode": transport_mode,
            "packaging_insulation_quality": packaging_level,
            "transit_time_hours": transit_time_hours.round(2),
            "route_distance_km": route_distance_km.round(1),
            "num_coldchain_handoffs": num_coldchain_handoffs,
            "ambient_temp_avg_c": ambient_temp_avg_c.round(2),
            "ambient_temp_max_c": ambient_temp_max_c.round(2),
            "num_temp_excursions": num_temp_excursions,
            "refrigeration_failure": refrigeration_failure,
            "carrier_reliability_score": carrier_reliability_score.round(4),
            "customs_hold_hours": customs_hold_hours.round(2),
            "shipment_month": shipment_month,
            REG_TARGET: shelf_life_remaining.round(2),
            TARGET: spoiled,
        }
    )
    return df


def main() -> pd.DataFrame:
    """Generate, persist, and briefly summarise the dataset."""
    ensure_dirs()
    df = generate_dataset()
    df.to_csv(DATASET_PATH, index=False)

    # Commit a small, stratified, human-readable sample to the repo.
    half = SAMPLE_SIZE // 2
    parts = [
        grp.sample(min(half, len(grp)), random_state=RANDOM_SEED)
        for _, grp in df.groupby(TARGET)
    ]
    sample = (
        pd.concat(parts)
        .sample(frac=1, random_state=RANDOM_SEED)
        .reset_index(drop=True)
    )
    sample.to_csv(SAMPLE_PATH, index=False)

    rate = df[TARGET].mean()
    print(f"Generated {len(df):,} shipments -> {DATASET_PATH}")
    print(f"Spoilage rate: {rate:.1%}  (target ~20-35%)")
    print(f"Mean shelf-life remaining: {df[REG_TARGET].mean():.1f}/100")
    print(f"Wrote {len(sample)} rows to {SAMPLE_PATH}")
    return df


if __name__ == "__main__":
    main()
