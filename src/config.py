"""Central configuration: random seed, paths, and domain constants.

Everything that controls reproducibility or wiring lives here so the rest of
the codebase stays declarative and easy to audit.
"""

from pathlib import Path

# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
RANDOM_SEED = 42

# --------------------------------------------------------------------------- #
# Dataset size
# --------------------------------------------------------------------------- #
N_SHIPMENTS = 20_000
SAMPLE_SIZE = 200  # small, human-readable sample committed to the repo

# --------------------------------------------------------------------------- #
# Paths (all resolved relative to the project root, so scripts work from
# anywhere: `python run.py`, `python src/predict.py`, or pytest)
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
FIGURES_DIR = ROOT / "figures"
MODELS_DIR = ROOT / "models"

DATASET_PATH = DATA_DIR / "shipments.csv"            # full dataset (git-ignored)
SAMPLE_PATH = DATA_DIR / "sample_shipments.csv"      # small committed sample
MODEL_PATH = MODELS_DIR / "best_model.joblib"        # classifier pipeline
REG_MODEL_PATH = MODELS_DIR / "shelf_life_model.joblib"  # regression pipeline
METRICS_PATH = MODELS_DIR / "metrics.json"           # recorded headline metrics

# --------------------------------------------------------------------------- #
# Targets
# --------------------------------------------------------------------------- #
TARGET = "spoiled"                      # binary classification label (1 = spoiled)
REG_TARGET = "shelf_life_remaining"     # bonus regression target (0-100)

# --------------------------------------------------------------------------- #
# Train / validation / test split
# --------------------------------------------------------------------------- #
TEST_SIZE = 0.20
VAL_SIZE = 0.20   # fraction of the *training* remainder used for validation
CV_FOLDS = 5

# --------------------------------------------------------------------------- #
# Domain definitions
# --------------------------------------------------------------------------- #
# Each product category has an intrinsic thermal sensitivity multiplier:
# pharma (vaccine/biologic) and seafood degrade fastest; produce is hardier.
PRODUCT_CATEGORIES = {
    "fresh_produce": 0.85,
    "dairy": 1.00,
    "seafood": 1.35,
    "frozen": 1.20,
    "vaccine": 1.65,
    "biologic": 1.75,
}

# How common each category is in the shipment mix (must sum to 1).
PRODUCT_MIX = {
    "fresh_produce": 0.28,
    "dairy": 0.20,
    "seafood": 0.16,
    "frozen": 0.18,
    "vaccine": 0.10,
    "biologic": 0.08,
}

# Regions with an average ambient (climate) temperature in deg C. Shipments
# crossing hot regions accumulate more thermal stress.
REGIONS = {
    "North_America": 16.0,
    "Europe": 13.0,
    "East_Asia": 21.0,
    "Southeast_Asia": 30.0,
    "Middle_East": 33.0,
    "South_America": 25.0,
    "Africa": 31.0,
    "Oceania": 19.0,
}

# Transport modes: (base transit multiplier, temperature-control quality,
# typical handoff count). Air is fast and tightly controlled; sea reefer is
# slow with many handoffs; road reefer sits in between.
TRANSPORT_MODES = {
    "air": {"speed_kmh": 750.0, "control": 0.90, "base_handoffs": 3},
    "road_reefer": {"speed_kmh": 65.0, "control": 0.70, "base_handoffs": 4},
    "sea_reefer": {"speed_kmh": 35.0, "control": 0.78, "base_handoffs": 6},
}
TRANSPORT_MIX = {"air": 0.30, "road_reefer": 0.35, "sea_reefer": 0.35}

# Packaging insulation quality is ordinal: basic < standard < premium.
PACKAGING_LEVELS = {"basic": 0, "standard": 1, "premium": 2}
PACKAGING_MIX = {"basic": 0.40, "standard": 0.40, "premium": 0.20}

# Non-additive suitability penalty for each (product, transport_mode) pairing.
# The *ranking* of modes deliberately differs by product -- e.g. frozen cargo
# rides a sea reefer well but suffers on long road hauls, while fresh produce
# rots at sea yet is fine by air. Because the best mode depends on the product,
# this effect cannot be expressed as (product effect + mode effect), so it
# genuinely rewards models that capture interactions (trees / gradient boosting)
# over a main-effects linear baseline.
MODE_PRODUCT_PENALTY = {
    ("fresh_produce", "air"): 0.0, ("fresh_produce", "road_reefer"): 0.4, ("fresh_produce", "sea_reefer"): 1.0,
    ("dairy", "air"): 0.0, ("dairy", "road_reefer"): 0.3, ("dairy", "sea_reefer"): 0.6,
    ("seafood", "air"): 0.0, ("seafood", "road_reefer"): 0.6, ("seafood", "sea_reefer"): 0.9,
    ("frozen", "air"): 0.2, ("frozen", "road_reefer"): 0.8, ("frozen", "sea_reefer"): 0.3,
    ("vaccine", "air"): 0.0, ("vaccine", "road_reefer"): 0.7, ("vaccine", "sea_reefer"): 0.5,
    ("biologic", "air"): 0.0, ("biologic", "road_reefer"): 0.8, ("biologic", "sea_reefer"): 0.5,
}

# --------------------------------------------------------------------------- #
# Feature groups (used to build the preprocessing ColumnTransformer)
# --------------------------------------------------------------------------- #
CATEGORICAL_FEATURES = [
    "product_category",
    "origin_region",
    "destination_region",
    "transport_mode",
]

# Ordinal feature kept as an integer 0/1/2 (already encoded during generation).
ORDINAL_FEATURES = ["packaging_insulation_quality"]

NUMERIC_FEATURES = [
    "transit_time_hours",
    "route_distance_km",
    "num_coldchain_handoffs",
    "ambient_temp_avg_c",
    "ambient_temp_max_c",
    "num_temp_excursions",
    "refrigeration_failure",
    "carrier_reliability_score",
    "customs_hold_hours",
    "shipment_month",
]

FEATURE_COLUMNS = CATEGORICAL_FEATURES + ORDINAL_FEATURES + NUMERIC_FEATURES


def ensure_dirs() -> None:
    """Create output directories if they do not yet exist."""
    for directory in (DATA_DIR, FIGURES_DIR, MODELS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
