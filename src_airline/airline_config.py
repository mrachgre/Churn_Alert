"""
airline_config.py
Centralized configuration for the AIRLINE Customer Satisfaction pipeline.
Completely isolated from the E-Commerce and COFINFAD pipelines.
"""
import os

DATASET = "AIRLINE"

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR     = os.path.join(BASE_DIR, "data",    "AIRLINE")
DATA_INT_DIR = os.path.join(BASE_DIR, "data",    "AIRLINE", "intermediate")
MODEL_DIR    = os.path.join(BASE_DIR, "models",  "AIRLINE")
OUT_DIR      = os.path.join(BASE_DIR, "outputs", "AIRLINE")
for _d in [DATA_INT_DIR, MODEL_DIR, OUT_DIR]:
    os.makedirs(_d, exist_ok=True)

# Actual filename (has spaces — do NOT rename the original)
CSV_FILENAME = "Customer Satisfaction in Airline export 2026-06-16 07-20-47.csv"
CSV_PATH     = os.path.join(DATA_DIR, CSV_FILENAME)

# ── Reference counts from README (for verification in Step 1) ─────────────────
REF = {
    "total_rows":        129_880,
    "satisfied":          71_087,
    "dissatisfied":       58_793,
    "loyal":             106_100,
    "disloyal":           23_780,
    "business_travel":    89_693,
    "personal_travel":    40_187,
    "class_business":     62_160,
    "class_eco":          58_309,
    "class_eco_plus":      9_411,
    "null_arrival_delay":    393,
}

# ── Column groups ─────────────────────────────────────────────────────────────
TARGET_COL = "satisfaction"

SERVICE_COLS = [
    "Seat comfort",
    "Departure/Arrival time convenient",
    "Food and drink",
    "Gate location",
    "Inflight wifi service",
    "Inflight entertainment",
    "Online support",
    "Ease of Online booking",
    "On-board service",
    "Leg room service",
    "Baggage handling",
    "Checkin service",
    "Cleanliness",
    "Online boarding",
]

DELAY_COLS   = ["Departure Delay in Minutes", "Arrival Delay in Minutes"]
NUMERIC_COLS = ["Flight Distance", "Departure Delay in Minutes", "Arrival Delay in Minutes"]
CAT_COLS     = ["Customer Type", "Type of Travel", "Class"]

# Column with known no-zero (per README: Baggage handling min=1)
NO_ZERO_COLS = ["Baggage handling"]

# Column to use as case-study for 0-vs-rating analysis
WIFI_COL = "Inflight wifi service"

# Outlier threshold for delay (minutes)
DELAY_EXTREME_THRESHOLD = 500

# ── Phase 2 constants ─────────────────────────────────────────────────────────

# Ground experience composite score
GROUND_COLS = [
    "Ease of Online booking", "Online boarding", "Checkin service",
    "Gate location", "Baggage handling",
]

# Inflight experience composite score
INFLIGHT_COLS = [
    "Seat comfort", "Food and drink", "Inflight wifi service",
    "Inflight entertainment", "On-board service", "Leg room service",
    "Cleanliness",
]

# Features used for KMeans clustering (no categorical leakage)
KMEANS_FEATURES = [
    "ground_experience_score", "inflight_experience_score",
    "delay_severity", "Age", "Flight Distance",
]

# Encoding decisions (Phase 2 confirmed)
# Class: Ordinal  |  Customer Type: Binary  |  Type of Travel: Binary
CLASS_MAP         = {"Eco": 0, "Eco Plus": 1, "Business": 2}
CUSTOMER_TYPE_MAP = {"disloyal Customer": 0, "Loyal Customer": 1}
TRAVEL_TYPE_MAP   = {"Personal Travel": 0, "Business travel": 1}

# Columns to EXCLUDE when joining profile (potential leaky cols)
LEAKY_COLS: list = []  # no equivalent in Airline (all cols are survey-time)

# Model artifact names
MODEL_NAMES = {
    "xgb_base":  "xgb_base_model.pkl",
    "xgb_clust": "xgb_clust_model.pkl",
    "rf_clust":  "rf_clust_model.pkl",
    "dt_clust":  "dt_clust_model.pkl",
    "lr_base":   "lr_base_model.pkl",
    "lr_dist":   "lr_dist_model.pkl",
}

# Optuna trials
OPTUNA_TRIALS   = 20
RANDOM_STATE    = 42
TEST_SIZE       = 0.20
SMOTE_THRESHOLD = 0.60   # skip SMOTE if minority class >= 40%
