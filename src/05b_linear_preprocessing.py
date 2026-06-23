"""
Step 5b: Linear Model-Specific Preprocessing
============================================================
Input : ecommerce_churn_train.csv  (split sẵn từ Step 02)
        ecommerce_churn_test.csv   (split sẵn từ Step 02)

Lý do tách riêng khỏi pipeline chung (Step 05):
  - Tree models không nhạy cảm với outlier / scale → dùng output Step 05
  - Linear models nhạy với outlier → cần IQR clip riêng

Pipeline:
  1. Feature engineering (giống Step 05)
  2. Missing-indicator flags cho 6 cột MAR/MNAR (giống Step 05)
  3. Outlier removal: IQR clip (1.5×IQR) trên numeric cols — chỉ train
  4. OHE: reuse encoder.pkl từ Step 05 (cùng category mapping)
  5. SimpleImputer mean MỚI (fit on cleaned train)
  6. MinMaxScaler MỚI (fit on cleaned train)
  7. SMOTE (training set only)

Variant _dist (cho LR-Dist):
  Sau bước 6, load kmeans_fe_model.pkl từ Step 05,
  thêm K cột dist_to_cluster_i vào cả train và test.

Outputs
-------
  data/X_train_linear.csv          base features, SMOTE-balanced
  data/X_test_linear.csv           base features
  data/X_train_linear_dist.csv     base + dist features, SMOTE-balanced
  data/X_test_linear_dist.csv      base + dist features
  data/y_train_linear.csv          nhãn sau SMOTE (dùng chung base & dist)
  models/linear_scaler.pkl         MinMaxScaler (fitted on cleaned train)
  models/linear_imputer.pkl        SimpleImputer mean (fitted on cleaned train)
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE

from pipeline_config import DATA_DIR, MODEL_DIR   # noqa: E402

TRAIN_PATH = os.path.join(DATA_DIR, "ecommerce_churn_train.csv")
TEST_PATH  = os.path.join(DATA_DIR, "ecommerce_churn_test.csv")

TARGET       = "Churn"
CAT_AS_STR   = {"CityTier": str, "Complain": str}
RANDOM_STATE = 42
IQR_MULT     = 1.5

# Đồng bộ với Step 05
MAR_MNAR_COLS = [
    "Tenure",
    "OrderCount",
    "OrderAmountHikeFromlastYear",
    "CouponUsed",
    "WarehouseToHome",
    "HourSpendOnApp",
]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=CAT_AS_STR)
    df[TARGET] = df[TARGET].astype(int)
    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["device_per_tenure"]  = df["NumberOfDeviceRegistered"] / (df["Tenure"].fillna(0) + 1)
    df["cashback_per_order"] = df["CashbackAmount"] / (df["OrderCount"].fillna(0) + 1)
    df["inactivity_ratio"]   = df["DaySinceLastOrder"] / (df["Tenure"].fillna(0) + 1)
    return df


def add_missing_flags(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        if col in df.columns:
            df[f"{col}_was_missing"] = df[col].isnull().astype(int)
    return df


def prepare_features(df: pd.DataFrame):
    drop_cols = ["CustomerID", TARGET]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df[TARGET]
    return X, y


def remove_outliers_iqr(X: pd.DataFrame,
                         num_cols: list,
                         multiplier: float = IQR_MULT):
    """
    IQR clip trên numeric cols — giữ nguyên số hàng (clip, không drop).
    Chỉ fit/apply trên training data; test KHÔNG clip.
    """
    X_clean = X.copy()
    report  = {}
    for col in num_cols:
        if col not in X_clean.columns:
            continue
        q1  = X_clean[col].quantile(0.25)
        q3  = X_clean[col].quantile(0.75)
        iqr = q3 - q1
        lo  = q1 - multiplier * iqr
        hi  = q3 + multiplier * iqr
        n_out = ((X_clean[col] < lo) | (X_clean[col] > hi)).sum()
        X_clean[col] = X_clean[col].clip(lo, hi)
        if n_out > 0:
            report[col] = {"lower": round(lo, 4), "upper": round(hi, 4), "clipped": n_out}
    return X_clean, report


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def run_linear_preprocessing(random_state: int = RANDOM_STATE):
    print(f"\n{'='*60}")
    print("STEP 5b: LINEAR MODEL PREPROCESSING")
    print("         (Outlier Removal + IQR Clip, SimpleImputer Mean)")
    print(f"{'='*60}")

    # ── Load ──────────────────────────────────────────────────────────────────
    train_raw = load(TRAIN_PATH)
    test_raw  = load(TEST_PATH)
    print(f"[05b] Train: {train_raw.shape} | Test: {test_raw.shape}")

    # ── Feature engineering ───────────────────────────────────────────────────
    train_raw = feature_engineering(train_raw)
    test_raw  = feature_engineering(test_raw)

    # ── Missing-indicator flags (MAR/MNAR) ────────────────────────────────────
    train_raw = add_missing_flags(train_raw, MAR_MNAR_COLS)
    test_raw  = add_missing_flags(test_raw,  MAR_MNAR_COLS)
    print(f"[05b] Missing-indicator flags added for: {MAR_MNAR_COLS}")

    # ── Tách X, y ─────────────────────────────────────────────────────────────
    X_train, y_train = prepare_features(train_raw)
    X_test,  y_test  = prepare_features(test_raw)
    print(f"[05b] Features: {X_train.shape[1]} | "
          f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")
    print(f"[05b] Before SMOTE → Class 0: {(y_train==0).sum()} | Class 1: {(y_train==1).sum()}")

    # ── Identify column groups ────────────────────────────────────────────────
    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()

    # ── 1. Outlier removal — train only ──────────────────────────────────────
    X_train_clean, report = remove_outliers_iqr(X_train, num_cols)
    y_train_clean = y_train.loc[X_train_clean.index]

    if report:
        print(f"[05b] Outlier clip (IQR×{IQR_MULT}) — {len(report)} columns affected:")
        for col, info in report.items():
            print(f"       {col:<35} clipped {info['clipped']:>4} values  "
                  f"[{info['lower']:.2f}, {info['upper']:.2f}]")
    else:
        print("[05b] No outliers clipped (IQR check passed for all numeric cols)")

    # ── 2. OHE: reuse encoder.pkl từ Step 05 ─────────────────────────────────
    encoder   = joblib.load(os.path.join(MODEL_DIR, "encoder.pkl"))
    ohe_names = encoder.get_feature_names_out(cat_cols).tolist()

    def encode(X_raw):
        cat_enc = pd.DataFrame(encoder.transform(X_raw[cat_cols]),
                               columns=ohe_names, index=X_raw.index)
        return pd.concat([X_raw[num_cols].reset_index(drop=True),
                          cat_enc.reset_index(drop=True)], axis=1)

    X_train_enc = encode(X_train_clean)
    X_test_enc  = encode(X_test)
    all_cols    = X_train_enc.columns.tolist()
    print(f"[05b] After OHE: {len(all_cols)} features")

    # ── 3. SimpleImputer mean MỚI (fit on cleaned train) ─────────────────────
    num_enc_cols = X_train_enc.select_dtypes(include=[np.number]).columns.tolist()

    linear_imputer = SimpleImputer(strategy="mean")
    X_train_enc[num_enc_cols] = linear_imputer.fit_transform(X_train_enc[num_enc_cols])
    X_test_enc[num_enc_cols]  = linear_imputer.transform(X_test_enc[num_enc_cols])
    print(f"[05b] SimpleImputer (mean) done. "
          f"Missing remaining: {X_train_enc.isnull().sum().sum()}")

    # ── 4. MinMaxScaler MỚI (fit on cleaned train) ───────────────────────────
    linear_scaler = MinMaxScaler()
    X_train_enc[num_enc_cols] = linear_scaler.fit_transform(X_train_enc[num_enc_cols])
    X_test_enc[num_enc_cols]  = linear_scaler.transform(X_test_enc[num_enc_cols])
    # Guard: test có thể có outlier nằm ngoài range train sau clip
    X_test_enc[num_enc_cols]  = X_test_enc[num_enc_cols].clip(0, 1)
    print(f"[05b] MinMaxScaler (linear) applied.")

    # ── 5. SMOTE — base features ─────────────────────────────────────────────
    smote = SMOTE(random_state=random_state, k_neighbors=5)
    X_tr_lin_arr, y_tr_lin = smote.fit_resample(X_train_enc, y_train_clean)
    X_tr_lin = pd.DataFrame(X_tr_lin_arr, columns=all_cols)
    print(f"[05b] After SMOTE (base) → "
          f"Class 0: {(y_tr_lin==0).sum()} | Class 1: {(y_tr_lin==1).sum()}")

    # ── 6. Variant _dist: fit KMeans riêng trên 05b feature space ───────────
    # Không tái dùng kmeans_fe_model.pkl vì Step 05b có IQR clip
    # → feature space khác (36 features) so với Step 05 (35 features)
    print(f"\n[05b] ── KMeans Distance Variant (fit riêng) ───────────────────")
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    sil_scores = []
    k_range = range(2, 11)
    for k_try in k_range:
        km_try = KMeans(n_clusters=k_try, random_state=random_state, n_init=10)
        labels = km_try.fit_predict(X_train_enc.values)
        sil_scores.append(silhouette_score(X_train_enc.values, labels))
    best_k = list(k_range)[int(np.argmax(sil_scores))]
    best_k = max(best_k, 3)
    print(f"[05b] Best K (silhouette) = {best_k}  (score={max(sil_scores):.4f})")

    km = KMeans(n_clusters=best_k, random_state=random_state, n_init=10)
    km.fit(X_train_enc.values)          # fit trên pre-SMOTE train của 05b
    joblib.dump(km, os.path.join(MODEL_DIR, "kmeans_linear_model.pkl"))
    print(f"[05b] Saved: kmeans_linear_model.pkl (K={best_k}, fitted on 05b train)")

    k = best_k
    dist_cols = [f"dist_to_cluster_{i}" for i in range(k)]

    # Append distances vào pre-SMOTE train và test
    X_train_comb = X_train_enc.copy()
    X_test_comb  = X_test_enc.copy()
    X_train_comb[dist_cols] = km.transform(X_train_enc.values)
    X_test_comb[dist_cols]  = km.transform(X_test_enc.values)
    all_dist_cols = all_cols + dist_cols

    # SMOTE cho dist variant
    X_tr_dist_arr, y_tr_dist = smote.fit_resample(X_train_comb, y_train_clean)
    X_tr_dist = pd.DataFrame(X_tr_dist_arr, columns=all_dist_cols)
    print(f"[05b] After SMOTE (dist)  → "
          f"Class 0: {(y_tr_dist==0).sum()} | Class 1: {(y_tr_dist==1).sum()}")

    # ── Save data ─────────────────────────────────────────────────────────────
    X_tr_lin.to_csv(   os.path.join(DATA_DIR, "X_train_linear.csv"),      index=False)
    X_test_enc.to_csv( os.path.join(DATA_DIR, "X_test_linear.csv"),       index=False)
    X_tr_dist.to_csv(  os.path.join(DATA_DIR, "X_train_linear_dist.csv"), index=False)
    X_test_comb.to_csv(os.path.join(DATA_DIR, "X_test_linear_dist.csv"),  index=False)

    y_tr_lin_s  = pd.Series(y_tr_lin,  name=TARGET)
    y_tr_dist_s = pd.Series(y_tr_dist, name=TARGET)
    y_tr_lin_s.to_csv( os.path.join(DATA_DIR, "y_train_linear.csv"),      index=False)
    y_tr_dist_s.to_csv(os.path.join(DATA_DIR, "y_train_linear_dist.csv"), index=False)

    print(f"\n[05b] Saved: X_train_linear.csv  ({X_tr_lin.shape})")
    print(f"[05b] Saved: X_test_linear.csv   ({X_test_enc.shape})")
    print(f"[05b] Saved: X_train_linear_dist.csv  ({X_tr_dist.shape})")
    print(f"[05b] Saved: X_test_linear_dist.csv")

    # ── Save models ───────────────────────────────────────────────────────────
    joblib.dump(linear_scaler,  os.path.join(MODEL_DIR, "linear_scaler.pkl"))
    joblib.dump(linear_imputer, os.path.join(MODEL_DIR, "linear_imputer.pkl"))
    print(f"[05b] Saved: linear_scaler.pkl, linear_imputer.pkl")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n[05b] ── Summary ──────────────────────────────────────────────")
    print(f"       Outlier clip: {len(report)} numeric columns affected")
    print(f"       LR-Base : {len(all_cols)} features")
    print(f"       LR-Dist : {len(all_dist_cols)} features (+{k} dist cols)")
    print(f"\n{'='*60}\n[05b] ✓ Xong.\n{'='*60}\n")

    return {
        "X_train_linear":      X_tr_lin,
        "X_test_linear":       X_test_enc,
        "X_train_linear_dist": X_tr_dist,
        "X_test_linear_dist":  X_test_comb,
        "y_train_linear":      y_tr_lin_s,
        "y_train_linear_dist": y_tr_dist_s,
    }


if __name__ == "__main__":
    run_linear_preprocessing()