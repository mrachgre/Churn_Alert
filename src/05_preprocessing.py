"""
Step 5: Preprocessing Pipeline
================================
Input : ecommerce_churn_train.csv  (split sẵn từ Step 02)
        ecommerce_churn_test.csv   (split sẵn từ Step 02)

Việc làm:
  1. Feature engineering (device_per_tenure, cashback_per_order, inactivity_ratio)
  2. Xử lý missing theo cơ chế đã chẩn đoán ở Step 02:
       • 6 cột MAR/MNAR → tạo cờ {col}_was_missing rồi impute mean
       • DaySinceLastOrder (~MCAR) → impute mean, không cần cờ
  3. OneHotEncoder  (fit on train only)
  4. SimpleImputer  (fit on train only) — mean strategy
  5. MinMaxScaler   (fit on train only)
  6. SMOTE          (chỉ áp lên train)
  7. K-Means Feature Engineering:
       • find_optimal_k: Elbow + Silhouette → 05_elbow_silhouette.png
       • Fit KMeans trên pre-SMOTE train (không leakage)
       • Strategy A: K cột dist_to_cluster_i  → X_train_dist / X_test_dist
       • Strategy B: 1 cột kmeans_cluster_id  → X_train_clust / X_test_clust
  8. Lưu artifacts: encoder, scaler, imputer, kmeans_fe_model (.pkl)
                   X/y train/test + dist/clust variants (.csv)
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE

from pipeline_config import DATA_DIR, MODEL_DIR, OUT_DIR   # noqa: E402

TRAIN_PATH = os.path.join(DATA_DIR, "ecommerce_churn_train.csv")
TEST_PATH  = os.path.join(DATA_DIR, "ecommerce_churn_test.csv")

TARGET       = "Churn"
CAT_AS_STR   = {"CityTier": str, "Complain": str}
RANDOM_STATE = 42

# Cột MAR/MNAR → cần tạo cờ was_missing trước khi impute
MAR_MNAR_COLS = [
    "Tenure",
    "OrderCount",
    "OrderAmountHikeFromlastYear",
    "CouponUsed",
    "WarehouseToHome",
    "HourSpendOnApp",
]

# Cột ~MCAR → impute mean, không cần cờ
MCAR_COLS = ["DaySinceLastOrder"]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=CAT_AS_STR)
    df[TARGET] = df[TARGET].astype(int)
    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Tạo 3 engineered features."""
    df = df.copy()
    df["device_per_tenure"]  = df["NumberOfDeviceRegistered"] / (df["Tenure"].fillna(0) + 1)
    df["cashback_per_order"] = df["CashbackAmount"] / (df["OrderCount"].fillna(0) + 1)
    df["inactivity_ratio"]   = df["DaySinceLastOrder"] / (df["Tenure"].fillna(0) + 1)
    print("[05] Engineered features: device_per_tenure, cashback_per_order, inactivity_ratio")
    return df


def add_missing_flags(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Tạo cột binary {col}_was_missing = 1 nếu giá trị gốc là NaN."""
    df = df.copy()
    for col in cols:
        if col in df.columns:
            df[f"{col}_was_missing"] = df[col].isnull().astype(int)
    created = [f"{c}_was_missing" for c in cols if c in df.columns]
    print(f"[05] Created missing-indicator flags: {created}")
    return df


def prepare_features(df: pd.DataFrame):
    """Tách X và y, bỏ cột CustomerID."""
    drop_cols = ["CustomerID", TARGET]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df[TARGET]
    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# K-Means helpers
# ─────────────────────────────────────────────────────────────────────────────
def find_optimal_k(X_scaled: np.ndarray, k_range=range(2, 11)) -> tuple:
    """
    Elbow (WCSS) + Silhouette sweep để chọn K tối ưu.
    Lưu outputs/05_elbow_silhouette.png làm bằng chứng lựa chọn.
    """
    wcss_list, sil_list = [], []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X_scaled)
        wcss_list.append(km.inertia_)
        sil_list.append(silhouette_score(X_scaled, labels))

    best_k = list(k_range)[int(np.argmax(sil_list))]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(list(k_range), wcss_list, "o-", color="#3b82f6", linewidth=2, markersize=7)
    ax1.set_title("Elbow Method (WCSS)", fontweight="bold")
    ax1.set_xlabel("Number of Clusters (K)")
    ax1.set_ylabel("WCSS / Inertia")
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    ax2.plot(list(k_range), sil_list, "s-", color="#22c55e", linewidth=2, markersize=7)
    ax2.set_title("Silhouette Score", fontweight="bold")
    ax2.set_xlabel("Number of Clusters (K)")
    ax2.set_ylabel("Silhouette Score")
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    ax2.axvline(best_k, color="#ef4444", linestyle="--", label=f"Best K={best_k}")
    ax2.legend()

    plt.suptitle("Optimal K Selection for K-Means Clustering", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "05_elbow_silhouette.png"), dpi=150)
    plt.close()
    print(f"[05] Saved: 05_elbow_silhouette.png")
    print(f"[05] Best K by silhouette: {best_k}  (score={max(sil_list):.4f})")
    return best_k, sil_list, wcss_list


def build_kmeans_features(
    X_train_pre_smote: pd.DataFrame,
    X_train_post_smote: pd.DataFrame,
    X_test_enc: pd.DataFrame,
    optimal_k: int,
) -> tuple:
    """
    Fit KMeans trên pre-SMOTE training data (không leakage), rồi apply lên:
      • X_train_post_smote (đã SMOTE, shape khớp X_train.csv)
      • X_test_enc

    Strategy A — distance features:
      Thêm K cột  dist_to_cluster_0 … dist_to_cluster_{K-1}
      → X_train_dist.csv / X_test_dist.csv

    Strategy B — cluster-ID feature:
      Thêm 1 cột  kmeans_cluster_id  (integer 0…K-1)
      → X_train_clust.csv / X_test_clust.csv

    Artifact dùng chung: kmeans_fe_model.pkl
    """
    km = KMeans(n_clusters=optimal_k, random_state=RANDOM_STATE, n_init=10)
    km.fit(X_train_pre_smote.values)

    dist_cols = [f"dist_to_cluster_{i}" for i in range(optimal_k)]

    # Strategy A: distances
    X_train_dist = X_train_post_smote.copy()
    X_train_dist[dist_cols] = km.transform(X_train_post_smote.values)

    X_test_dist = X_test_enc.copy()
    X_test_dist[dist_cols] = km.transform(X_test_enc.values)

    # Strategy B: cluster IDs
    X_train_clust = X_train_post_smote.copy()
    X_train_clust["kmeans_cluster_id"] = km.predict(X_train_post_smote.values)

    X_test_clust = X_test_enc.copy()
    X_test_clust["kmeans_cluster_id"] = km.predict(X_test_enc.values)

    # Save
    X_train_dist.to_csv( os.path.join(DATA_DIR, "X_train_dist.csv"),  index=False)
    X_test_dist.to_csv(  os.path.join(DATA_DIR, "X_test_dist.csv"),   index=False)
    X_train_clust.to_csv(os.path.join(DATA_DIR, "X_train_clust.csv"), index=False)
    X_test_clust.to_csv( os.path.join(DATA_DIR, "X_test_clust.csv"),  index=False)
    joblib.dump(km, os.path.join(MODEL_DIR, "kmeans_fe_model.pkl"))

    print(f"[05] KMeans(K={optimal_k}) fitted on {len(X_train_pre_smote):,} pre-SMOTE samples.")
    print(f"[05] Strategy A: {len(dist_cols)} distance columns → X_train_dist.csv / X_test_dist.csv")
    print(f"[05] Strategy B: 1 cluster_id column → X_train_clust.csv / X_test_clust.csv")
    print(f"[05] Saved: kmeans_fe_model.pkl")

    return X_train_dist, X_test_dist, X_train_clust, X_test_clust, km


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def run_preprocessing(random_state=RANDOM_STATE):
    print(f"\n{'='*60}")
    print("STEP 5: PREPROCESSING")
    print(f"{'='*60}")

    # ── Load ──────────────────────────────────────────────────────────────────
    train_raw = load(TRAIN_PATH)
    test_raw  = load(TEST_PATH)
    print(f"[05] Train: {train_raw.shape} | Test: {test_raw.shape}")

    # ── Feature engineering ───────────────────────────────────────────────────
    train_raw = feature_engineering(train_raw)
    test_raw  = feature_engineering(test_raw)

    # ── Missing-indicator flags (MAR/MNAR) — trước mọi imputation ────────────
    train_raw = add_missing_flags(train_raw, MAR_MNAR_COLS)
    test_raw  = add_missing_flags(test_raw,  MAR_MNAR_COLS)

    # ── Tách X, y ─────────────────────────────────────────────────────────────
    X_train, y_train = prepare_features(train_raw)
    X_test,  y_test  = prepare_features(test_raw)
    print(f"[05] Features: {X_train.shape[1]} | "
          f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")
    print(f"[05] Before SMOTE → Class 0: {(y_train==0).sum()} | Class 1: {(y_train==1).sum()}")
    # ── Lưu raw (human-readable) features cho Step 7/8 ─────────────────────
    # Đây là dữ liệu SAU feature engineering + missing-flags, TRƯỚC encode/
    # impute/scale — đúng cái Step 07 (cluster profiling) và Step 08
    # (actionable risk matrix) cần để hiển thị số liệu gốc cho business.
    X_train.to_csv(os.path.join(DATA_DIR, "X_train_raw.csv"), index=False)
    X_test.to_csv( os.path.join(DATA_DIR, "X_test_raw.csv"),  index=False)
    print(f"[05] Saved: X_train_raw.csv {X_train.shape} | X_test_raw.csv {X_test.shape}")
    # ── Identify column groups ────────────────────────────────────────────────
    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    print(f"[05] Categorical: {cat_cols}")
    print(f"[05] Numeric    : {num_cols}")

    # ── OneHotEncoder (fit on TRAIN only) ────────────────────────────────────
    encoder  = OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop="first")
    encoder.fit(X_train[cat_cols])
    ohe_names = encoder.get_feature_names_out(cat_cols).tolist()

    def encode(X):
        cat_enc = pd.DataFrame(encoder.transform(X[cat_cols]),
                               columns=ohe_names, index=X.index)
        return pd.concat([X[num_cols].reset_index(drop=True),
                          cat_enc.reset_index(drop=True)], axis=1)

    X_train_enc = encode(X_train)
    X_test_enc  = encode(X_test)

    # ── SimpleImputer mean (fit on TRAIN only) ────────────────────────────────
    num_cols_enc = X_train_enc.select_dtypes(include=[np.number]).columns.tolist()

    imputer = SimpleImputer(strategy="mean")
    X_train_enc[num_cols_enc] = imputer.fit_transform(X_train_enc[num_cols_enc])
    X_test_enc[num_cols_enc]  = imputer.transform(X_test_enc[num_cols_enc])
    print(f"[05] Imputation done (mean). Missing remaining: {X_train_enc.isnull().sum().sum()}")

    # ── MinMaxScaler (fit on TRAIN only) ─────────────────────────────────────
    scaler = MinMaxScaler()
    X_train_enc[num_cols_enc] = scaler.fit_transform(X_train_enc[num_cols_enc])
    X_test_enc[num_cols_enc]  = scaler.transform(X_test_enc[num_cols_enc])
    print(f"[05] MinMaxScaler applied.")

    # ── SMOTE (chỉ áp lên train) ──────────────────────────────────────────────
    smote = SMOTE(random_state=random_state, k_neighbors=5)
    X_train_res, y_train_res = smote.fit_resample(X_train_enc, y_train)
    print(f"[05] After SMOTE → Class 0: {(y_train_res==0).sum()} "
          f"| Class 1: {(y_train_res==1).sum()}")

    # ── Save base artifacts ───────────────────────────────────────────────────
    joblib.dump(encoder, os.path.join(MODEL_DIR, "encoder.pkl"))
    joblib.dump(scaler,  os.path.join(MODEL_DIR, "scaler.pkl"))
    joblib.dump(imputer, os.path.join(MODEL_DIR, "imputer.pkl"))

    X_train_res_df = pd.DataFrame(X_train_res, columns=X_train_enc.columns)
    y_train_res_s  = pd.Series(y_train_res, name=TARGET)

    X_train_res_df.to_csv(os.path.join(DATA_DIR, "X_train.csv"), index=False)
    X_test_enc.to_csv(    os.path.join(DATA_DIR, "X_test.csv"),  index=False)
    y_train_res_s.to_csv( os.path.join(DATA_DIR, "y_train.csv"), index=False)
    y_test.to_csv(        os.path.join(DATA_DIR, "y_test.csv"),  index=False)

    print(f"[05] Saved: encoder.pkl, scaler.pkl, imputer.pkl")
    print(f"[05] Saved: X_train.csv {X_train_res_df.shape} | X_test.csv {X_test_enc.shape}")
    print(f"[05] Feature count after encoding: {X_train_res_df.shape[1]}")

    # ── K-Means Feature Engineering ───────────────────────────────────────────
    print(f"\n[05] ── K-Means Feature Engineering ────────────────────────────")
    optimal_k, _, _ = find_optimal_k(X_train_enc.values, k_range=range(2, 11))
    optimal_k = max(optimal_k, 3)
    build_kmeans_features(
        X_train_pre_smote=X_train_enc,
        X_train_post_smote=X_train_res_df,
        X_test_enc=X_test_enc,
        optimal_k=optimal_k,
    )

    print(f"\n{'='*60}\n[05] ✓ Xong.\n{'='*60}\n")

    return (X_train_res_df, X_test_enc,
            y_train_res_s.reset_index(drop=True),
            y_test.reset_index(drop=True),
            {"encoder": encoder, "scaler": scaler, "imputer": imputer,
             "cat_cols": cat_cols, "num_cols": num_cols,
             "ohe_feature_names": ohe_names,
             "optimal_k": optimal_k})


if __name__ == "__main__":
    run_preprocessing()