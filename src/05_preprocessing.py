"""
Step 5: Preprocessing Pipeline
- Feature engineering (device_per_tenure, cashback_per_order, inactivity_ratio)
- Stratified train-test split (80/20)
- OneHotEncoder (fit on train only)
- MinMaxScaler + KNNImputer (fit on train only)
- SMOTE to balance classes
- K-Means Feature Engineering (Strategy A: distance cols, Strategy B: cluster_id col)
- Save all fitted transformers as .pkl
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.impute import KNNImputer
from imblearn.over_sampling import SMOTE

from pipeline_config import DATA_DIR, MODEL_DIR, OUT_DIR   # noqa: E402
RFM_PATH   = os.path.join(DATA_DIR, "ecommerce_churn_rfm.csv")


def load(path=RFM_PATH):
    df = pd.read_csv(path, dtype={"CityTier": str, "Complain": str})
    df["Churn"] = df["Churn"].astype(int)
    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Create 3 engineered features ."""
    df = df.copy()
    df["device_per_tenure"]  = df["NumberOfDeviceRegistered"] / (df["Tenure"].fillna(0) + 1)
    df["cashback_per_order"] = df["CashbackAmount"] / (df["OrderCount"].fillna(0) + 1)
    df["inactivity_ratio"]   = df["DaySinceLastOrder"] / (df["Tenure"].fillna(0) + 1)
    print("[05] Engineered features: device_per_tenure, cashback_per_order, inactivity_ratio")
    return df


def prepare_features(df: pd.DataFrame):
    """Drop ID-like and label cols; separate X and y."""
    drop_cols = ["CustomerID", "Churn"]
    # Keep rfm_total, rfm_segment; drop R_score/F_score/M_score (redundant)
    drop_cols += [c for c in ["R_score", "F_score", "M_score"] if c in df.columns]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df["Churn"]
    return X, y


def run_preprocessing(path=RFM_PATH, random_state=42):
    df = load(path)
    print(f"\n{'='*60}")
    print("STEP 5: PREPROCESSING")
    print(f"{'='*60}")

    # ── Feature engineering ──────────────────────────────────────────────────
    df = feature_engineering(df)

    # ── Prepare X, y ─────────────────────────────────────────────────────────
    X, y = prepare_features(df)
    print(f"[05] Features: {X.shape[1]} | Samples: {X.shape[0]}")

    # ── Train-Test split (stratified 80/20) ──────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_state)
    print(f"[05] Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")
    print(f"[05] Before SMOTE → Class 0: {(y_train==0).sum()} | Class 1: {(y_train==1).sum()}")

    # ── Identify column groups ────────────────────────────────────────────────
    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    print(f"[05] Categorical columns: {cat_cols}")
    print(f"[05] Numeric columns   : {num_cols}")

    # ── OneHotEncoder (fit on TRAIN only) ────────────────────────────────────
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop="first")
    encoder.fit(X_train[cat_cols])

    ohe_feature_names = encoder.get_feature_names_out(cat_cols).tolist()

    def encode(X, enc, cat_cols, num_cols):
        cat_enc = pd.DataFrame(enc.transform(X[cat_cols]),
                               columns=ohe_feature_names, index=X.index)
        num_part = X[num_cols].reset_index(drop=True)
        cat_part = cat_enc.reset_index(drop=True)
        return pd.concat([num_part, cat_part], axis=1)

    X_train_enc = encode(X_train, encoder, cat_cols, num_cols)
    X_test_enc  = encode(X_test,  encoder, cat_cols, num_cols)

    # ── MinMaxScaler + KNNImputer (fit on TRAIN only) ────────────────────────
    num_cols_imp = X_train_enc.select_dtypes(include=[np.number]).columns.tolist()

    scaler  = MinMaxScaler()
    X_train_enc[num_cols_imp] = scaler.fit_transform(X_train_enc[num_cols_imp])
    X_test_enc[num_cols_imp]  = scaler.transform(X_test_enc[num_cols_imp])

    imputer = KNNImputer(n_neighbors=5)
    X_train_enc[num_cols_imp] = imputer.fit_transform(X_train_enc[num_cols_imp])
    X_test_enc[num_cols_imp]  = imputer.transform(X_test_enc[num_cols_imp])

    print(f"[05] Scaling + KNN imputation done. Missing remaining: "
          f"{X_train_enc.isnull().sum().sum()}")

    # ── SMOTE (only on training set) ─────────────────────────────────────────
    smote = SMOTE(random_state=random_state, k_neighbors=5)
    X_train_res, y_train_res = smote.fit_resample(X_train_enc, y_train)
    print(f"[05] After SMOTE  → Class 0: {(y_train_res==0).sum()} "
          f"| Class 1: {(y_train_res==1).sum()}")

    # ── Save artifacts ────────────────────────────────────────────────────────
    joblib.dump(encoder,  os.path.join(MODEL_DIR, "encoder.pkl"))
    joblib.dump(scaler,   os.path.join(MODEL_DIR, "scaler.pkl"))
    joblib.dump(imputer,  os.path.join(MODEL_DIR, "imputer.pkl"))

    # Save processed splits
    X_train_res_df = pd.DataFrame(X_train_res, columns=X_train_enc.columns)
    X_test_enc_df  = pd.DataFrame(X_test_enc,  columns=X_train_enc.columns)

    X_train_res_df.to_csv(os.path.join(DATA_DIR, "X_train.csv"), index=False)
    X_test_enc_df.to_csv( os.path.join(DATA_DIR, "X_test.csv"),  index=False)
    y_train_res.to_csv(   os.path.join(DATA_DIR, "y_train.csv"), index=False)
    y_test.to_csv(        os.path.join(DATA_DIR, "y_test.csv"),  index=False)

    # Also save raw (un-encoded) test set for clustering
    X_test.to_csv(os.path.join(DATA_DIR, "X_test_raw.csv"), index=False)
    y_test.to_csv(os.path.join(DATA_DIR, "y_test_raw.csv"), index=False)
    
    print(f"[05] Saved: encoder, scaler, imputer | X_train, X_test, y_train, y_test")
    print(f"[05] Feature count after encoding: {X_train_res_df.shape[1]}")

    # ── K-Means Feature Engineering (Strategies A & B) ───────────────────────
    print(f"\n[05] ── K-Means Feature Engineering ───────────────────────────")
    optimal_k, _, _ = find_optimal_k(X_train_enc.values, k_range=range(2, 11))
    optimal_k = max(optimal_k, 3)
    build_kmeans_features(
        X_train_pre_smote=X_train_enc,
        X_train_post_smote=X_train_res_df,
        X_test_enc=X_test_enc_df,
        optimal_k=optimal_k,
    )

    return (X_train_res_df, X_test_enc_df,
            y_train_res.reset_index(drop=True),
            y_test.reset_index(drop=True),
            {"encoder": encoder, "scaler": scaler, "imputer": imputer,
             "cat_cols": cat_cols, "num_cols": num_cols,
             "ohe_feature_names": ohe_feature_names,
             "optimal_k": optimal_k})


# ═══════════════════════════════════════════════════════════════════════════════
# K-Means Feature Engineering helpers
# ═══════════════════════════════════════════════════════════════════════════════

def find_optimal_k(X_scaled: np.ndarray, k_range=range(2, 11)) -> tuple:
    """
    Elbow (WCSS) + Silhouette sweep to select optimal K.
    Moved here from Step 7 so K is determined BEFORE model training.
    Saves outputs/05_elbow_silhouette.png as documented justification.
    """
    wcss_list, sil_list = [], []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        wcss_list.append(km.inertia_)
        sil_list.append(silhouette_score(X_scaled, labels))

    best_k = list(k_range)[int(np.argmax(sil_list))]

    # ── Plot Elbow + Silhouette ───────────────────────────────────────────────
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
    Fit KMeans on pre-SMOTE training data (no leakage), then apply to:
      • X_train_post_smote (SMOTE-augmented, same shape as X_train.csv)
      • X_test_enc

    Strategy A — distance features:
      Appends K columns  dist_to_cluster_0 … dist_to_cluster_{K-1}
      Saved as X_train_dist.csv / X_test_dist.csv

    Strategy B — cluster-ID feature:
      Appends 1 column  kmeans_cluster_id  (integer 0…K-1)
      Saved as X_train_clust.csv / X_test_clust.csv

    One shared model artifact: kmeans_fe_model.pkl
    (reused by Step 7 for business profiling without re-fitting)
    """
    # ── Fit on REAL training data only (pre-SMOTE) ────────────────────────────
    km = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    km.fit(X_train_pre_smote.values)

    dist_cols = [f"dist_to_cluster_{i}" for i in range(optimal_k)]

    # ── Strategy A: distances ─────────────────────────────────────────────────
    train_dists  = km.transform(X_train_post_smote.values)
    test_dists   = km.transform(X_test_enc.values)

    X_train_dist = X_train_post_smote.copy()
    X_train_dist[dist_cols] = train_dists

    X_test_dist  = X_test_enc.copy()
    X_test_dist[dist_cols]  = test_dists

    # ── Strategy B: cluster IDs ───────────────────────────────────────────────
    train_cluster_ids = km.predict(X_train_post_smote.values)
    test_cluster_ids  = km.predict(X_test_enc.values)

    X_train_clust = X_train_post_smote.copy()
    X_train_clust["kmeans_cluster_id"] = train_cluster_ids

    X_test_clust  = X_test_enc.copy()
    X_test_clust["kmeans_cluster_id"]  = test_cluster_ids

    # ── Save ─────────────────────────────────────────────────────────────────
    X_train_dist.to_csv(os.path.join(DATA_DIR, "X_train_dist.csv"),  index=False)
    X_test_dist.to_csv( os.path.join(DATA_DIR, "X_test_dist.csv"),   index=False)
    X_train_clust.to_csv(os.path.join(DATA_DIR, "X_train_clust.csv"), index=False)
    X_test_clust.to_csv( os.path.join(DATA_DIR, "X_test_clust.csv"),  index=False)
    joblib.dump(km, os.path.join(MODEL_DIR, "kmeans_fe_model.pkl"))

    print(f"[05] KMeans(K={optimal_k}) fitted on {len(X_train_pre_smote):,} pre-SMOTE samples.")
    print(f"[05] Strategy A: {len(dist_cols)} distance columns  "
          f"→ X_train_dist.csv / X_test_dist.csv")
    print(f"[05] Strategy B: 1 cluster_id column  "
          f"→ X_train_clust.csv / X_test_clust.csv")
    print(f"[05] Saved: kmeans_fe_model.pkl")

    return X_train_dist, X_test_dist, X_train_clust, X_test_clust, km


if __name__ == "__main__":
    run_preprocessing()
