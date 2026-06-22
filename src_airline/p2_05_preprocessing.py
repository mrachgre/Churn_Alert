"""
p2_05_preprocessing.py  —  Step 5: Tree Path Preprocessing
===========================================================
① Impute composite scores (median, fit train) — KMeans needs non-NaN
② SMOTE evaluation — skip if minority class >= 40%
③ KMeans FE (Elbow + Silhouette K=2..8, fit train only)
   → Strategy A: distances to centroids
   → Strategy B: cluster_id
④ Save: encoder.pkl, imputer_composite.pkl, kmeans_model.pkl
   X_train_base, X_train_clust, X_train_dist (parquet)
   X_test_base,  X_test_clust,  X_test_dist
"""
import pandas as pd
import numpy as np
import os
import joblib
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def run_step5(cfg, train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    sep = "=" * 70
    print(f"\n{sep}\nSTEP 5: PREPROCESSING — TREE PATH\n{sep}")

    y_train = train_df["target"].values
    y_test  = test_df["target"].values
    X_train = train_df.drop(columns=["target"]).copy()
    X_test  = test_df.drop(columns=["target"]).copy()

    # ── ①  Impute composite scores ────────────────────────────────────────────
    print(f"\n[5.1] Impute composite scores (median, fit train):")
    composite_cols = ["ground_experience_score", "inflight_experience_score", "delay_severity"]
    composite_avail = [c for c in composite_cols if c in X_train.columns]

    imp_comp = SimpleImputer(strategy="median")
    X_train[composite_avail] = imp_comp.fit_transform(X_train[composite_avail])
    X_test[composite_avail]  = imp_comp.transform(X_test[composite_avail])
    joblib.dump(imp_comp, os.path.join(cfg.MODEL_DIR, "imputer_composite.pkl"))
    print(f"  Imputed {len(composite_avail)} cols with train median.")
    print(f"  Medians: { {c: round(float(imp_comp.statistics_[i]),3) for i, c in enumerate(composite_avail)} }")

    # ── ②  SMOTE evaluation ────────────────────────────────────────────────────
    print(f"\n[5.2] SMOTE evaluation:")
    minority_frac = min(y_train.mean(), 1 - y_train.mean())
    print(f"  Training set: dissatisfied={1-y_train.mean():.3f} ({(1-y_train.mean())*100:.1f}%)")
    print(f"  Satisfied   : {y_train.mean():.3f}  ({y_train.mean()*100:.1f}%)")
    print(f"  Minority fraction: {minority_frac:.3f}")

    apply_smote = minority_frac < (1 - cfg.SMOTE_THRESHOLD)
    if apply_smote:
        print(f"  → Minority < {(1-cfg.SMOTE_THRESHOLD)*100:.0f}% → SMOTE WILL BE APPLIED in modelling step")
    else:
        print(f"  → Minority ≥ {(1-cfg.SMOTE_THRESHOLD)*100:.0f}% → SMOTE SKIPPED")
        print(f"     Reason: class balance ({minority_frac*100:.1f}% / {(1-minority_frac)*100:.1f}%) "
              f"is acceptable. XGBoost handles this natively via scale_pos_weight.")

    # ── ③  KMeans Feature Engineering ────────────────────────────────────────
    print(f"\n[5.3] KMeans FE — Elbow + Silhouette (K=2..8):")
    km_cols = [c for c in cfg.KMEANS_FEATURES if c in X_train.columns]
    print(f"  KMeans input features ({len(km_cols)}): {km_cols}")

    X_km_train = X_train[km_cols].copy()
    X_km_test  = X_test[km_cols].copy()

    # Scale for KMeans
    km_scaler = StandardScaler()
    Xk_tr = km_scaler.fit_transform(X_km_train)
    Xk_te = km_scaler.transform(X_km_test)

    inertias, silhouettes = [], []
    K_range = range(2, 9)
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=cfg.RANDOM_STATE, n_init=10)
        labels = km.fit_predict(Xk_tr)
        inertias.append(km.inertia_)
        sil = silhouette_score(Xk_tr, labels, sample_size=5000, random_state=42)
        silhouettes.append(sil)
        print(f"  K={k}  inertia={km.inertia_:,.0f}  silhouette={sil:.4f}")

    best_k = list(K_range)[np.argmax(silhouettes)]
    print(f"\n  Best K by silhouette: {best_k}")

    # Plot elbow + silhouette
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(list(K_range), inertias, "o-", color="#3b82f6", linewidth=2)
    axes[0].set_title("Elbow Method"); axes[0].set_xlabel("K"); axes[0].set_ylabel("Inertia")
    axes[0].grid(alpha=0.3)
    axes[1].plot(list(K_range), silhouettes, "o-", color="#22c55e", linewidth=2)
    axes[1].axvline(best_k, color="red", linestyle="--", linewidth=1.5, label=f"Best K={best_k}")
    axes[1].set_title("Silhouette Score"); axes[1].set_xlabel("K"); axes[1].set_ylabel("Score")
    axes[1].legend(); axes[1].grid(alpha=0.3)
    plt.suptitle(f"Step 5: KMeans — Elbow & Silhouette (Best K={best_k})", fontweight="bold")
    plt.tight_layout()
    elbow_path = os.path.join(cfg.OUT_DIR, "p2_05_elbow_silhouette.png")
    plt.savefig(elbow_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved → {elbow_path}")

    # Fit final KMeans on train (pre-SMOTE)
    km_final = KMeans(n_clusters=best_k, random_state=cfg.RANDOM_STATE, n_init=10)
    km_final.fit(Xk_tr)
    joblib.dump(km_final, os.path.join(cfg.MODEL_DIR, "kmeans_model.pkl"))
    joblib.dump(km_scaler, os.path.join(cfg.MODEL_DIR, "kmeans_scaler.pkl"))

    # Strategy B: cluster_id
    X_train["kmeans_cluster_id"] = km_final.predict(Xk_tr)
    X_test["kmeans_cluster_id"]  = km_final.predict(Xk_te)

    # Strategy A: distances to centroids
    train_dists = km_final.transform(Xk_tr)
    test_dists  = km_final.transform(Xk_te)
    dist_cols = [f"dist_centroid_{i}" for i in range(best_k)]
    X_train_dist = X_train.copy()
    X_test_dist  = X_test.copy()
    for i, dc in enumerate(dist_cols):
        X_train_dist[dc] = train_dists[:, i]
        X_test_dist[dc]  = test_dists[:, i]

    # ── Save artifacts ────────────────────────────────────────────────────────
    tr_base_path  = os.path.join(cfg.DATA_INT_DIR, "X_train_base.parquet")
    te_base_path  = os.path.join(cfg.DATA_INT_DIR, "X_test_base.parquet")
    tr_clust_path = os.path.join(cfg.DATA_INT_DIR, "X_train_clust.parquet")
    te_clust_path = os.path.join(cfg.DATA_INT_DIR, "X_test_clust.parquet")
    tr_dist_path  = os.path.join(cfg.DATA_INT_DIR, "X_train_dist.parquet")
    te_dist_path  = os.path.join(cfg.DATA_INT_DIR, "X_test_dist.parquet")
    np.save(os.path.join(cfg.DATA_INT_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(cfg.DATA_INT_DIR, "y_test.npy"),  y_test)

    # Base = without cluster features
    base_drop = ["kmeans_cluster_id"] + dist_cols
    X_train_base = X_train.drop(columns=[c for c in base_drop if c in X_train.columns])
    X_test_base  = X_test.drop(columns=[c for c in base_drop if c in X_test.columns])

    X_train_base.to_parquet(tr_base_path, index=False)
    X_test_base.to_parquet(te_base_path,  index=False)
    X_train.to_parquet(tr_clust_path,     index=False)  # clust = base + cluster_id
    X_test.to_parquet(te_clust_path,      index=False)
    X_train_dist.to_parquet(tr_dist_path, index=False)
    X_test_dist.to_parquet(te_dist_path,  index=False)

    print(f"\n  X_train_base shape : {X_train_base.shape}")
    print(f"  X_train_clust shape: {X_train.shape}")
    print(f"  X_train_dist shape : {X_train_dist.shape}")

    result = {
        "apply_smote": apply_smote,
        "best_k":      best_k,
        "km_model":    km_final,
        "dist_cols":   dist_cols,
        "X_train_base":  X_train_base,
        "X_test_base":   X_test_base,
        "X_train_clust": X_train,
        "X_test_clust":  X_test,
        "X_train_dist":  X_train_dist,
        "X_test_dist":   X_test_dist,
        "y_train": y_train,
        "y_test":  y_test,
    }

    print(f"\n{'─'*70}\nSTEP 5 DONE  (best_k={best_k}, apply_smote={apply_smote})\n{'─'*70}")
    return result


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    import src_airline.airline_config as cfg
    train = pd.read_parquet(os.path.join(cfg.DATA_INT_DIR, "train_fe.parquet"))
    test  = pd.read_parquet(os.path.join(cfg.DATA_INT_DIR, "test_fe.parquet"))
    run_step5(cfg, train, test)
