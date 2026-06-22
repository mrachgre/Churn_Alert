"""
p2_00_prepare.py  —  Phase 2 Data Preparation
===============================================
Applies Phase 1 decisions to produce clean train/test sets:
  1. Recode 14 rating cols: 0 → NaN
  2. Create was_na_<col> binary indicators
  3. Drop Arrival Delay (keep Departure)
  4. Ordinal/binary encode categorical cols
  5. Stratified train/test split 80/20
  6. Save parquet checkpoints
"""
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split


def run_prepare(cfg) -> tuple[pd.DataFrame, pd.DataFrame]:
    sep = "=" * 70
    print(f"\n{sep}\nPHASE 2 — STEP 0: DATA PREPARATION\n{sep}")

    # ── Load raw ──────────────────────────────────────────────────────────────
    df = pd.read_csv(cfg.CSV_PATH)
    print(f"  Raw shape: {df.shape[0]:,} × {df.shape[1]}")

    # ── 1. Recode 0 → NaN for 14 rating cols ─────────────────────────────────
    rating_cols = [c for c in cfg.SERVICE_COLS if c in df.columns]
    print(f"\n[0.1] Recoding 0 → NaN for {len(rating_cols)} rating columns:")
    for col in rating_cols:
        n_zeros = (df[col] == 0).sum()
        df[f"was_na_{col}"] = (df[col] == 0).astype(int)
        df[col] = df[col].replace(0, np.nan)
        print(f"  {col:<45} recoded {n_zeros:,} zeros → NaN")

    was_na_cols = [f"was_na_{c}" for c in rating_cols]
    print(f"  Created {len(was_na_cols)} was_na_* indicator columns.")

    # ── 2. Drop Arrival Delay ─────────────────────────────────────────────────
    arr_col = "Arrival Delay in Minutes"
    if arr_col in df.columns:
        df = df.drop(columns=[arr_col])
        print(f"\n[0.2] Dropped '{arr_col}' (keeping Departure Delay only).")

    # ── 3. Encode target ──────────────────────────────────────────────────────
    df["target"] = df[cfg.TARGET_COL].map({"satisfied": 1, "dissatisfied": 0})
    df = df.drop(columns=[cfg.TARGET_COL])
    print(f"\n[0.3] Target encoded: satisfied=1, dissatisfied=0")
    print(f"  distribution: {dict(df['target'].value_counts().sort_index())}")

    # ── 4. Ordinal/binary encode categoricals ─────────────────────────────────
    df["Class_ord"]   = df["Class"].map(cfg.CLASS_MAP)
    df["loyal"]       = df["Customer Type"].map(cfg.CUSTOMER_TYPE_MAP)
    df["biz_travel"]  = df["Type of Travel"].map(cfg.TRAVEL_TYPE_MAP)
    df = df.drop(columns=["Class", "Customer Type", "Type of Travel"])
    print(f"\n[0.4] Encoded: Class_ord (Eco=0,EcoPlus=1,Business=2), "
          f"loyal (0/1), biz_travel (0/1)")

    # ── 5. Train / Test split ─────────────────────────────────────────────────
    y = df["target"]
    X = df.drop(columns=["target"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size    = cfg.TEST_SIZE,
        random_state = cfg.RANDOM_STATE,
        stratify     = y,
    )

    train_df = X_train.copy(); train_df["target"] = y_train.values
    test_df  = X_test.copy();  test_df["target"]  = y_test.values

    print(f"\n[0.5] Train/Test split (stratified 80/20, seed={cfg.RANDOM_STATE}):")
    for name, part in [("Train", train_df), ("Test", test_df)]:
        sat_rate = part["target"].mean()
        print(f"  {name}: {part.shape[0]:,} rows  "
              f"satisfied={sat_rate*100:.1f}%  dissatisfied={(1-sat_rate)*100:.1f}%")

    # ── 6. Save ───────────────────────────────────────────────────────────────
    tr_path = os.path.join(cfg.DATA_INT_DIR, "train_prepared.parquet")
    te_path = os.path.join(cfg.DATA_INT_DIR, "test_prepared.parquet")
    train_df.to_parquet(tr_path, index=False)
    test_df.to_parquet(te_path,  index=False)
    print(f"\n  Saved → {tr_path}")
    print(f"  Saved → {te_path}")
    print(f"  Columns ({train_df.shape[1]}): {list(train_df.columns[:8])} ...")

    print(f"\n{'─'*70}\nSTEP 0 DONE\n{'─'*70}")
    return train_df, test_df


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    import src_airline.airline_config as cfg
    run_prepare(cfg)
