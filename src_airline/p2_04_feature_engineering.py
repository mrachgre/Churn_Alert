"""
p2_04_feature_engineering.py  —  Step 4: Composite Features
=============================================================
Creates 3 aggregate features to replace RFM:
  - ground_experience_score   (mean of 5 ground/booking cols, skipna)
  - inflight_experience_score (mean of 7 inflight cols, skipna)
  - delay_severity            (departure_delay / (flight_distance + 1))
Applied identically on train and test (no fitting required).
"""
import pandas as pd
import numpy as np
import os


def _add_features(df: pd.DataFrame, cfg) -> pd.DataFrame:
    df = df.copy()

    # 1. Ground experience score
    g_avail = [c for c in cfg.GROUND_COLS if c in df.columns]
    df["ground_experience_score"] = df[g_avail].mean(axis=1, skipna=True)

    # 2. Inflight experience score
    i_avail = [c for c in cfg.INFLIGHT_COLS if c in df.columns]
    df["inflight_experience_score"] = df[i_avail].mean(axis=1, skipna=True)

    # 3. Delay severity (relative to flight distance)
    dep_col = "Departure Delay in Minutes"
    fd_col  = "Flight Distance"
    if dep_col in df.columns and fd_col in df.columns:
        df["delay_severity"] = df[dep_col] / (df[fd_col] + 1)
    else:
        df["delay_severity"] = np.nan

    return df


def run_step4(cfg, train_df: pd.DataFrame, test_df: pd.DataFrame
              ) -> tuple[pd.DataFrame, pd.DataFrame]:
    sep = "=" * 70
    print(f"\n{sep}\nSTEP 4: FEATURE ENGINEERING — COMPOSITE SCORES\n{sep}")

    print(f"\n  Ground cols  ({len(cfg.GROUND_COLS)}): {cfg.GROUND_COLS}")
    print(f"  Inflight cols({len(cfg.INFLIGHT_COLS)}): {cfg.INFLIGHT_COLS}")
    print(f"  delay_severity = Departure Delay / (Flight Distance + 1)")

    train_df = _add_features(train_df, cfg)
    test_df  = _add_features(test_df,  cfg)

    new_cols = ["ground_experience_score", "inflight_experience_score", "delay_severity"]

    print(f"\n[4.1] describe() of 3 new composite columns (train set):")
    print(train_df[new_cols].describe(percentiles=[.25, .5, .75, .95]).round(4).to_string())

    print(f"\n[4.2] NaN count in composite columns:")
    for col in new_cols:
        n_nan_tr = train_df[col].isna().sum()
        n_nan_te = test_df[col].isna().sum()
        pct_tr   = n_nan_tr / len(train_df) * 100
        pct_te   = n_nan_te / len(test_df)  * 100
        note     = ""
        if col == "ground_experience_score":
            note = "← NaN only if ALL 5 ground cols were NaN"
        elif col == "inflight_experience_score":
            note = "← NaN only if ALL 7 inflight cols were NaN"
        print(f"  {col:<35}  train={n_nan_tr:,} ({pct_tr:.3f}%)  "
              f"test={n_nan_te:,} ({pct_te:.3f}%)  {note}")

    # Save
    tr_path = os.path.join(cfg.DATA_INT_DIR, "train_fe.parquet")
    te_path = os.path.join(cfg.DATA_INT_DIR, "test_fe.parquet")
    train_df.to_parquet(tr_path, index=False)
    test_df.to_parquet(te_path,  index=False)
    print(f"\n  Saved → {tr_path}")
    print(f"  Saved → {te_path}")
    print(f"  Total columns now: {train_df.shape[1]}")

    print(f"\n{'─'*70}\nSTEP 4 DONE\n{'─'*70}")
    return train_df, test_df


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    import src_airline.airline_config as cfg
    train = pd.read_parquet(os.path.join(cfg.DATA_INT_DIR, "train_prepared.parquet"))
    test  = pd.read_parquet(os.path.join(cfg.DATA_INT_DIR, "test_prepared.parquet"))
    run_step4(cfg, train, test)
