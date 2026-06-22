"""
Step 7: Summary Report
======================
Consolidates all findings from Steps 1-6.
Lists every issue found and pending decisions.
Pipeline STOPS HERE — no encoding/modeling.
"""
import pandas as pd
import numpy as np
import os


def run_step7(cfg, df: pd.DataFrame):
    sep = "=" * 70
    print(f"\n{sep}")
    print("STEP 7: SUMMARY REPORT — ALL FINDINGS")
    print(f"{sep}")

    target_binary = df[cfg.TARGET_COL].map({"satisfied": 1, "dissatisfied": 0}) \
        if cfg.TARGET_COL in df.columns else None

    # ── 7.1  Dataset overview ─────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print("7.1  DATASET OVERVIEW")
    print(f"{'─'*70}")
    print(f"  Rows    : {df.shape[0]:,}  (README: {cfg.REF['total_rows']:,}  "
          f"{'✅' if df.shape[0]==cfg.REF['total_rows'] else '⚠'})")
    print(f"  Columns : {df.shape[1]}")
    if target_binary is not None:
        sat_rate = target_binary.mean()
        print(f"  Target  : {cfg.TARGET_COL}  — "
              f"satisfied={int(target_binary.sum()):,} ({sat_rate*100:.1f}%)  "
              f"dissatisfied={int((target_binary==0).sum()):,} ({(1-sat_rate)*100:.1f}%)")
    n_dup = df.duplicated().sum()
    print(f"  Duplicates (full row): {n_dup:,}  ({n_dup/len(df)*100:.3f}%)")

    # ── 7.2  Null summary ─────────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print("7.2  NULL VALUES")
    print(f"{'─'*70}")
    null_s = df.isnull().sum()
    nullcols = null_s[null_s > 0]
    if len(nullcols) == 0:
        print("  ✅ No nulls found.")
    for col, n in nullcols.items():
        expected = col == "Arrival Delay in Minutes"
        print(f"  {col:<45}  {n:>6} ({n/len(df)*100:.3f}%)  "
              f"{'✅ expected' if expected else '⚠ UNEXPECTED'}")
    print(f"\n  Pending decision: impute Arrival Delay (options: drop, median, Departure Delay)")

    # ── 7.3  Rating zeros ─────────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print("7.3  RATING ZEROS (0 = 'Not Applicable'?)")
    print(f"{'─'*70}")
    zero_pcts = {}
    for col in cfg.SERVICE_COLS:
        if col in df.columns:
            n0 = (df[col] == 0).sum()
            zero_pcts[col] = n0 / len(df) * 100
    if zero_pcts:
        for col, pct in sorted(zero_pcts.items(), key=lambda x: -x[1]):
            flag = "⚠ high" if pct > 5 else ""
            print(f"  {col:<45}  {pct:>6.2f}%  {flag}")
    bag_col = "Baggage handling"
    if bag_col in df.columns:
        n_bag_zero = (df[bag_col] == 0).sum()
        print(f"\n  '{bag_col}' zeros: {n_bag_zero}  "
              f"{'✅ none (matches README min=1)' if n_bag_zero==0 else '⚠ found zeros!'}")

    # Case study result
    if cfg.WIFI_COL in df.columns and target_binary is not None:
        r0 = target_binary[df[cfg.WIFI_COL]==0].mean() if (df[cfg.WIFI_COL]==0).any() else None
        r1 = target_binary[df[cfg.WIFI_COL]==1].mean() if (df[cfg.WIFI_COL]==1).any() else None
        r3 = target_binary[df[cfg.WIFI_COL]==3].mean() if (df[cfg.WIFI_COL]==3).any() else None
        if all(x is not None for x in [r0, r1, r3]):
            supports_na = abs(r0 - r3) < abs(r0 - r1)
            print(f"\n  '{cfg.WIFI_COL}' case study (sat rate at rating=0 vs 1 vs 3):")
            print(f"    rating=0: {r0:.3f}  rating=1: {r1:.3f}  rating=3: {r3:.3f}")
            print(f"    → {'✅ SUPPORTS N/A hypothesis' if supports_na else '⚠ CONTRADICTS N/A hypothesis'}")
    print(f"\n  Pending decision: (a) treat 0 as-is  (b) recode 0→NaN + indicator  "
          f"(c) separate category")

    # ── 7.4  Multicollinearity ────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print("7.4  MULTICOLLINEARITY — Departure vs Arrival Delay")
    print(f"{'─'*70}")
    dep_col = "Departure Delay in Minutes"
    arr_col = "Arrival Delay in Minutes"
    if dep_col in df.columns and arr_col in df.columns:
        valid = df[[dep_col, arr_col]].dropna()
        corr  = valid[dep_col].corr(valid[arr_col])
        print(f"  Pearson r = {corr:.4f}  {'⚠ HIGH — consider dropping one' if corr > 0.90 else ''}")
    print(f"  Pending decision: (a) keep both  (b) keep Departure only  "
          f"(c) use delay_diff  (d) keep Arrival only")

    # ── 7.5  Outliers ─────────────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print("7.5  OUTLIERS")
    print(f"{'─'*70}")
    for col in cfg.NUMERIC_COLS:
        if col not in df.columns:
            continue
        s     = df[col].dropna()
        n_ext = (s > cfg.DELAY_EXTREME_THRESHOLD).sum() \
            if "Delay" in col else 0
        print(f"  {col:<45}  min={s.min():.0f}  max={s.max():.0f}  "
              f"p99={s.quantile(.99):.0f}  "
              f"(n > {cfg.DELAY_EXTREME_THRESHOLD}min: {n_ext})" if "Delay" in col
              else f"  {col:<45}  min={s.min():.0f}  max={s.max():.0f}  p99={s.quantile(.99):.0f}")
    print(f"  Pending decision: (a) keep as-is (tree models)  "
          f"(b) cap at p99  (c) log-transform (linear models)")

    # ── 7.6  Charts generated ─────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print("7.6  CHARTS GENERATED (outputs/AIRLINE/)")
    print(f"{'─'*70}")
    for f in sorted(os.listdir(cfg.OUT_DIR)):
        if f.endswith(".png"):
            print(f"  {f}")

    # ── 7.7  Pending decisions checklist ──────────────────────────────────────
    print(f"\n{'─'*70}")
    print("7.7  DECISIONS NEEDED BEFORE ENCODING/MODELING")
    print(f"{'─'*70}")
    decisions = [
        ("NULL — Arrival Delay (393 rows)",
         "Drop rows / impute with Departure Delay / impute with median"),
        ("RATING ZERO — 14 service cols",
         "Treat as-is / recode 0→NaN + indicator col / keep as separate level"),
        ("MULTICOLLINEARITY — Departure vs Arrival Delay",
         "Keep both / drop one / use delay_diff / use only Departure"),
        ("OUTLIERS — extreme delays (> 500 min)",
         "Keep as-is / cap at p99 / log-transform"),
        ("DUPLICATES — fully identical rows",
         "Keep all / drop duplicates (minimal impact either way)"),
        ("TARGET ENCODING",
         "satisfied→1 / dissatisfied→0 (straightforward)"),
        ("CATEGORICAL ENCODING",
         "OHE / Ordinal (Class has natural order: Eco < Eco Plus < Business)"),
    ]
    for i, (issue, options) in enumerate(decisions, 1):
        print(f"\n  {i}. {issue}")
        print(f"     Options: {options}")

    print(f"\n{'='*70}")
    print("PIPELINE PAUSED — Review findings above before proceeding to Part 2")
    print("(Encoding + Modeling).")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    import src_airline.airline_config as cfg
    df = pd.read_csv(cfg.CSV_PATH)
    run_step7(cfg, df)
