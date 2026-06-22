"""
Step 4: Pattern Analysis of 393 Null Rows in Arrival Delay
===========================================================
- Profile the 393 null rows: Departure Delay distribution
- Check distribution across Class, Type of Travel, Customer Type
- Test for systematic (non-random) missing pattern via chi-square
NO imputation/drop — report only.
"""
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency


def run_step4(cfg, df: pd.DataFrame):
    sep = "=" * 70
    print(f"\n{sep}\nSTEP 4: NULL PATTERN ANALYSIS — Arrival Delay\n{sep}")

    arr_col = "Arrival Delay in Minutes"
    dep_col = "Departure Delay in Minutes"

    if arr_col not in df.columns:
        print(f"  ⚠ Column '{arr_col}' not found — skipping"); return

    null_mask    = df[arr_col].isnull()
    non_null_mask = ~null_mask
    n_null       = null_mask.sum()

    print(f"\n  Total null in '{arr_col}': {n_null:,}")
    print(f"  Total non-null            : {non_null_mask.sum():,}")

    # ── 4.1  Departure Delay distribution for null rows ───────────────────────
    print(f"\n[4.1] Departure Delay in the {n_null} null-Arrival rows:")
    if dep_col in df.columns:
        dep_null = df.loc[null_mask, dep_col]
        desc = dep_null.describe(percentiles=[.25, .5, .75, .9, .99])
        print(f"  describe():\n{desc.to_string()}")
        n_dep_zero = (dep_null == 0).sum()
        n_dep_large = (dep_null > 100).sum()
        print(f"\n  Departure Delay = 0   : {n_dep_zero:,}  ({n_dep_zero/n_null*100:.1f}%)")
        print(f"  Departure Delay > 100 : {n_dep_large:,}  ({n_dep_large/n_null*100:.1f}%)")

        # Compare vs non-null rows
        dep_non_null = df.loc[non_null_mask, dep_col]
        print(f"\n  Mean departure delay:")
        print(f"    Rows where Arrival=null  : {dep_null.mean():.2f} min")
        print(f"    Rows where Arrival≠null  : {dep_non_null.mean():.2f} min")
    else:
        print(f"  ⚠ '{dep_col}' not found")

    # ── 4.2  Sample of the 393 null rows ─────────────────────────────────────
    print(f"\n[4.2] Sample of null rows (up to 20 rows):")
    cols_show = [c for c in [dep_col, arr_col, "Class",
                              "Type of Travel", "Customer Type",
                              "satisfaction", "Flight Distance"] if c in df.columns]
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(df.loc[null_mask, cols_show].head(20).to_string(index=True))

    # ── 4.3  Distribution across categorical columns ───────────────────────────
    print(f"\n[4.3] Distribution of null vs non-null rows across categorical columns:")
    cat_check = [c for c in cfg.CAT_COLS if c in df.columns]
    for col in cat_check:
        print(f"\n  ── {col} ──")
        total_vc  = df[col].value_counts()
        null_vc   = df.loc[null_mask, col].value_counts()
        header    = f"  {'Value':<28}  {'Total':>8}  {'% of total':>10}  {'Null n':>7}  {'% null in group':>16}"
        print(header)
        print("  " + "─" * (len(header) - 2))
        for val in total_vc.index:
            tot  = total_vc.get(val, 0)
            nul  = null_vc.get(val, 0)
            pct_tot = tot / len(df) * 100
            pct_nul = nul / tot * 100 if tot > 0 else 0
            print(f"  {str(val):<28}  {tot:>8,}  {pct_tot:>9.1f}%  {nul:>7,}  {pct_nul:>15.3f}%")

        # Chi-square test: is null rate uniform across groups?
        contingency = pd.crosstab(df[col], null_mask)
        if contingency.shape == (contingency.shape[0], 2):
            chi2, p, dof, _ = chi2_contingency(contingency)
            print(f"\n  Chi-square test (null vs non-null across {col} groups):")
            print(f"    χ² = {chi2:.4f},  df = {dof},  p-value = {p:.4e}")
            if p < 0.05:
                print(f"    ⚠ p < 0.05 → null pattern is NOT random w.r.t. {col}")
                print(f"       (systematic missing — MAR or MNAR, not MCAR)")
            else:
                print(f"    ✅ p ≥ 0.05 → null appears random w.r.t. {col} (consistent with MCAR)")

    print(f"\n  NOTE: No imputation or drop performed. Options to decide:")
    print(f"    (a) Drop the 393 rows (~0.3% — minimal loss)")
    print(f"    (b) Impute with Departure Delay (given r≈0.96)")
    print(f"    (c) Impute with median/mean of Arrival Delay")

    print(f"\n{'─'*70}\nSTEP 4 DONE\n{'─'*70}")


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    import src_airline.airline_config as cfg
    df = pd.read_csv(cfg.CSV_PATH)
    run_step4(cfg, df)
