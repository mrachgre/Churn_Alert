"""
Step 3: Departure vs Arrival Delay Correlation & Derived Feature
================================================================
- Pearson correlation (expected ~0.96)
- describe() side-by-side
- delay_diff = Arrival - Departure (informational only, no decision yet)
NO transformations committed.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os


def run_step3(cfg, df: pd.DataFrame):
    sep = "=" * 70
    print(f"\n{sep}\nSTEP 3: DEPARTURE vs ARRIVAL DELAY — CORRELATION\n{sep}")

    dep_col = "Departure Delay in Minutes"
    arr_col = "Arrival Delay in Minutes"

    for c in [dep_col, arr_col]:
        if c not in df.columns:
            print(f"  ⚠ Column '{c}' not found — skipping step"); return

    # ── 3.1  Pearson correlation ──────────────────────────────────────────────
    print(f"\n[3.1] Pearson correlation (on rows without NaN in either column):")
    valid = df[[dep_col, arr_col]].dropna()
    corr  = valid[dep_col].corr(valid[arr_col])
    print(f"  n (valid pairs) : {len(valid):,}")
    print(f"  n (dropped NaN) : {len(df) - len(valid):,}")
    print(f"  Pearson r       : {corr:.6f}")
    if corr > 0.95:
        print(f"  ✅ Very high correlation ({corr:.3f}) — typical for delay data.")
        print(f"  → Consider keeping ONLY one column (usually Departure) or the derived diff.")
    elif corr > 0.80:
        print(f"  ⚠ High but not extreme ({corr:.3f}) — both columns carry some unique info.")
    else:
        print(f"  ⚠ Moderate/low correlation ({corr:.3f}) — unexpected, investigate further.")

    # ── 3.2  describe() side-by-side ─────────────────────────────────────────
    print(f"\n[3.2] describe() side-by-side (excluding NaN):")
    desc = df[[dep_col, arr_col]].describe(percentiles=[.5, .75, .9, .95, .99])
    with pd.option_context("display.float_format", "{:.2f}".format):
        print(desc.to_string())

    # ── 3.3  Derived feature: delay_diff ─────────────────────────────────────
    print(f"\n[3.3] Derived feature: delay_diff = Arrival - Departure")
    print(f"  (rows with NaN in Arrival are NaN in delay_diff too)")
    delay_diff = df[arr_col] - df[dep_col]
    desc_diff  = delay_diff.describe(percentiles=[.1, .25, .5, .75, .9, .99])
    print(desc_diff.to_string())

    n_neg = (delay_diff < 0).sum()
    n_pos = (delay_diff > 0).sum()
    n_zer = (delay_diff == 0).sum()
    print(f"\n  delay_diff < 0 (arrived earlier than departed late): {n_neg:,}  "
          f"({n_neg/len(df)*100:.1f}%)")
    print(f"  delay_diff = 0 (departure delay = arrival delay)    : {n_zer:,}  "
          f"({n_zer/len(df)*100:.1f}%)")
    print(f"  delay_diff > 0 (arrived later than departed)        : {n_pos:,}  "
          f"({n_pos/len(df)*100:.1f}%)")
    print(f"\n  NOTE: delay_diff computed for reference only. "
          f"No column added to df yet — await decision.")

    # ── 3.4  Scatter plot (sample to keep plot fast) ─────────────────────────
    sample_n = min(5_000, len(valid))
    samp     = valid.sample(sample_n, random_state=42)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].scatter(samp[dep_col], samp[arr_col], alpha=0.2, s=5, color="#3b82f6")
    axes[0].set_xlabel(dep_col)
    axes[0].set_ylabel(arr_col)
    axes[0].set_title(f"Scatter (sample n={sample_n:,})\nPearson r = {corr:.4f}")
    axes[0].set_xlim(-10, min(samp[dep_col].max() * 1.05, 800))
    axes[0].set_ylim(-10, min(samp[arr_col].max() * 1.05, 800))
    axes[0].plot([0, 800], [0, 800], "r--", linewidth=1, label="y=x")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].hist(delay_diff.dropna(), bins=80, color="#22c55e", edgecolor="none", alpha=0.8)
    axes[1].axvline(0, color="red", linestyle="--", linewidth=1.5, label="diff=0")
    axes[1].set_xlabel("delay_diff (Arrival − Departure, minutes)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Distribution of delay_diff")
    axes[1].legend(fontsize=8)
    axes[1].grid(axis="y", alpha=0.3)

    out_path = os.path.join(cfg.OUT_DIR, "03_delay_correlation.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"\n  Plot saved → {out_path}")

    print(f"\n{'─'*70}\nSTEP 3 DONE\n{'─'*70}")


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    import src_airline.airline_config as cfg
    df = pd.read_csv(cfg.CSV_PATH)
    run_step3(cfg, df)
