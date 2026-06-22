"""
Step 5: Outlier Analysis — Flight Distance & Delay Columns
==========================================================
- describe() + custom percentiles (90, 95, 99, 99.9)
- Count extreme delays > 500 min
- Histograms (saved as PNG)
NO capping or log-transform — report only.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os


def run_step5(cfg, df: pd.DataFrame):
    sep = "=" * 70
    print(f"\n{sep}\nSTEP 5: OUTLIER ANALYSIS — Flight Distance & Delay\n{sep}")

    cols = [c for c in cfg.NUMERIC_COLS if c in df.columns]

    # ── 5.1  Custom percentile table ──────────────────────────────────────────
    print(f"\n[5.1] Extended percentile table:")
    pcts = [.10, .25, .50, .75, .90, .95, .99, .999]
    stats_rows = []
    for col in cols:
        s   = df[col].dropna()
        row = {
            "column":  col,
            "count":   len(s),
            "nulls":   df[col].isnull().sum(),
            "mean":    s.mean(),
            "std":     s.std(),
            "min":     s.min(),
            "max":     s.max(),
        }
        for p in pcts:
            row[f"p{int(p*1000):04d}"] = s.quantile(p)
        stats_rows.append(row)

    header_map = {f"p{int(p*1000):04d}": f"p{int(p*100)}" if p < 1
                  else f"p{p*100:.1f}" for p in pcts}
    for row in stats_rows:
        print(f"\n  ── {row['column']} ──")
        print(f"    n={row['count']:,}  nulls={row['nulls']:,}  "
              f"mean={row['mean']:.2f}  std={row['std']:.2f}")
        print(f"    min={row['min']:.2f}  max={row['max']:.2f}")
        pct_str = "  ".join(
            f"p{int(p*100) if p<1 else int(p*100)}={row[f'p{int(p*1000):04d}']:.1f}"
            for p in pcts
        )
        print(f"    {pct_str}")

    # ── 5.2  Extreme delay count ───────────────────────────────────────────────
    print(f"\n[5.2] Extreme delay analysis (threshold = {cfg.DELAY_EXTREME_THRESHOLD} min):")
    for col in ["Departure Delay in Minutes", "Arrival Delay in Minutes"]:
        if col not in df.columns:
            continue
        s      = df[col].dropna()
        n_ext  = (s > cfg.DELAY_EXTREME_THRESHOLD).sum()
        pct    = n_ext / len(df) * 100
        n_zero = (s == 0).sum()
        pct_z  = n_zero / len(df) * 100
        print(f"\n  {col}:")
        print(f"    delay = 0            : {n_zero:,}  ({pct_z:.1f}%)")
        print(f"    delay > {cfg.DELAY_EXTREME_THRESHOLD} min      : {n_ext:,}  ({pct:.3f}%)  "
              f"{'⚠ investigate' if n_ext > 10 else '✅ very few'}")
        if n_ext > 0:
            extreme_rows = df[df[col] > cfg.DELAY_EXTREME_THRESHOLD][[
                c for c in ["Class", "Type of Travel", col,
                             "Flight Distance", "satisfaction"]
                if c in df.columns
            ]]
            print(f"    Sample of rows with delay > {cfg.DELAY_EXTREME_THRESHOLD} min:")
            print(extreme_rows.head(10).to_string(index=True))

    # ── 5.3  Histograms ───────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    plot_defs = [
        ("Flight Distance",           50,  "#3b82f6"),
        ("Departure Delay in Minutes", 80, "#f59e0b"),
        ("Arrival Delay in Minutes",   80, "#ef4444"),
    ]

    for ax, (col, bins, color) in zip(axes, plot_defs):
        if col not in df.columns:
            ax.set_title(f"{col}\nNOT FOUND")
            continue
        data = df[col].dropna()
        ax.hist(data, bins=bins, color=color, edgecolor="none", alpha=0.85)
        p95 = data.quantile(0.95)
        p99 = data.quantile(0.99)
        ax.axvline(p95, color="white", linestyle="--", linewidth=1.5,
                   label=f"p95={p95:.0f}")
        ax.axvline(p99, color="yellow", linestyle=":", linewidth=1.5,
                   label=f"p99={p99:.0f}")
        ax.set_title(f"{col}\n(n={len(data):,}, max={data.max():.0f})")
        ax.set_xlabel(col)
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Step 5: Outlier Analysis — Distribution of Numeric Columns",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    out_path = os.path.join(cfg.OUT_DIR, "05_outliers_distribution.png")
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"\n[5.3] Histogram saved → {out_path}")

    # ── 5.4  Flight Distance outlier check ────────────────────────────────────
    if "Flight Distance" in df.columns:
        fd = df["Flight Distance"].dropna()
        print(f"\n[5.4] Flight Distance:")
        print(f"    min = {fd.min():.0f}  |  max = {fd.max():.0f}")
        print(f"    p99 = {fd.quantile(.99):.0f}  |  p99.9 = {fd.quantile(.999):.0f}")
        n_short = (fd < 100).sum()
        n_long  = (fd > 5000).sum()
        print(f"    < 100 (very short) : {n_short:,}  ({n_short/len(df)*100:.2f}%)")
        print(f"    > 5000 (very long) : {n_long:,}  ({n_long/len(df)*100:.2f}%)")

    print(f"\n  NOTE: No capping or log-transform applied. Decisions needed:")
    print(f"    - Whether to cap extreme delays (> {cfg.DELAY_EXTREME_THRESHOLD} min)")
    print(f"    - Whether to log-transform right-skewed delays for linear models")
    print(f"    - (Tree/XGBoost models handle skew natively — no transform needed)")

    print(f"\n{'─'*70}\nSTEP 5 DONE\n{'─'*70}")


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    import src_airline.airline_config as cfg
    df = pd.read_csv(cfg.CSV_PATH)
    run_step5(cfg, df)
