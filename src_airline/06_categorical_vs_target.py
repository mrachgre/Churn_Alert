"""
Step 6: Categorical Variables vs Target (Satisfaction)
======================================================
- Crosstab + % satisfied within each group
- Bar chart saved as PNG
Report only — no encoding decisions.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os


def run_step6(cfg, df: pd.DataFrame):
    sep = "=" * 70
    print(f"\n{sep}\nSTEP 6: CATEGORICAL VARIABLES vs TARGET\n{sep}")

    if cfg.TARGET_COL not in df.columns:
        print(f"  ⚠ Target column '{cfg.TARGET_COL}' not found — skipping"); return

    target_binary = df[cfg.TARGET_COL].map({"satisfied": 1, "dissatisfied": 0})
    overall_rate  = target_binary.mean()
    print(f"\n  Overall satisfaction rate : {overall_rate:.4f} ({overall_rate*100:.2f}%)")

    cat_cols = [c for c in cfg.CAT_COLS if c in df.columns]

    # ── 6.1  Crosstab per categorical column ──────────────────────────────────
    fig_rows = len(cat_cols)
    fig, axes = plt.subplots(fig_rows, 1, figsize=(10, 4 * fig_rows))
    if fig_rows == 1:
        axes = [axes]

    for ax, col in zip(axes, cat_cols):
        print(f"\n[6.x] {col}")
        groups = df.groupby(col)

        rows = []
        for grp_val, grp_df in groups:
            tb = target_binary[grp_df.index]
            n  = len(tb)
            sat_n   = (tb == 1).sum()
            disat_n = (tb == 0).sum()
            sat_pct = sat_n / n * 100
            rows.append({
                "Group":        grp_val,
                "Total":        n,
                "Satisfied":    sat_n,
                "Dissatisfied": disat_n,
                "% Satisfied":  sat_pct,
                "Δ overall":    sat_pct - overall_rate * 100,
            })

        summary = pd.DataFrame(rows).sort_values("% Satisfied", ascending=False)

        # Print
        print(f"  {'Group':<28}  {'Total':>8}  {'Satisfied':>10}  "
              f"{'% Sat':>8}  {'Δ overall':>10}")
        print("  " + "─" * 72)
        for _, r in summary.iterrows():
            diff_str = f"{r['Δ overall']:+.1f}pp"
            flag = "↑" if r["Δ overall"] > 3 else ("↓" if r["Δ overall"] < -3 else "≈")
            print(f"  {str(r['Group']):<28}  {r['Total']:>8,}  {r['Satisfied']:>10,}  "
                  f"{r['% Satisfied']:>7.1f}%  {diff_str:>10}  {flag}")

        # Bar chart
        colors = ["#22c55e" if d > 3 else "#ef4444" if d < -3 else "#3b82f6"
                  for d in summary["Δ overall"]]
        bars = ax.bar(summary["Group"].astype(str), summary["% Satisfied"],
                      color=colors, edgecolor="white", width=0.5)
        ax.axhline(overall_rate * 100, color="white", linestyle="--",
                   linewidth=1.5, label=f"Overall {overall_rate*100:.1f}%")
        ax.set_title(f"% Satisfied by {col}", fontweight="bold")
        ax.set_ylabel("% Satisfied")
        ax.set_ylim(0, 100)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        for bar, (_, r) in zip(bars, summary.iterrows()):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1,
                    f"{r['% Satisfied']:.1f}%",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.suptitle("Step 6: Satisfaction Rate by Categorical Group",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    out_path = os.path.join(cfg.OUT_DIR, "06_categorical_vs_target.png")
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"\n  Chart saved → {out_path}")

    # ── 6.2  Interpretation summary ───────────────────────────────────────────
    print(f"\n[6.2] Quick interpretation:")
    # Use a temp dataframe with integer target to avoid mean() on string column
    _tmp = df[cat_cols].copy()
    _tmp["_target_int"] = target_binary.values
    for col in cat_cols:
        groups = _tmp.groupby(col)["_target_int"].mean()
        max_g  = groups.idxmax()
        min_g  = groups.idxmin()
        print(f"  {col}: highest sat in '{max_g}' ({groups[max_g]*100:.1f}%), "
              f"lowest in '{min_g}' ({groups[min_g]*100:.1f}%)")

    print(f"\n  NOTE: This is exploratory — not formal feature selection.")

    print(f"\n{'─'*70}\nSTEP 6 DONE\n{'─'*70}")


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    import src_airline.airline_config as cfg
    df = pd.read_csv(cfg.CSV_PATH)
    run_step6(cfg, df)
