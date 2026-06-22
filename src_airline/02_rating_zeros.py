"""
Step 2: Rating Zero ("Not Applicable") Analysis
================================================
For each of the 14 service rating columns:
  - Full value_counts (0-5)
  - % of zeros
  - Verify Baggage handling has no 0
  - Case study: compare satisfaction rate at rating=0 vs 1 vs 3
    for Inflight wifi service (evidence for "0 = N/A" hypothesis)
NO transformations — report only.
"""
import pandas as pd
import numpy as np


def run_step2(cfg, df: pd.DataFrame):
    sep = "=" * 70
    print(f"\n{sep}\nSTEP 2: RATING ZERO ('NOT APPLICABLE') ANALYSIS\n{sep}")

    # ── 2.1  Distribution of each service column ──────────────────────────────
    print(f"\n[2.1] Value distribution for each of the {len(cfg.SERVICE_COLS)} service rating columns:")

    zero_pcts = {}
    for col in cfg.SERVICE_COLS:
        if col not in df.columns:
            print(f"  ⚠ Column '{col}' NOT FOUND — skipping"); continue
        vc   = df[col].value_counts().sort_index()
        n0   = vc.get(0, 0)
        pct0 = n0 / len(df) * 100
        zero_pcts[col] = pct0
        vals_str = "  ".join(f"{v}:{c:,}({c/len(df)*100:.1f}%)" for v, c in vc.items())
        print(f"\n  {col}:")
        print(f"    {vals_str}")
        flag = ""
        if col in cfg.NO_ZERO_COLS and n0 > 0:
            flag = f"⚠⚠ README says min=1 but found {n0} zeros!"
        elif col in cfg.NO_ZERO_COLS and n0 == 0:
            flag = "✅ no zeros (matches README min=1)"
        print(f"    → % zeros: {pct0:.2f}%  {flag}")

    # ── 2.2  Ranked zero% across all service columns ──────────────────────────
    print(f"\n[2.2] Columns ranked by % zeros (highest first):")
    print(f"  {'Column':<45}  {'% Zero':>8}")
    print(f"  {'─'*45}  {'─'*8}")
    for col, pct in sorted(zero_pcts.items(), key=lambda x: -x[1]):
        bar = "█" * int(pct / 2)
        print(f"  {col:<45}  {pct:>7.2f}%  {bar}")

    # Summary: columns with notably higher zero rates
    mean_z = np.mean(list(zero_pcts.values()))
    high_z = {k: v for k, v in zero_pcts.items() if v > mean_z + 2}
    if high_z:
        print(f"\n  Columns with zero% > (mean + 2pp = {mean_z+2:.1f}%) — likely 'optional' services:")
        for c, p in high_z.items():
            print(f"    {c}: {p:.2f}%")

    # ── 2.3  Case study: Inflight wifi service ────────────────────────────────
    print(f"\n[2.3] Case study — '{cfg.WIFI_COL}': satisfaction rate at rating=0 vs 1 vs 3")
    print(f"  Hypothesis: if 0 means 'N/A', then satisfaction at rating=0 should")
    print(f"  differ from rating=1 ('very bad') and be closer to the overall average.")
    print()

    wcol = cfg.WIFI_COL
    if wcol in df.columns:
        target_binary = df[cfg.TARGET_COL].map({"satisfied": 1, "dissatisfied": 0})
        overall_sat   = target_binary.mean()
        print(f"  Overall satisfaction rate: {overall_sat:.3f} ({overall_sat*100:.1f}%)")
        print()
        for rating in [0, 1, 2, 3, 4, 5]:
            mask = df[wcol] == rating
            n    = mask.sum()
            if n == 0:
                continue
            sat_rate = target_binary[mask].mean()
            bar      = "█" * int(sat_rate * 30)
            diff     = sat_rate - overall_sat
            print(f"  rating={rating}  n={n:>6,}  sat_rate={sat_rate:.3f}  "
                  f"(Δ overall={diff:+.3f})  {bar}")
        print()
        # Key conclusion
        r0 = target_binary[df[wcol] == 0].mean() if (df[wcol] == 0).any() else None
        r1 = target_binary[df[wcol] == 1].mean() if (df[wcol] == 1).any() else None
        r3 = target_binary[df[wcol] == 3].mean() if (df[wcol] == 3).any() else None
        if r0 is not None and r1 is not None and r3 is not None:
            diff_01 = abs(r0 - r1)
            diff_03 = abs(r0 - r3)
            print(f"  |sat(0) - sat(1)| = {diff_01:.3f}")
            print(f"  |sat(0) - sat(3)| = {diff_03:.3f}")
            if diff_03 < diff_01:
                print(f"  → ✅ SUPPORTS 'N/A' hypothesis: rating=0 is closer to rating=3 than to rating=1")
            else:
                print(f"  → ⚠ CONTRADICTS 'N/A' hypothesis: rating=0 is closer to rating=1 (very bad)")

    print(f"\n  NOTE: No changes made. Review the zero% distribution above and decide")
    print(f"  whether to: (a) treat 0 as-is, (b) NaN it, or (c) create an indicator column.")

    print(f"\n{'─'*70}\nSTEP 2 DONE\n{'─'*70}")


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    import src_airline.airline_config as cfg
    df = pd.read_csv(cfg.CSV_PATH)
    run_step2(cfg, df)
