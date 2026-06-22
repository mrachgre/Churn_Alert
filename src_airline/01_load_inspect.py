"""
Step 1: Load & General Inspection
==================================
- Shape, dtypes, first 5 rows
- Null % per column (verify only Arrival Delay has nulls)
- Duplicate row count (report only, no drop)
- value_counts for target + 3 categorical cols (verify vs README)
"""
import pandas as pd


def run_step1(cfg) -> pd.DataFrame:
    sep = "=" * 70
    print(f"\n{sep}\nSTEP 1: LOAD & GENERAL INSPECTION\n{sep}")

    # 1.1 Load
    print(f"\n[1.1] Loading: {cfg.CSV_PATH}")
    df = pd.read_csv(cfg.CSV_PATH)
    row_ok = df.shape[0] == cfg.REF["total_rows"]
    ref_rows = cfg.REF["total_rows"]
    row_msg = "✅ matches README" if row_ok else f"⚠ README says {ref_rows:,}"
    print(f"  Shape : {df.shape[0]:,} rows × {df.shape[1]} cols  {row_msg}")

    # 1.2 dtypes
    print(f"\n[1.2] Column dtypes:")
    for col in df.columns:
        print(f"  {col:<45}  {str(df[col].dtype):<10}  nunique={df[col].nunique()}")

    # 1.3 First 5 rows
    print(f"\n[1.3] First 5 rows:")
    with pd.option_context("display.max_columns", None, "display.width", 220):
        print(df.head().to_string(index=False))

    # 1.4 Null %
    print(f"\n[1.4] Null count & % per column (non-zero only):")
    null_s = df.isnull().sum()
    unexpected = False
    printed_any = False
    for col in df.columns:
        n = null_s[col]
        if n == 0:
            continue
        pct = n / len(df) * 100
        printed_any = True
        if col == "Arrival Delay in Minutes":
            match = ("✅ EXPECTED" if n == cfg.REF["null_arrival_delay"]
                     else f"⚠ expected {cfg.REF['null_arrival_delay']}, got {n}")
        else:
            match = "⚠⚠ UNEXPECTED — not in README"
            unexpected = True
        print(f"  {col:<45}  {n:>6} ({pct:.3f}%)  {match}")
    if not printed_any:
        print("  (no nulls found)")
    if not unexpected:
        print("  ✅ Only 'Arrival Delay in Minutes' has nulls — matches README.")

    # 1.5 Duplicates
    print(f"\n[1.5] Fully-duplicate rows (all 22 cols identical):")
    n_dup = df.duplicated().sum()
    print(f"  Count : {n_dup:,}  ({n_dup/len(df)*100:.3f}%)")
    if n_dup > 0:
        print("  ⚠ NOT auto-dropped — could be different passengers with same answers.")
        print(f"  Sample:")
        with pd.option_context("display.max_columns", None, "display.width", 220):
            print(df[df.duplicated(keep=False)].head(4).to_string(index=False))
    else:
        print("  ✅ No duplicate rows.")

    # 1.6 value_counts vs README
    checks = [
        (cfg.TARGET_COL,
         {"satisfied": cfg.REF["satisfied"], "dissatisfied": cfg.REF["dissatisfied"]}),
        ("Customer Type",
         {"Loyal Customer": cfg.REF["loyal"], "disloyal Customer": cfg.REF["disloyal"]}),
        ("Type of Travel",
         {"Business travel": cfg.REF["business_travel"],
          "Personal Travel": cfg.REF["personal_travel"]}),
        ("Class",
         {"Business": cfg.REF["class_business"],
          "Eco": cfg.REF["class_eco"],
          "Eco Plus": cfg.REF["class_eco_plus"]}),
    ]
    print(f"\n[1.6] value_counts verification against README:")
    all_ok = True
    for col, exp in checks:
        if col not in df.columns:
            print(f"\n  ⚠ Column '{col}' not found!"); continue
        print(f"\n  ── {col} ──")
        for val, cnt in df[col].value_counts().items():
            pct = cnt / len(df) * 100
            e = exp.get(val)
            ok = (cnt == e) if e is not None else None
            flag = ("✅" if ok else f"⚠ expected {e:,}" if e else "⚠ not in README")
            if ok is False or ok is None:
                all_ok = False
            print(f"    {str(val):<28} {cnt:>8,} ({pct:.1f}%)  {flag}")

    print(f"\n  README verification: {'✅ ALL MATCH' if all_ok else '⚠ MISMATCHES — see above'}")

    print(f"\n{'─'*70}\nSTEP 1 DONE\n{'─'*70}")
    return df


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    import src_airline.airline_config as cfg
    run_step1(cfg)
