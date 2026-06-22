"""
Airline Customer Satisfaction — EDA Pipeline Runner
====================================================
Runs Steps 1-7 sequentially. Stops after Step 7 for review.
Data is loaded ONCE and passed through all steps.

Usage:
    python run_airline.py
"""
import sys
import os
import time
import importlib.util

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR  = os.path.join(BASE_DIR, "src_airline")
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, SRC_DIR)

import airline_config as cfg


def load_step(filename: str):
    """Load a src_airline/ module by filename (handles numeric prefixes)."""
    path = os.path.join(SRC_DIR, filename)
    spec = importlib.util.spec_from_file_location("step_module", path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def banner(title: str, step: int, total: int = 7):
    w = 70
    print(f"\n{'#'*w}")
    print(f"#  [{step}/{total}] {title}")
    print(f"{'#'*w}")


def main():
    total_start = time.time()

    # ── Verify data file exists ───────────────────────────────────────────────
    if not os.path.exists(cfg.CSV_PATH):
        print(f"\n{'!'*70}")
        print(f"  ERROR: Data file not found!")
        print(f"  Expected:\n    {cfg.CSV_PATH}")
        print(f"{'!'*70}\n")
        sys.exit(1)

    print(f"\n{'='*70}")
    print(f"  ✈  AIRLINE Customer Satisfaction — EDA Pipeline")
    print(f"  Dataset : {cfg.CSV_FILENAME}")
    print(f"  Output  : {cfg.OUT_DIR}")
    print(f"{'='*70}")

    # ── Step 1: Load & inspect ────────────────────────────────────────────────
    banner("Load & General Inspection", 1)
    df = load_step("01_load_inspect.py").run_step1(cfg)

    # ── Step 2: Rating zeros ──────────────────────────────────────────────────
    banner("Rating Zero ('Not Applicable') Analysis", 2)
    load_step("02_rating_zeros.py").run_step2(cfg, df)

    # ── Step 3: Delay correlation ─────────────────────────────────────────────
    banner("Departure vs Arrival Delay Correlation", 3)
    load_step("03_delay_correlation.py").run_step3(cfg, df)

    # ── Step 4: Null pattern ──────────────────────────────────────────────────
    banner("Null Pattern Analysis — Arrival Delay (393 rows)", 4)
    load_step("04_null_pattern.py").run_step4(cfg, df)

    # ── Step 5: Outliers ──────────────────────────────────────────────────────
    banner("Outlier Analysis — Flight Distance & Delay", 5)
    load_step("05_outliers.py").run_step5(cfg, df)

    # ── Step 6: Categorical vs target ─────────────────────────────────────────
    banner("Categorical Variables vs Target (Satisfaction)", 6)
    load_step("06_categorical_vs_target.py").run_step6(cfg, df)

    # ── Step 7: Summary report ────────────────────────────────────────────────
    banner("Summary Report & Decisions Checklist", 7)
    load_step("07_summary_report.py").run_step7(cfg, df)

    # ── Done ──────────────────────────────────────────────────────────────────
    elapsed = time.time() - total_start
    print(f"  ⏱  Total runtime : {elapsed:.1f}s")
    print(f"  📊  Charts saved  : {cfg.OUT_DIR}")
    print(f"\n  ⏸  PIPELINE PAUSED — Review Step 7 decisions before Part 2.\n")


if __name__ == "__main__":
    main()
