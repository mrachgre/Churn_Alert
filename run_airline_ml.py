"""
run_airline_ml.py  —  Airline Satisfaction Phase 2: ML Pipeline
================================================================
Steps 0 → 3 → 4 → 5 → 5b → 6 → 7 → 8
Completely separate from run_airline.py (Phase 1 EDA).

Usage:
    python run_airline_ml.py
"""
import sys
import os
import time
import importlib.util

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR  = os.path.join(BASE_DIR, "src_airline")
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, SRC_DIR)

import airline_config as cfg


def load(filename):
    path = os.path.join(SRC_DIR, filename)
    spec = importlib.util.spec_from_file_location("_step", path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def banner(step, title, total=9):
    w = 70
    print(f"\n{'#'*w}\n#  [{step}/{total}] {title}\n{'#'*w}")


def main():
    t0 = time.time()

    print(f"\n{'='*70}")
    print(f"  ✈  AIRLINE Satisfaction — Phase 2: ML Pipeline")
    print(f"  Dataset : {cfg.CSV_FILENAME}")
    print(f"  Models  : {cfg.MODEL_DIR}")
    print(f"  Output  : {cfg.OUT_DIR}")
    print(f"{'='*70}")

    if not os.path.exists(cfg.CSV_PATH):
        print(f"\n  ERROR: {cfg.CSV_PATH} not found!"); sys.exit(1)

    # Step 0: Prepare
    banner(0, "Data Preparation (Phase 2 decisions)")
    train_df, test_df = load("p2_00_prepare.py").run_prepare(cfg)

    # Step 3: EDA Correlation (on training set only)
    banner(3, "EDA Correlation — Cramér's V, Point-Biserial, Heatmap")
    # Step 4 must run first to produce composite features for Step 3 chart
    # Run FE briefly first, then EDA
    train_fe, test_fe = load("p2_04_feature_engineering.py").run_step4(cfg, train_df, test_df)

    banner(3, "EDA Correlation — Cramér's V, Point-Biserial, Heatmap (on FE data)")
    load("p2_03_eda_correlation.py").run_step3(cfg, train_fe)

    # Step 5: Tree preprocessing + KMeans
    banner(5, "Preprocessing — Tree Path + KMeans FE")
    step5 = load("p2_05_preprocessing.py").run_step5(cfg, train_fe, test_fe)

    # Step 5b: Linear preprocessing
    banner("5b", "Preprocessing — Linear Path (clip + impute + MinMax + Chi²)")
    step5b = load("p2_05b_linear_preprocessing.py").run_step5b(cfg, step5)

    # Step 6: Modelling
    banner(6, "Modelling — 6 Models (Optuna XGB + GridSearch others)")
    step6 = load("p2_06_modelling.py").run_step6(cfg, step5, step5b)

    print(f"\n{'!'*70}")
    print(f"  ⏸  PAUSED after Step 6 — Review model_comparison above.")
    print(f"  Press ENTER to continue to Step 7 (Cluster Profiling)...")
    print(f"{'!'*70}")
    input()

    # Step 7: Cluster profiling
    banner(7, "Cluster Profiling — Passenger Personas")
    cluster_df = load("p2_07_cluster_profiling.py").run_step7(cfg, step5, step6)

    print(f"\n{'!'*70}")
    print(f"  ⏸  PAUSED after Step 7 — Review cluster personas above.")
    print(f"  Press ENTER to continue to Step 8 (Risk Scoring)...")
    print(f"{'!'*70}")
    input()

    # Step 8: Risk scoring + action matrix
    banner(8, "Risk Scoring + Action Matrix")
    action_df = load("p2_08_risk_scoring.py").run_step8(cfg, step5, step6, cluster_df)

    # Final summary
    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"  ✅  PIPELINE COMPLETE")
    print(f"  ⏱  Total time : {elapsed/60:.1f} min")
    print(f"\n  Artifacts:")
    print(f"    models/AIRLINE/       — {len(os.listdir(cfg.MODEL_DIR))} files")
    print(f"    outputs/AIRLINE/      — {len([f for f in os.listdir(cfg.OUT_DIR) if f.endswith('.png')])} charts")
    print(f"    data/AIRLINE/*.csv    — model_comparison, cluster_summary, targeted_action_list")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
