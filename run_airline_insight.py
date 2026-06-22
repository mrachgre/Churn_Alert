"""
run_airline_insight.py  —  Steps D-new & E-new: Descriptive Insight Reports
============================================================================
Usage:
    python run_airline_insight.py
"""
import sys, os, importlib.util, numpy as np, pandas as pd, joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR  = os.path.join(BASE_DIR, "src_airline")
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, SRC_DIR)
import airline_config as cfg


def load_mod(filename):
    path = os.path.join(SRC_DIR, filename)
    spec = importlib.util.spec_from_file_location("_step", path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    import time
    t0 = time.time()

    print(f"\n{'='*70}")
    print(f"  ✈  AIRLINE — Descriptive Risk Insight Reports")
    print(f"  Output: data/AIRLINE/")
    print(f"{'='*70}")

    D = cfg.DATA_INT_DIR
    M = cfg.MODEL_DIR

    # Load all inputs
    print("\n  Loading inputs...")
    X_test_base  = pd.read_parquet(os.path.join(D, "X_test_base.parquet"))
    X_test_clust = pd.read_parquet(os.path.join(D, "X_test_clust.parquet"))
    cal_model    = joblib.load(os.path.join(M, "xgb_base_model.pkl"))
    action_df    = pd.read_csv(os.path.join(cfg.DATA_DIR, "targeted_action_list.csv"))

    print(f"  X_test_base : {X_test_base.shape}")
    print(f"  action_df   : {action_df.shape}")

    mod = load_mod("p2_insight_report.py")

    # Step D-new
    print(f"\n{'#'*70}\n#  STEP D-NEW: Customer Risk Insight CSV\n{'#'*70}")
    report_df, shap_values = mod.run_step_d_new(
        cfg, X_test_base, X_test_clust, action_df, cal_model
    )

    # Step E-new
    print(f"\n{'#'*70}\n#  STEP E-NEW: System-Level Insight Report (Markdown)\n{'#'*70}")
    mod.run_step_e_new(cfg, report_df, X_test_base, action_df, shap_values)

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"  ✅  DONE in {elapsed:.1f}s")
    print(f"\n  Outputs:")
    print(f"    data/AIRLINE/customer_risk_insight_report.csv")
    print(f"    data/AIRLINE/system_level_insight_report.md")
    print(f"    data/AIRLINE/shap_global_importance.csv")
    print(f"    data/AIRLINE/critical_driver_frequency.csv")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
