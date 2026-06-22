"""
run_airline_shap.py  —  SHAP Analysis + Personalized Action Matrix
===================================================================
Usage:
    python run_airline_shap.py           # Run Steps A, B, C then PAUSE
    python run_airline_shap.py --step D  # Generate action dict template
    python run_airline_shap.py --step E  # Build v2 list (after filling dict)
"""
import sys, os, importlib.util, numpy as np, pandas as pd, joblib

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


shap_mod = load("p2_shap_analysis.py")
D = cfg.DATA_INT_DIR
M = cfg.MODEL_DIR


def load_data():
    print("  Loading test data & models...")
    X_test_base  = pd.read_parquet(os.path.join(D, "X_test_base.parquet"))
    X_test_clust = pd.read_parquet(os.path.join(D, "X_test_clust.parquet"))
    y_test       = np.load(os.path.join(D, "y_test.npy"))
    cal_model    = joblib.load(os.path.join(M, "xgb_base_model.pkl"))
    action_df    = pd.read_csv(os.path.join(cfg.DATA_DIR, "targeted_action_list.csv"))
    cluster_series = X_test_clust["kmeans_cluster_id"].reset_index(drop=True)
    print(f"  X_test_base : {X_test_base.shape}")
    print(f"  action_df   : {action_df.shape}")
    return X_test_base, X_test_clust, y_test, cal_model, action_df, cluster_series


def main():
    import time
    t0 = time.time()

    step_arg = None
    for i, arg in enumerate(sys.argv[1:]):
        if arg in ("--step", "-s") and i + 1 < len(sys.argv) - 1:
            step_arg = sys.argv[i + 2].upper()
        elif arg.upper() in ("D", "E"):
            step_arg = arg.upper()

    print(f"\n{'='*70}")
    print(f"  ✈  AIRLINE — SHAP Analysis + Personalized Action Matrix")
    print(f"  Step arg: {step_arg or 'A+B+C (default)'}")
    print(f"{'='*70}")

    if step_arg == "D":
        # Step D only: generate template
        _, _, _, _, _, _ = load_data()
        shap_global = pd.read_csv(os.path.join(cfg.DATA_DIR, "shap_global_importance.csv"))
        freq_df     = pd.read_csv(os.path.join(cfg.DATA_DIR, "critical_driver_frequency.csv"))
        shap_mod.run_step_d(cfg, shap_global, freq_df)
        return

    if step_arg == "E":
        # Step E only: build v2 list (needs action_dict filled)
        action_df = pd.read_csv(os.path.join(cfg.DATA_DIR, "targeted_action_list.csv"))
        # Re-load top_driver cols if saved from Step C
        v2_base_path = os.path.join(cfg.DATA_DIR, "targeted_action_list_with_drivers.csv")
        if os.path.exists(v2_base_path):
            action_df = pd.read_csv(v2_base_path)
        dict_path = os.path.join(cfg.DATA_DIR, "action_dictionary_template.csv")
        shap_mod.run_step_e(cfg, action_df, dict_path)
        return

    # Default: Steps A → B → C → pause
    X_test_base, X_test_clust, y_test, cal_model, action_df, cluster_series = load_data()

    # Step A
    print(f"\n{'#'*70}\n#  [A] SHAP Global Importance\n{'#'*70}")
    explainer, shap_values, shap_rank_df = shap_mod.run_step_a(
        cfg, X_test_base, cal_model
    )

    # Step B
    print(f"\n{'#'*70}\n#  [B] SHAP by Cluster\n{'#'*70}")
    shap_mod.run_step_b(cfg, X_test_base, shap_values, cluster_series)

    # Step C
    print(f"\n{'#'*70}\n#  [C] SHAP — Critical Risk Group\n{'#'*70}")
    action_df_with_drivers, freq_df = shap_mod.run_step_c(
        cfg, X_test_base, shap_values, action_df
    )

    # Save enriched action_df (with top_driver cols) for Step E later
    action_df_with_drivers.to_csv(
        os.path.join(cfg.DATA_DIR, "targeted_action_list_with_drivers.csv"),
        index=False, encoding="utf-8-sig"
    )
    print(f"\n  targeted_action_list_with_drivers.csv saved (for Step E)")

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"  ⏱  Steps A+B+C done in {elapsed:.1f}s")
    print(f"\n  ⏸  PIPELINE PAUSED after Step C")
    print(f"     → Review SHAP drivers above")
    print(f"     → Then run: python run_airline_shap.py --step D")
    print(f"       (generates action_dictionary_template.csv to fill in)")
    print(f"     → Fill the CSV, then run: python run_airline_shap.py --step E")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
