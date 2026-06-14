"""
Churn Alert — Root Pipeline Runner
Run this from the Project-User-Churn/ directory:
    python run_pipeline.py
"""

import sys, os, time, importlib.util
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE_DIR, "data")
RAW_XLSX  = os.path.join(DATA_DIR, "E Commerce Dataset.xlsx")
RAW_CSV   = os.path.join(DATA_DIR, "ecommerce_churn.csv")
RAW_PATH  = RAW_XLSX if os.path.exists(RAW_XLSX) else RAW_CSV


def load_src(filename: str):
    """Load a src/ module by filename, bypassing numeric-prefix import restriction."""
    path = os.path.join(BASE_DIR, "src", filename)
    spec = importlib.util.spec_from_file_location("module", path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def banner(title: str):
    print(f"\n{'='*70}")
    print(f"#  {title}")
    print(f"{'='*70}")


def check_dataset():
    if not os.path.exists(RAW_XLSX) and not os.path.exists(RAW_CSV):
        print(f"\n{'!'*70}")
        print("  ERROR: Dataset not found!")
        print(f"  Expected: {RAW_XLSX}")
        print("  OR:       " + RAW_CSV)
        print("\n  Download via: python data/download.py")
        print(f"{'!'*70}\n")
        sys.exit(1)
    found = RAW_XLSX if os.path.exists(RAW_XLSX) else RAW_CSV
    print(f"[✓] Dataset found: {os.path.basename(found)}")


def main():
    total_start = time.time()
    check_dataset()

    # ── Step 1 ───────────────────────────────────────────────────────────────
    banner("Step 1: Data Cleaning")
    load_src("01_data_cleaning.py").clean(RAW_PATH)

    # ── Step 2 ───────────────────────────────────────────────────────────────
    banner("Step 2: Missing Data Analysis")
    load_src("02_missing_analysis.py").analyze_missing()

    # ── Step 3 ───────────────────────────────────────────────────────────────
    banner("Step 3: EDA")
    load_src("03_eda.py").run_eda()

    # ── Step 4 ───────────────────────────────────────────────────────────────
    banner("Step 4: RFM Segmentation")
    load_src("04_rfm_segmentation.py").run_rfm()

    # ── Step 5 ───────────────────────────────────────────────────────────────
    banner("Step 5: Preprocessing")
    load_src("05_preprocessing.py").run_preprocessing()

    # ── Step 6 ────────────────────────────────────────────────────────────────────────
    banner("Step 6: Modelling (XGBoost + Optuna + SHAP + Multi-Model Comparison)")
    load_src("06_modelling.py").run_modelling()

    # ── Step 6b ─────────────────────────────────────────────────────────────────────
    banner("Step 6b: Semi-supervised LR (Strategy C — K-Means Label Propagation)")
    load_src("06b_semisupervised.py").run_semisupervised()

    # ── Step 7 ────────────────────────────────────────────────────────────────────────
    banner("Step 7: Cluster Profiling (Business Reporting — model from Step 5)")
    load_src("07_clustering.py").run_clustering()


    # ── Step 8 ───────────────────────────────────────────────────────────────
    banner("Step 8: Survival Analysis")
    load_src("08_survival_analysis.py").run_survival_analysis()

    elapsed = (time.time() - total_start) / 60
    print(f"\n{'='*70}")
    print(f"  ✅  FULL PIPELINE COMPLETE  ({elapsed:.1f} minutes)")
    print(f"  Outputs  → outputs/")
    print(f"  Models   → models/")
    print(f"  Data     → data/")
    print(f"\n  ▶  Launch dashboard:")
    print(f"     streamlit run app/streamlit_app.py --server.port 8502 --browser.gatherUsageStats false")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()