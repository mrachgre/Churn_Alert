"""
Step 1: Data Cleaning
- Load raw CSV
- Fix data types (CityTier, Complain → categorical)
- Merge typo variants in PreferredLoginDevice and PreferredPaymentMode
"""

import pandas as pd
import numpy as np
import os

from pipeline_config import DATA_DIR   # noqa: E402

# Support both .xlsx (kagglehub download) and .csv
RAW_XLSX   = os.path.join(DATA_DIR, "E Commerce Dataset.xlsx")
RAW_CSV    = os.path.join(DATA_DIR, "ecommerce_churn.csv")
RAW_PATH   = RAW_XLSX if os.path.exists(RAW_XLSX) else RAW_CSV
CLEAN_PATH = os.path.join(DATA_DIR, "ecommerce_churn_clean.csv")


def load_data(path: str = RAW_PATH) -> pd.DataFrame:
    """Load raw file — supports .xlsx (sheet 'E Comm') or .csv."""
    if str(path).endswith(".xlsx"):
        df = pd.read_excel(path, sheet_name="E Comm")
        print(f"[01] Loaded Excel sheet 'E Comm': {df.shape[0]:,} rows × {df.shape[1]} columns")
    else:
        df = pd.read_csv(path)
        print(f"[01] Loaded CSV: {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


def fix_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Convert CityTier and Complain to object (categorical)."""
    df = df.copy()
    df["CityTier"] = df["CityTier"].astype("object")
    df["Complain"] = df["Complain"].astype("object")
    print(f"[01] CityTier → object | Complain → object")
    return df


def fix_typos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge duplicate / misspelled category values:
      PreferredLoginDevice : 'Mobile Phone', 'Phone'  → 'Mobile'
      PreferredPaymentMode : 'CC' → 'Credit Card'
                            'COD' → 'Cash on Delivery'
    """
    df = df.copy()

    # --- PreferredLoginDevice ---
    device_map = {
        "Mobile Phone": "Mobile",
        "Phone": "Mobile",
    }
    df["PreferredLoginDevice"] = df["PreferredLoginDevice"].replace(device_map)

    # --- PreferredPaymentMode ---
    payment_map = {
        "CC": "Credit Card",
        "COD": "Cash on Delivery",
    }
    df["PreferredPaymentMode"] = df["PreferredPaymentMode"].replace(payment_map)

    print("[01] Typos fixed:")
    print(f"       PreferredLoginDevice values  : {df['PreferredLoginDevice'].unique().tolist()}")
    print(f"       PreferredPaymentMode values  : {df['PreferredPaymentMode'].unique().tolist()}")
    return df


def clean(path: str = RAW_PATH, save: bool = True) -> pd.DataFrame:
    df = load_data(path)
    df = fix_dtypes(df)
    df = fix_typos(df)
    if save:
        df.to_csv(CLEAN_PATH, index=False)
        print(f"[01] Saved cleaned data → {CLEAN_PATH}")
    return df


if __name__ == "__main__":
    clean()
