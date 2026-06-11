"""
Step 5: Preprocessing Pipeline
- Feature engineering (device_per_tenure, cashback_per_order, inactivity_ratio)
- Stratified train-test split (80/20)
- OneHotEncoder (fit on train only)
- MinMaxScaler + KNNImputer (fit on train only)
- SMOTE to balance classes
- Save all fitted transformers as .pkl
"""

import pandas as pd
import numpy as np
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.impute import KNNImputer
from imblearn.over_sampling import SMOTE

RFM_PATH   = os.path.join(os.path.dirname(__file__), "..", "data", "ecommerce_churn_rfm.csv")
MODEL_DIR  = os.path.join(os.path.dirname(__file__), "..", "models")
DATA_DIR   = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(MODEL_DIR, exist_ok=True)


def load(path=RFM_PATH):
    df = pd.read_csv(path, dtype={"CityTier": str, "Complain": str})
    df["Churn"] = df["Churn"].astype(int)
    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Create 3 engineered features ."""
    df = df.copy()
    df["device_per_tenure"]  = df["NumberOfDeviceRegistered"] / (df["Tenure"].fillna(0) + 1)
    df["cashback_per_order"] = df["CashbackAmount"] / (df["OrderCount"].fillna(0) + 1)
    df["inactivity_ratio"]   = df["DaySinceLastOrder"] / (df["Tenure"].fillna(0) + 1)
    print("[05] Engineered features: device_per_tenure, cashback_per_order, inactivity_ratio")
    return df


def prepare_features(df: pd.DataFrame):
    """Drop ID-like and label cols; separate X and y."""
    drop_cols = ["CustomerID", "Churn"]
    # Keep rfm_total, rfm_segment; drop R_score/F_score/M_score (redundant)
    drop_cols += [c for c in ["R_score", "F_score", "M_score"] if c in df.columns]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df["Churn"]
    return X, y


def run_preprocessing(path=RFM_PATH, random_state=42):
    df = load(path)
    print(f"\n{'='*60}")
    print("STEP 5: PREPROCESSING")
    print(f"{'='*60}")

    # ── Feature engineering ──────────────────────────────────────────────────
    df = feature_engineering(df)

    # ── Prepare X, y ─────────────────────────────────────────────────────────
    X, y = prepare_features(df)
    print(f"[05] Features: {X.shape[1]} | Samples: {X.shape[0]}")

    # ── Train-Test split (stratified 80/20) ──────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_state)
    print(f"[05] Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")
    print(f"[05] Before SMOTE → Class 0: {(y_train==0).sum()} | Class 1: {(y_train==1).sum()}")

    # ── Identify column groups ────────────────────────────────────────────────
    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    print(f"[05] Categorical columns: {cat_cols}")
    print(f"[05] Numeric columns   : {num_cols}")

    # ── OneHotEncoder (fit on TRAIN only) ────────────────────────────────────
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop="first")
    encoder.fit(X_train[cat_cols])

    ohe_feature_names = encoder.get_feature_names_out(cat_cols).tolist()

    def encode(X, enc, cat_cols, num_cols):
        cat_enc = pd.DataFrame(enc.transform(X[cat_cols]),
                               columns=ohe_feature_names, index=X.index)
        num_part = X[num_cols].reset_index(drop=True)
        cat_part = cat_enc.reset_index(drop=True)
        return pd.concat([num_part, cat_part], axis=1)

    X_train_enc = encode(X_train, encoder, cat_cols, num_cols)
    X_test_enc  = encode(X_test,  encoder, cat_cols, num_cols)

    # ── MinMaxScaler + KNNImputer (fit on TRAIN only) ────────────────────────
    num_cols_imp = X_train_enc.select_dtypes(include=[np.number]).columns.tolist()

    scaler  = MinMaxScaler()
    X_train_enc[num_cols_imp] = scaler.fit_transform(X_train_enc[num_cols_imp])
    X_test_enc[num_cols_imp]  = scaler.transform(X_test_enc[num_cols_imp])

    imputer = KNNImputer(n_neighbors=5)
    X_train_enc[num_cols_imp] = imputer.fit_transform(X_train_enc[num_cols_imp])
    X_test_enc[num_cols_imp]  = imputer.transform(X_test_enc[num_cols_imp])

    print(f"[05] Scaling + KNN imputation done. Missing remaining: "
          f"{X_train_enc.isnull().sum().sum()}")

    # ── SMOTE (only on training set) ─────────────────────────────────────────
    smote = SMOTE(random_state=random_state, k_neighbors=5)
    X_train_res, y_train_res = smote.fit_resample(X_train_enc, y_train)
    print(f"[05] After SMOTE  → Class 0: {(y_train_res==0).sum()} "
          f"| Class 1: {(y_train_res==1).sum()}")

    # ── Save artifacts ────────────────────────────────────────────────────────
    joblib.dump(encoder,  os.path.join(MODEL_DIR, "encoder.pkl"))
    joblib.dump(scaler,   os.path.join(MODEL_DIR, "scaler.pkl"))
    joblib.dump(imputer,  os.path.join(MODEL_DIR, "imputer.pkl"))

    # Save processed splits
    X_train_res_df = pd.DataFrame(X_train_res, columns=X_train_enc.columns)
    X_test_enc_df  = pd.DataFrame(X_test_enc,  columns=X_train_enc.columns)

    X_train_res_df.to_csv(os.path.join(DATA_DIR, "X_train.csv"), index=False)
    X_test_enc_df.to_csv( os.path.join(DATA_DIR, "X_test.csv"),  index=False)
    y_train_res.to_csv(   os.path.join(DATA_DIR, "y_train.csv"), index=False)
    y_test.to_csv(        os.path.join(DATA_DIR, "y_test.csv"),  index=False)

    # Also save raw (un-encoded) test set for clustering
    X_test.to_csv(os.path.join(DATA_DIR, "X_test_raw.csv"), index=False)
    y_test.to_csv(os.path.join(DATA_DIR, "y_test_raw.csv"), index=False)
    
    print(f"[05] Saved: encoder, scaler, imputer | X_train, X_test, y_train, y_test")
    print(f"[05] Feature count after encoding: {X_train_res_df.shape[1]}")

    return (X_train_res_df, X_test_enc_df,
            y_train_res.reset_index(drop=True),
            y_test.reset_index(drop=True),
            {"encoder": encoder, "scaler": scaler, "imputer": imputer,
             "cat_cols": cat_cols, "num_cols": num_cols,
             "ohe_feature_names": ohe_feature_names})


if __name__ == "__main__":
    run_preprocessing()
