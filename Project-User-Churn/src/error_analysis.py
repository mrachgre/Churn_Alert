"""
Advanced false-positive / false-negative analysis using saved predictions.
"""

import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

FEATURES = [
    "Tenure", "SatisfactionScore", "OrderCount", "CashbackAmount",
    "NumberOfDeviceRegistered",
]


def _read_csv(name):
    path = os.path.join(DATA_DIR, name)
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def run_error_analysis():
    df = _read_csv("targeted_action_list.csv")
    if df is None:
        df = _read_csv("predictions.csv")
    raw = _read_csv("X_test_raw.csv")
    if df is None:
        return _write_unavailable("Missing predictions.csv or targeted_action_list.csv.")
    if raw is not None:
        for col in raw.columns:
            if col not in df.columns:
                df[col] = raw[col].values[:len(df)]

    if "y_true" not in df.columns:
        y_df = _read_csv("y_test.csv")
        if y_df is not None:
            df["y_true"] = y_df.squeeze().values[:len(df)]
    if "churn_prediction" not in df.columns:
        if "churn_probability" in df.columns:
            df["churn_prediction"] = (df["churn_probability"] >= 0.5).astype(int)
        else:
            return _write_unavailable("Prediction labels or probabilities are unavailable.")
    if "y_true" not in df.columns:
        return _write_unavailable("Ground-truth labels are unavailable.")

    df["error_type"] = np.select(
        [
            (df["y_true"] == 0) & (df["churn_prediction"] == 1),
            (df["y_true"] == 1) & (df["churn_prediction"] == 0),
            (df["y_true"] == df["churn_prediction"]),
        ],
        ["False Positive", "False Negative", "Correct"],
        default="Unknown",
    )

    rows = []
    for error_type in ["Correct", "False Positive", "False Negative"]:
        subset = df[df["error_type"] == error_type]
        row = {"Segment": "Overall", "Error Type": error_type, "Count": int(len(subset))}
        for col in FEATURES:
            row[f"{col} Mean"] = round(float(pd.to_numeric(subset.get(col), errors="coerce").mean()), 4) if col in subset else np.nan
        rows.append(row)

    for segment_col in ["cluster", "risk_category", "persona", "MaritalStatus", "Gender"]:
        if segment_col not in df.columns:
            continue
        for segment, group in df.groupby(segment_col, dropna=False):
            for error_type in ["False Positive", "False Negative"]:
                subset = group[group["error_type"] == error_type]
                if subset.empty:
                    continue
                row = {"Segment": f"{segment_col}={segment}", "Error Type": error_type, "Count": int(len(subset))}
                for col in FEATURES:
                    row[f"{col} Mean"] = round(float(pd.to_numeric(subset.get(col), errors="coerce").mean()), 4) if col in subset else np.nan
                rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT_DIR, "error_analysis.csv"), index=False)
    _write_summary(df, out)
    print("[Research] Saved: error_analysis.csv, error_analysis_summary.md")
    return out


def _write_unavailable(reason):
    pd.DataFrame(columns=["Segment", "Error Type", "Count"]).to_csv(
        os.path.join(OUT_DIR, "error_analysis.csv"), index=False
    )
    with open(os.path.join(OUT_DIR, "error_analysis_summary.md"), "w", encoding="utf-8") as f:
        f.write("# Error Analysis Summary\n\n")
        f.write(reason + "\n")
    return pd.DataFrame()


def _write_summary(df, out):
    fp = int((df["error_type"] == "False Positive").sum())
    fn = int((df["error_type"] == "False Negative").sum())
    segment_errors = out[out["Segment"] != "Overall"].sort_values("Count", ascending=False)
    top_segment = segment_errors.iloc[0]["Segment"] if not segment_errors.empty else "Unavailable"
    fn_profile = out[(out["Segment"] == "Overall") & (out["Error Type"] == "False Negative")]

    lines = [
        "# Error Analysis Summary",
        "",
        f"- False positives: {fp}.",
        f"- False negatives: {fn}.",
        f"- Segment contributing most observed errors: {top_segment}.",
        "",
        "## Hardest Customers To Classify",
    ]
    if not fn_profile.empty:
        row = fn_profile.iloc[0]
        lines.append(
            "False negatives are churned customers predicted as retained. Their average profile is "
            + ", ".join([f"{c}={row.get(c + ' Mean', np.nan)}" for c in FEATURES])
            + "."
        )
    else:
        lines.append("No false negatives were observed in the available prediction file.")

    lines.extend([
        "",
        "## Business Risk",
        "False negatives are the highest-risk error type because the business does not intervene for customers who actually churn. This can lead to lost lifetime value, missed service recovery, and underestimation of dissatisfaction in operational reporting.",
    ])
    with open(os.path.join(OUT_DIR, "error_analysis_summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    run_error_analysis()
