"""
Step 2: Missing Data Analysis  (optimized)
==========================================
Context: Part 1 (01_data_cleaning.py) đã sửa CẤU TRÚC và lưu ecommerce_churn_clean.csv.
Bước 2 là bước PHÂN TÍCH ĐẦU TIÊN → theo nguyên tắc, CHIA TRAIN/TEST NGAY ĐẦU BƯỚC,
rồi mọi chẩn đoán chỉ nhìn TRAIN (test bị niêm phong → không leakage).

Việc làm (report-only, KHÔNG impute / KHÔNG drop):
  0. Split 80/20 stratified — LẦN CHIA DUY NHẤT, lưu ra đĩa (idempotent).
  1. % thiếu mỗi cột (train)            → 02_missing_bar.png
  2. Missingness matrix (missingno)     → 02_missing_matrix.png
  3. Cơ chế thiếu: χ² + effect size     → bảng + 02_missing_report.csv
  4. Phân phối cột thiếu (annotate n/%) → 02_missing_distributions.png
  5. Gợi ý chiến lược impute cho Step 3 (median/mean + cờ missing-indicator)

So với bản gốc:
  • Bản gốc chạy trên TOÀN BỘ data (peeking test). Bản này chỉ train.
  • Bản gốc phân loại MCAR/MAR bằng ngưỡng tay 0.05. Bản này thêm χ² p-value.
  • Vá lỗi lưới subplot khi có 0/1 cột thiếu; biến `i` không còn rò rỉ.
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import missingno as msno
from scipy.stats import chi2_contingency
from sklearn.model_selection import train_test_split

from pipeline_config import DATA_DIR, OUT_DIR   # noqa: E402

CLEAN_PATH = os.path.join(DATA_DIR, "ecommerce_churn_clean.csv")
TRAIN_PATH = os.path.join(DATA_DIR, "ecommerce_churn_train.csv")
TEST_PATH  = os.path.join(DATA_DIR, "ecommerce_churn_test.csv")
IDX_TR     = os.path.join(DATA_DIR, "train_index.npy")
IDX_TE     = os.path.join(DATA_DIR, "test_index.npy")

RANDOM_STATE = 42
TEST_SIZE    = 0.20
TARGET       = "Churn"
CAT_AS_STR   = {"CityTier": str, "Complain": str}


# ─────────────────────────────────────────────────────────────────────────────
# 0) SPLIT — lần chia DUY NHẤT, idempotent
#    Nếu đã có file split → load lại (để Step 3 và các step khác dùng CHUNG,
#    tránh chia hai lần lệch nhau). Nếu chưa → chia & lưu.
# ─────────────────────────────────────────────────────────────────────────────
def get_or_make_split(test_size=TEST_SIZE, random_state=RANDOM_STATE) -> pd.DataFrame:
    if os.path.exists(TRAIN_PATH) and os.path.exists(IDX_TR):
        train = pd.read_csv(TRAIN_PATH, dtype=CAT_AS_STR)
        train[TARGET] = train[TARGET].astype(int)
        print(f"[02] Reuse split sẵn có → train {train.shape} (KHÔNG chia lại).")
        return train

    df = pd.read_csv(CLEAN_PATH, dtype=CAT_AS_STR)
    df[TARGET] = df[TARGET].astype(int)
    train, test = train_test_split(
        df, test_size=test_size, stratify=df[TARGET], random_state=random_state
    )
    train.to_csv(TRAIN_PATH, index=False)
    test.to_csv(TEST_PATH, index=False)
    np.save(IDX_TR, train.index.to_numpy())
    np.save(IDX_TE, test.index.to_numpy())
    print(f"[02] Split stratified {1-test_size:.0%}/{test_size:.0%} — LẦN CHIA DUY NHẤT.")
    print(f"     Train {len(train):,} (churn {train[TARGET].mean()*100:.1f}%) | "
          f"Test {len(test):,} (churn {test[TARGET].mean()*100:.1f}%)")
    print(f"     Saved → train/test CSV + index .npy")
    return train


# ─────────────────────────────────────────────────────────────────────────────
# 1) % thiếu mỗi cột
# ─────────────────────────────────────────────────────────────────────────────
def missing_percent(df: pd.DataFrame) -> pd.Series:
    s = (df.isnull().sum() / len(df) * 100)
    return s[s > 0].sort_values(ascending=False)


def plot_missing_bar(miss: pd.Series):
    if miss.empty:
        print("[02] Train không có missing — bỏ qua bar chart.")
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(miss.index, miss.values, color="#ef4444")
    for b, v in zip(bars, miss.values):
        ax.text(v + 0.1, b.get_y() + b.get_height() / 2, f"{v:.1f}%", va="center", fontsize=10)
    ax.set_xlabel("Missing (%)"); ax.invert_yaxis()
    ax.set_title("Missing Data Rate by Column (Train)", fontsize=14, fontweight="bold")
    plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "02_missing_bar.png"), dpi=150); plt.close()
    print("[02] Saved: 02_missing_bar.png")


# ─────────────────────────────────────────────────────────────────────────────
# 2) Missingness matrix
# ─────────────────────────────────────────────────────────────────────────────
def plot_missing_matrix(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(14, 6))
    msno.matrix(df, ax=ax, sparkline=False, color=(0.93, 0.27, 0.27))
    ax.set_title("Missingness Matrix (Train)", fontsize=14, fontweight="bold")
    plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "02_missing_matrix.png"), dpi=150); plt.close()
    print("[02] Saved: 02_missing_matrix.png")


# ─────────────────────────────────────────────────────────────────────────────
# 3) Cơ chế thiếu — χ² + effect size  (DIAGNOSTIC ONLY)
#    Kiểm tra: cờ is_missing có phụ thuộc target Churn không?
#      • effect size = |churn(missing) − churn(present)|
#      • ý nghĩa    = χ²(is_missing × Churn) → p-value
#    Lưu ý: đây kiểm tra phụ thuộc-vào-target (proxy của MAR), KHÔNG phải
#    Little's MCAR test đầy đủ. Tham số impute thực tế vẫn fit TRÊN TRAIN ở Step 3.
# ─────────────────────────────────────────────────────────────────────────────
def missing_mechanism(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in df.columns[df.isnull().any()]:
        ind = df[col].isnull()
        n = int(ind.sum())
        if n == 0:
            continue
        ch_m = df.loc[ind,  TARGET].mean()
        ch_p = df.loc[~ind, TARGET].mean()
        diff = abs(ch_m - ch_p)
        try:
            chi2, p, _, _ = chi2_contingency(pd.crosstab(ind.astype(int), df[TARGET]))
        except Exception:
            chi2, p = np.nan, np.nan
        if not np.isnan(p) and p < 0.05 and diff > 0.05:
            mech = "MAR/MNAR"
        elif not np.isnan(p) and p < 0.05:
            mech = "MAR (yếu)"
        else:
            mech = "~MCAR"
        skew = df[col].dropna().skew() if pd.api.types.is_numeric_dtype(df[col]) else np.nan
        rows.append({
            "Column": col, "N_Missing": n, "Pct": round(n / len(df) * 100, 2),
            "Churn|miss": round(ch_m, 3), "Churn|pres": round(ch_p, 3),
            "|diff|": round(diff, 3),
            "chi2_p": (round(p, 4) if not np.isnan(p) else None),
            "Skew": (round(skew, 2) if not np.isnan(skew) else None),
            "Mechanism": mech,
        })
    rep = pd.DataFrame(rows).sort_values("N_Missing", ascending=False)
    if not rep.empty:
        rep.to_csv(os.path.join(OUT_DIR, "02_missing_report.csv"), index=False)
        print("\n[02] Cơ chế thiếu (χ² + effect size):")
        print(rep.to_string(index=False))
        print("[02] Saved: 02_missing_report.csv")
    return rep


# ─────────────────────────────────────────────────────────────────────────────
# 4) Phân phối cột thiếu — annotate n + missing% để không quên dữ liệu đã bị bỏ
# ─────────────────────────────────────────────────────────────────────────────
def plot_missing_distributions(df: pd.DataFrame, cols: list):
    cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    if not cols:
        print("[02] Không có cột numeric thiếu để vẽ phân phối.")
        return
    ncol = min(len(cols), 3)
    nrow = (len(cols) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 4 * nrow), squeeze=False)
    axes = axes.flatten()
    for i, col in enumerate(cols):
        d = df[col].dropna()
        miss_pct = df[col].isnull().mean() * 100
        axes[i].hist(d, bins=30, color="#3b82f6", alpha=0.8, edgecolor="white")
        axes[i].axvline(d.median(), color="#f59e0b", ls="--", lw=2, label=f"median={d.median():.1f}")
        axes[i].set_title(f"{col}\nskew={d.skew():.2f} | n={len(d):,} | missing={miss_pct:.1f}%",
                          fontsize=10)
        axes[i].legend(fontsize=8)
    for j in range(len(cols), len(axes)):   # ẩn ô thừa (không phụ thuộc biến `i`)
        axes[j].set_visible(False)
    plt.suptitle("Distribution of Columns with Missing (Train, NaN dropped)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "02_missing_distributions.png"), dpi=150); plt.close()
    print("[02] Saved: 02_missing_distributions.png")


# ─────────────────────────────────────────────────────────────────────────────
# 5) Gợi ý chiến lược impute cho Step 3
# ─────────────────────────────────────────────────────────────────────────────
def impute_recommendations(rep: pd.DataFrame):
    if rep.empty:
        return
    print(f"\n[02] GỢI Ý IMPUTE (Step 3 thực thi, fit TRÊN TRAIN):")
    for _, r in rep.iterrows():
        base = "median" if (r["Skew"] is None or abs(r["Skew"]) > 1) else "mean"
        flag = "  +cờ was_missing (thiếu mang tín hiệu)" if r["Mechanism"].startswith("MAR") else ""
        print(f"   {r['Column']:<28} thiếu {r['Pct']:>5.2f}%  [{r['Mechanism']:<9}] → {base}{flag}")
    print("   (lệch |skew|>1 → median robust; MAR/MNAR → thêm indicator để model dùng được "
          "chính sự-kiện-thiếu)")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY
# ─────────────────────────────────────────────────────────────────────────────
def analyze_missing():
    print(f"\n{'='*60}\nSTEP 2: MISSING DATA ANALYSIS (train only)\n{'='*60}")
    train = get_or_make_split()

    miss = missing_percent(train)
    plot_missing_bar(miss)
    plot_missing_matrix(train)
    rep = missing_mechanism(train)
    plot_missing_distributions(train, miss.index.tolist())
    impute_recommendations(rep)

    print(f"\n{'='*60}\n[02] ✓ Xong — report-only, test vẫn niêm phong.\n{'='*60}\n")
    return train, rep


if __name__ == "__main__":
    analyze_missing()