"""
pipeline_config.py — Central path configuration for Churn Alert pipeline.

Mỗi dataset có thư mục riêng biệt:
  data/{dataset}/     — processed CSV files
  models/{dataset}/   — trained model artifacts
  outputs/{dataset}/  — visualisation outputs

Sử dụng:
  1. run_pipeline.py đặt os.environ["PIPELINE_DATASET"] = "E_Commer_Data"
     TRƯỚC KHI import bất kỳ step nào.
  2. Khi chạy standalone (python src/05_preprocessing.py),
     mặc định PIPELINE_DATASET = "E_Commer_Data".
"""

import os

_BASE = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))

# Dataset name — có thể override bằng env var
DATASET = os.environ.get("PIPELINE_DATASET", "E_Commer_Data")

DATA_DIR  = os.path.join(_BASE, "data",    DATASET)
MODEL_DIR = os.path.join(_BASE, "models",  DATASET)
OUT_DIR   = os.path.join(_BASE, "outputs", DATASET)

# Ensure directories exist
for _d in (DATA_DIR, MODEL_DIR, OUT_DIR):
    os.makedirs(_d, exist_ok=True)
