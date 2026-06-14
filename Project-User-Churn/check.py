import joblib, pandas as pd, numpy as np

calibrated = joblib.load("models/xgb_clust_model.pkl")
X_test_clust = pd.read_csv("data/X_test_clust.csv")
y_test = pd.read_csv("data/y_test.csv").squeeze()

probs = calibrated.predict_proba(X_test_clust)[:, 1]
print("min/max/mean:", probs.min(), probs.max(), probs.mean())
print("unique values (top 10):", np.unique(probs)[:10])
print("% rows == 0.0:", (probs == 0.0).mean())
print("classes_:", calibrated.classes_)
