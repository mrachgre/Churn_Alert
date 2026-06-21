# Model Stability Summary

Repeated Stratified K-Fold validation used 5 folds and 10 repetitions. This does not replace the original train/test split; it provides a robustness audit across resampled training folds.

- Logistic Regression: ROC-AUC = 0.913 +/- 0.006, F1 = 0.842 +/- 0.007.
- Random Forest: ROC-AUC = 0.997 +/- 0.001, F1 = 0.976 +/- 0.004.
- XGBoost: ROC-AUC = 0.990 +/- 0.002, F1 = 0.948 +/- 0.006.

Best mean discrimination is observed for Random Forest. Low standard deviation supports generalization confidence; larger variance would indicate dependence on a favorable split.