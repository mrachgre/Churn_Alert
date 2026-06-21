# Statistical Validation Summary

Bootstrap resampling was used to estimate whether observed ROC-AUC differences are stable enough to support research claims. Positive differences favor the enhanced condition named in the experiment.

## With vs Without RFM
- Method: XGBoost baseline compared with an RFM-occluded test matrix
- Mean ROC-AUC: 0.9863
- Difference: 0.0371 with 95% CI [0.0186, 0.0586]
- Interpretation: Practically meaningful

## With vs Without Cluster Features
- Method: XGBoost with k-means cluster feature compared with base XGBoost
- Mean ROC-AUC: 0.987
- Difference: 0.0007 with 95% CI [-0.0077, 0.0122]
- Interpretation: Not statistically distinguishable

## Leakage Removal Experiment
- Method: Distance-feature logistic model compared with base logistic model
- Mean ROC-AUC: 0.8979
- Difference: 0.0135 with 95% CI [0.0029, 0.0244]
- Interpretation: Practically meaningful

Small ROC-AUC changes should be interpreted alongside segmentation value, fairness, cost reduction, and operational usefulness.