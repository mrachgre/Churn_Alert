# Feature Redundancy Summary

Mutual information is used here as the empirical information-gain signal for the binary churn target. The analysis aggregates one-hot encoded columns back to business-level feature names where appropriate.

## Why Model Performance Is High
The strongest predictors include direct behavioral recency, complaint status, tenure, cashback behavior, and engineered RFM/activity features. These variables are close to the churn mechanism, so the model can separate churned and retained customers with unusually strong discrimination.

## Most Dominant Features
- device_per_tenure: rank 1, MI=0.3617, corr=0.5804, mean SHAP impact=1.7972.
- Tenure: rank 2, MI=0.3610, corr=-0.5032, mean SHAP impact=0.4200.
- NumberOfAddress: rank 3, MI=0.2459, corr=0.0512, mean SHAP impact=0.8641.
- DaySinceLastOrder: rank 4, MI=0.2353, corr=-0.2115, mean SHAP impact=0.3051.
- Complain: rank 5, MI=0.0745, corr=0.2949, mean SHAP impact=0.8991.
- WarehouseToHome: rank 6, MI=0.2916, corr=0.0759, mean SHAP impact=0.3683.
- inactivity_ratio: rank 7, MI=0.2330, corr=0.2170, mean SHAP impact=0.1394.
- NumberOfDeviceRegistered: rank 8, MI=0.2120, corr=0.1631, mean SHAP impact=0.3727.
- CouponUsed: rank 9, MI=0.2235, corr=-0.0263, mean SHAP impact=0.7069.
- SatisfactionScore: rank 10, MI=0.1890, corr=0.1509, mean SHAP impact=0.4776.

## Potentially Dominant Features
- Tenure, DaySinceLastOrder, Complain

## Leakage Assessment
No definitive leakage is proven by this post-hoc ranking alone. However, very high ROC-AUC combined with dominant features such as complaint status, days since last order, tenure, cashback amount, or engineered recency ratios should be defended as behaviorally plausible and monitored as possible proxy leakage if any field was recorded after churn became known.