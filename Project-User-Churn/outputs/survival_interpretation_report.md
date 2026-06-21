# Survival Validation Report

## Pipeline Audit
- Event definition: Churn = 1 is treated as the churn event.
- Censoring definition: Churn = 0 is treated as right-censored, meaning the customer had not churned by the observed tenure.
- Time variable: Tenure is used as the duration variable. Zero-tenure records are shifted to 0.5 months for Cox model stability.
- Observed event rate: 16.8%; censored share: 83.2%.

## Why Hazard Ratios May Look Counterintuitive
Hazard ratios are conditional estimates, not simple correlations. SatisfactionScore HR > 1 or DaySinceLastOrder HR < 1 can appear when correlated variables suppress each other, when dissatisfied customers churn early, when recent buyers include both active and rescue-at-risk customers, or when the proportional hazards assumption is weak for a feature.

Business caveat: treat low-confidence or PH-violating features as directional signals for further investigation, not as standalone causal claims.

# Survival Business Interpretation

## Business-Friendly Findings
- Complain: HR=3.25. Customers with higher values have approximately 3.25x higher churn hazard, conditional on the other Cox covariates. Confidence: High.
- NumberOfDeviceRegistered: HR=1.49. Customers with higher values have approximately 1.49x higher churn hazard, conditional on the other Cox covariates. Confidence: Low.
- CityTier: HR=1.36. Customers with higher values have approximately 1.36x higher churn hazard, conditional on the other Cox covariates. Confidence: High.
- SatisfactionScore: HR=1.24. Customers with higher values have approximately 1.24x higher churn hazard, conditional on the other Cox covariates. Confidence: High.
- DaySinceLastOrder: HR=0.88. Customers with higher values have approximately 0.88x lower churn hazard, conditional on the other Cox covariates. Confidence: High.
- OrderCount: HR=1.08. Customers with higher values have approximately 1.08x higher churn hazard, conditional on the other Cox covariates. Confidence: High.
- NumberOfAddress: HR=1.04. Customers with higher values have approximately 1.04x higher churn hazard, conditional on the other Cox covariates. Confidence: Medium.
- HourSpendOnApp: HR=1.03. Customers with higher values have approximately 1.03x higher churn hazard, conditional on the other Cox covariates. Confidence: Low.