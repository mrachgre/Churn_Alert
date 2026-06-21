# Research Findings

This section summarizes the main empirical contributions of the churn project in an academic reporting style. The findings combine ablation-style validation, leakage-aware diagnostics, clustering interpretation, survival modeling, fairness, stability, and business cost analysis.

## Finding 1: Does RFM improve churn prediction?
- Hypothesis: RFM features provide incremental predictive signal beyond behavioral variables.
- Methodology: Bootstrap ROC-AUC comparison using saved model predictions and RFM feature occlusion.
- Result: ROC-AUC difference = 0.0371
- Conclusion: Predictive improvement is potentially meaningful and should be validated on future cohorts.
- Business impact: Even if prediction lift is small, RFM remains useful for segmentation and campaign design.

## Finding 2: Do cluster features improve churn prediction and interpretation?
- Hypothesis: K-means membership captures latent behavioral structure.
- Methodology: Bootstrap ROC-AUC comparison plus persona profiling by churn and engagement features.
- Result: ROC-AUC difference = 0.0007; personas generated = 4.
- Conclusion: Predictive improvement is negligible.
- Business impact: Cluster personas turn model outputs into targeted retention strategies.

## Finding 3: Which customer factors affect churn timing?
- Hypothesis: Complaint and engagement variables change churn hazard over tenure.
- Methodology: Cox proportional hazards model with PH assumption diagnostics.
- Result: 7 Cox features have medium/high interpretation confidence.
- Conclusion: Survival analysis is defensible when PH diagnostics and confidence flags are reported.
- Business impact: Hazard ratios support timing-aware retention prioritization.

## Finding 4: Which model produces the best economic retention value?
- Hypothesis: The highest discrimination model also reduces expected churn cost.
- Methodology: Confusion-matrix cost model with intervention cost and avoided churn loss.
- Result: Best model = Random Forest, cost reduction = 82.79%.
- Conclusion: Business value should be judged by expected savings, not ROC-AUC alone.
- Business impact: Provides a deployment-oriented criterion for retention model selection.

## Finding 5: Does the churn model show disparity across protected groups?
- Hypothesis: Model error and selection rates are similar across Gender and MaritalStatus.
- Methodology: Group metrics, demographic parity difference, and equal opportunity difference.
- Result: Maximum observed disparity = 18.26%.
- Conclusion: Disparity requires monitoring
- Business impact: Supports responsible retention targeting and identifies monitoring needs.

Overall, the project moves beyond point-estimate ROC-AUC by adding uncertainty, interpretability, operational cost, fairness, time-to-event reasoning, and segment-level actionability.