# Airline Passenger Satisfaction — System-Level Risk Insight Report

_Generated from SHAP analysis on test set (n=25,976 passengers)_

---
> **Scope:** Descriptive analysis only. No action recommendations included.

## 1. Risk Tier Distribution

| Risk Category | Count | % of Total |
|:-------------|------:|-----------:|
| Very Low | 8,538 | 32.9% |
| Low | 3,223 | 12.4% |
| Medium | 3,825 | 14.7% |
| High | 5,199 | 20.0% |
| Critical | 5,191 | 20.0% |

10,390 passengers (40.0%) are in the **High or Critical** tier.

## 2. Critical Group — Driver Frequency & Actual Ratings

Critical group: **5,191 passengers** (20.0% of test set)

| Driver Feature | % Critical Customers | Mean Rating (Critical) | Mean Rating (All) | Δ |
|:--------------|---------------------:|----------------------:|------------------:|--:|
| Seat comfort | 86.3% | 2.16 | 2.95 | -0.79 |
| Inflight entertainment | 68.2% | 2.43 | 3.46 | -1.03 |
| loyal | 48.6% | 0.51 | 0.82 | -0.30 |
| inflight_experience_score | 26.7% | 2.69 | 3.34 | -0.64 |
| Gate location | 9.8% | 3.07 | 2.99 | +0.07 |
| Ease of Online booking | 9.6% | 2.68 | 3.47 | -0.79 |
| Departure/Arrival time convenient | 9.0% | 2.75 | 3.15 | -0.39 |
| Age | 7.8% | 36.52 | 39.52 | -3.00 |
| Online support | 6.3% | 2.88 | 3.52 | -0.65 |
| Cleanliness | 5.8% | 3.20 | 3.71 | -0.51 |

**Top 3 drivers in the Critical group:**
- **Seat comfort**: present in 86.3% of Critical customers. Mean rating in Critical group: 2.16 (full-dataset mean: 2.95, Δ = -0.79).
- **Inflight entertainment**: present in 68.2% of Critical customers. Mean rating in Critical group: 2.43 (full-dataset mean: 3.46, Δ = -1.03).
- **loyal**: present in 48.6% of Critical customers. Mean rating in Critical group: 0.51 (full-dataset mean: 0.82, Δ = -0.30).

## 3. Cluster Persona Comparison — C0 vs C1

- **C0 (Hài Lòng Toàn Diện)**: 14,135 passengers (54.4%)
- **C1 (Trải Nghiệm Kém)**: 11,841 passengers (45.6%)

| Metric | C0 — Hài Lòng Toàn Diện | C1 — Trải Nghiệm Kém |
|:-------|:-----------------------:|:--------------------:|
| Mean risk_score | 0.350 | 0.673 |
| High/Critical rate | 17.1% | 67.3% |

**Mean service ratings by cluster:**

| Feature | C0 mean | C1 mean | C0−C1 |
|:--------|--------:|--------:|------:|
| Seat comfort | 3.42 | 2.38 | +1.04 |
| Inflight entertainment | 4.00 | 2.81 | +1.19 |
| Ease of Online booking | 4.24 | 2.57 | +1.67 |
| On-board service | 4.02 | 2.80 | +1.22 |
| Leg room service | 3.99 | 2.92 | +1.07 |
| Cleanliness | 4.18 | 3.15 | +1.03 |
| Food and drink | 3.36 | 2.54 | +0.82 |
| inflight_experience_score | 3.82 | 2.76 | +1.06 |
| ground_experience_score | 3.87 | 2.78 | +1.10 |

### 3b. C1 — Loyal vs Disloyal Sub-group Analysis

This section examines whether the SHAP prominence of `loyal` in C1 reflects genuine risk differences or a rating bias pattern.

**Loyal Customer** (n=8,916, 75.3% of C1):
  - Mean risk_score: 0.627
  - High/Critical rate: 61.1%

  Service rating means (within C1, Loyal Customer):
    - Seat comfort: 2.37
    - Inflight entertainment: 2.93
    - Ease of Online booking: 2.57
    - On-board service: 2.76
    - Leg room service: 2.89
    - Cleanliness: 3.05
    - Food and drink: 2.58
    - inflight_experience_score: 2.76
    - ground_experience_score: 2.75

**Disloyal Customer** (n=2,925, 24.7% of C1):
  - Mean risk_score: 0.811
  - High/Critical rate: 86.2%

  Service rating means (within C1, Disloyal Customer):
    - Seat comfort: 2.40
    - Inflight entertainment: 2.42
    - Ease of Online booking: 2.55
    - On-board service: 2.95
    - Leg room service: 2.99
    - Cleanliness: 3.45
    - Food and drink: 2.41
    - inflight_experience_score: 2.74
    - ground_experience_score: 2.86


| Feature | Loyal (C1) | Disloyal (C1) | Difference |
|:--------|----------:|-------------:|-----------:|
| Seat comfort | 2.37 | 2.40 | -0.03 |
| Inflight entertainment | 2.93 | 2.42 | +0.51 |
| Ease of Online booking | 2.57 | 2.55 | +0.02 |
| On-board service | 2.76 | 2.95 | -0.19 |
| Leg room service | 2.89 | 2.99 | -0.10 |
| Cleanliness | 3.05 | 3.45 | -0.40 |
| Food and drink | 2.58 | 2.41 | +0.17 |
| inflight_experience_score | 2.76 | 2.74 | +0.02 |
| ground_experience_score | 2.75 | 2.86 | -0.11 |

**Observation:** Mean risk_score — Loyal=0.627, Disloyal=0.811 (difference: -0.184).
Risk scores differ noticeably between Loyal and Disloyal sub-groups within C1, yet service ratings are similar — this pattern is consistent with a loyalty-related rating behavior difference rather than a large difference in actual service experience.

## 4. Top SHAP Drivers — C0 vs C1 Comparison (from Step B)

_Reference: SHAP mean|value| computed on test set per cluster._

| Rank | C0 Driver | C1 Driver |
|:----:|:----------|:----------|
| 1 | Seat comfort | Seat comfort |
| 2 | Inflight entertainment | Inflight entertainment |
| 3 | biz_travel | loyal ← **differs** |
| 4 | loyal | inflight_experience_score ← **differs** |
| 5 | Departure/Arrival time convenient | Departure/Arrival time convenient |
| 6 | inflight_experience_score | Gate location ← **differs** |
| 7 | Ease of Online booking | Ease of Online booking |
| 8 | Leg room service | biz_travel ← **differs** |
| 9 | Baggage handling | Baggage handling |
| 10 | Gate location | Cleanliness ← **differs** |

Key differences: `biz_travel` ranks 3rd in C0 but 8th in C1; `loyal` ranks 4th in C0 but 3rd in C1; `Gate location` ranks 10th in C0 but 6th in C1.

---
_Report based on XGB-Base model, test set n=25,976. SHAP values reflect model-derived feature contributions (log-odds scale)._