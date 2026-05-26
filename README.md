# 📺 StreamFlix Customer Churn Prediction

## 📌 Overview

A supervised machine learning pipeline that predicts which StreamFlix customers are likely to churn **within the next 30 days** — before they actually cancel. The goal is to give the business enough lead time to intervene with targeted retention strategies.

The model achieves **94% recall** on churners at an optimized threshold of 0.3, meaning it catches 356 out of 378 at-risk customers, giving retention teams a powerful early warning system.

---

## 🎯 Business Problem

Cancellations only appear in billing reports after it's too late to intervene. By predicting churn probability in advance, StreamFlix can:

- Launch **targeted discount campaigns** for high-risk customers
- Send **personalized content recommendations** to re-engage passive users
- Trigger **inactivity alerts** for users who haven't watched in 7+ days
- Offer **payment support** to customers with billing friction

---

## 📊 Dataset

| Property | Value |
|----------|-------|
| Records | 10,000 customers |
| Original Features | 37 |
| Engineered Features | 12 new → 49 total |
| Target Variable | `will_churn` (0 = retained, 1 = churned) |
| Churn Rate | 18.89% |
| Class Imbalance | 81% retained / 19% churned |

**Class imbalance handling:** SMOTE (Synthetic Minority Oversampling Technique) applied strictly within each training fold to prevent data leakage.

---

## 🔬 Exploratory Data Analysis

Key findings from EDA that directly shaped feature engineering:

| Finding | Action Taken |
|---------|-------------|
| Churned customers had significantly more days since last watch | Created `inactive_user_flag` (>7 days = inactive) |
| Annual billing customers churn less (financial commitment) | Included billing cycle as a predictor |
| Higher plan tiers show different churn rates | Subscription plan tier included as feature |
| Payment failures and account holds drive churn | Combined into `account_health_score` |

---

## 🛠️ Feature Engineering

12 new features added, all grounded in EDA findings:

| Feature | Description |
|---------|-------------|
| **Customer Lifecycle Stage** | Tenure-based loyalty tiers (New / Mid / Loyal) — captures non-linearity in early churn risk |
| **Engagement Depth** | Composite of sessions, unique titles, watchlist activity, ratings — binary split above/below median |
| **Inactive User Flag** | Binary: 1 if no watch session in 7+ days |
| **Session Gap Metric** | Continuous: average time between watch sessions |
| **Account Health Score** | Combined payment failures + account holds into single risk score |
| **Support Ticket Flag** | Binary: 1 if open support ticket exists |

---

## 🤖 Models Evaluated

| Model | ROC-AUC | Precision | Recall | F1 Score |
|-------|---------|-----------|--------|----------|
| Logistic Regression | 0.7028 | 0.3495 | **0.4554** | **0.3953** |
| Random Forest | **0.7402** | **0.5151** | 0.0602 | 0.1075 |
| XGBoost | 0.7262 | 0.4493 | 0.2019 | 0.2784 |

**Why Logistic Regression?** Although Random Forest achieved the highest ROC-AUC, it detected almost no actual churners (Recall: 0.06). For churn prediction, **Recall is the critical metric** — missing a churner costs more than incorrectly flagging a retained customer. Logistic Regression balances recall while remaining interpretable.

---

## ⚙️ Hyperparameter Tuning

GridSearchCV was used to optimize regularization strength `C`.

| Configuration | ROC-AUC |
|---------------|---------|
| Default (C=1.0) | 0.702 |
| **Tuned (C=0.00215)** | **0.737** |

---

## 🎚️ Threshold Optimization

Default classification threshold (0.5) is too conservative for churn prediction. We optimize for maximum recall while keeping precision manageable:

| Threshold | Precision | Recall |
|-----------|-----------|--------|
| 0.1 | 0.190 | 1.000 |
| 0.2 | 0.199 | 0.976 |
| **0.3** ✅ | **0.234** | **0.942** |
| 0.4 | 0.271 | 0.794 |
| 0.5 | 0.328 | 0.643 |

**Selected threshold: 0.3** — captures 94% of churners while keeping false positive rate manageable.

---

## 📈 Final Model Performance (Threshold = 0.3)

**Confusion Matrix:**

| | Predicted No Churn | Predicted Churn |
|--|--|--|
| **Actual No Churn** | 456 | 1,166 |
| **Actual Churn** | 22 | **356** |

The model correctly identifies **356 out of 378 churners**, missing only 22. High false positives (1,166) are acceptable because the cost of losing a customer outweighs the cost of offering a retention discount to a non-churner.

---

## 🎯 Customer Risk Segmentation

To address false positives and help businesses prioritize resources efficiently, customers are segmented into three risk groups based on predicted churn probability:

| Risk Segment | Count | Mean Probability | Share |
|-------------|-------|-----------------|-------|
| Low Risk | 2,458 | 0.220 | 24.6% |
| Medium Risk | 5,383 | 0.440 | 53.8% |
| **High Risk** | **2,159** | **0.722** | **21.6%** |

**22% of customers are high-risk** (avg 72% churn probability). Target these first.

---

## 🔑 Key Churn Drivers

**Top Risk Indicators (churn ↑):**
- Account on hold
- Inactive users (7+ days without watching)
- Long time since last watch session
- New customers (early lifecycle risk)

**Retention Signals (churn ↓):**
- High content completion rate
- Binge-watching behavior
- High engagement levels
- Longer subscription tenure
- Annual billing plan

---

## 💼 Business Recommendations

| Segment | Action |
|---------|--------|
| **High Risk** | Targeted discounts, personalized recommendations, proactive support outreach |
| **Medium Risk** | Content recommendations, encourage watchlist usage |
| **Low Risk** | Regular updates and new content notifications |
| **Inactive (7+ days)** | Email reminders, "continue watching" push notifications |
| **Payment failures** | Payment reminders, alternative payment options |

---

## 🚀 Model Deployment

The trained model is saved for production reuse:
```python
import joblib
model = joblib.load('streamflix_churn_model.pkl')
churn_probs = model.predict_proba(new_customers)[:, 1]
high_risk = new_customers[churn_probs > 0.3]
```

This allows StreamFlix to **periodically score new customer data** and flag at-risk users without retraining.

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)

| Library | Purpose |
|---------|---------|
| `pandas`, `numpy` | Data manipulation |
| `scikit-learn` | Logistic Regression, Random Forest, cross-validation, SMOTE |
| `xgboost` | XGBoost classifier |
| `imbalanced-learn` | SMOTE for class imbalance |
| `matplotlib`, `seaborn` | EDA and performance visualizations |
| `joblib` | Model serialization |



## 🚀 Getting Started

```bash
git clone https://github.com/KHUSHI0809/streamflix-churn-prediction.git
cd streamflix-churn-prediction
pip install -r requirements.txt
jupyter notebook notebooks/Data_602_Midterm_Code_final.ipynb
```

---

## 🏷️ Topics

`churn-prediction` `machine-learning` `logistic-regression` `smote` `class-imbalance` `customer-analytics` `feature-engineering` `threshold-optimization` `risk-segmentation` `python` `scikit-learn`
