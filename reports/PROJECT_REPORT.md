# Project Report: Customer Churn Prediction and Segmentation Dashboard

## 1. Introduction

Customer churn is one of the most important business problems in the telecom industry. Churn happens when customers stop using a company's services and move to another provider. High churn can reduce revenue, increase customer acquisition costs, and affect long-term business growth.

This project focuses on building a machine learning based dashboard that can predict customer churn and segment customers into meaningful groups. The project uses the Telco Customer Churn dataset and combines supervised machine learning for churn prediction with unsupervised learning for customer segmentation.

The final output is an interactive Streamlit dashboard where users can view business KPIs, predict churn probability for individual customers (with an explanation of why), score customers in bulk, and explore data-driven customer segments.

## 2. Problem Statement

Telecom companies need to identify customers who are likely to leave their service before they actually churn. If high-risk customers are detected early, the business can take preventive actions such as offering discounts, improving customer support, or providing personalized service plans.

The main problem addressed in this project is:

- Predict whether a customer is likely to churn, with an honest estimate of how well that prediction generalizes.
- Understand customer behavior using data-driven segmentation.
- Provide an interactive dashboard for business decision-making.

## 3. Objectives

- To analyze telecom customer data.
- To preprocess numerical and categorical features.
- To build and compare multiple machine learning models without leaking test-set information into model selection or threshold tuning.
- To tune model thresholds and select the best model using a metric appropriate for an imbalanced target.
- To perform customer segmentation using K-Means clustering, choosing the number of clusters from the data rather than assuming it.
- To create a Streamlit dashboard for churn prediction, batch scoring, explainability, and customer segmentation.
- To present model results in a clear and business-friendly way.

## 4. Dataset Description

The dataset used in this project is:

```text
WA_Fn-UseC_-Telco-Customer-Churn.csv
```

The dataset contains customer-level information from a telecom company: 7,043 rows, demographic details, account information, subscribed services, billing details, and churn status. The target variable, `Churn`, is imbalanced — 26.5% of customers churned.

## 5. Data Preprocessing

1. Loaded the dataset with Pandas.
2. Converted `TotalCharges` to numeric (a handful of blank strings become `NaN`, later median-imputed).
3. Converted `Churn` to binary (`Yes` → 1, `No` → 0).
4. Dropped `customerID` — an identifier, not a predictive feature (see Section 13 for a bug this project found and fixed related to this).
5. Engineered `service_count`, counting how many optional services each customer subscribes to.
6. Applied median imputation + standard scaling to numeric features, and most-frequent imputation + one-hot encoding to categorical features, inside a single `ColumnTransformer` / `Pipeline`.

## 6. Feature Engineering

Four engineered features are shared between `train.py` and `app.py` via `features.py`'s `add_engineered_features`, so the training and serving paths can't drift apart:

- `service_count`: how many of PhoneService, OnlineSecurity, OnlineBackup, TechSupport, StreamingTV, StreamingMovies a customer has — overall service adoption; also used as one of the four segmentation features.
- `avg_monthly_spend = TotalCharges / (tenure + 1)`: a smoothed historical average charge, more stable than raw `TotalCharges` for short-tenure customers.
- `charge_delta = MonthlyCharges - avg_monthly_spend`: how far current billing has drifted from that historical average — an upsell/downsell signal neither raw column captures alone.
- `is_new_customer = tenure <= 3`: an explicit flag for the single strongest churn predictor (per the t-test in Section 8), giving linear models a clean split they can't derive from continuous tenure alone.

`avg_monthly_spend` turned out to be the single most important feature by mean |SHAP value| in the final model — a good sign it captures real signal rather than adding noise.

## 7. Model Building and Evaluation Methodology

An earlier version of this pipeline tuned the decision threshold on the same test set used for final reporting — a form of leakage that inflates the reported accuracy. This version avoids that:

1. **Held-out test set.** 20% of the data is set aside first and touched exactly once, at the very end.
2. **Cross-validated comparison.** The remaining 80% is used for 5-fold `StratifiedKFold` cross-validation across 9 candidates — 5 model families (Logistic Regression, Decision Tree, Random Forest, SVM, XGBoost) crossed with how they handle the imbalanced target (see Section 7a) — reporting Accuracy, Precision, Recall, F1, ROC-AUC, and PR-AUC as mean ± std.
3. **Selection metric: PR-AUC** (average precision). With a 26.5% positive rate, accuracy is a poor ranking metric (a model that predicts "no churn" for everyone still scores ~73.5%). ROC-AUC is threshold-independent but can stay high while minority-class precision/recall is weak; PR-AUC tracks minority-class ranking quality directly. ROC-AUC and the other four metrics are still reported for context.
4. **Light hyperparameter tuning.** `RandomizedSearchCV` (25 iterations, 3-fold, scored on average precision) tunes the top 2 CV candidates over widened parameter grids.
5. **Ensemble as a fair competitor.** A `VotingClassifier` (soft voting) combining the two tuned top candidates is cross-validated the same way as every other candidate, and only becomes the final model if it beats the best single model on mean CV PR-AUC.
6. **Probability calibration.** The selected model is wrapped in isotonic calibration (`CalibratedClassifierCV`, internal CV, training split only), because every imbalance treatment distorts predicted probabilities toward the positive class and the dashboard reports a probability.
7. **Leakage-free threshold tuning.** `cross_val_predict` produces out-of-fold *calibrated* probabilities on the training data; the threshold is chosen to maximize F1 over those out-of-fold predictions — never touching the test set.
8. **Single final evaluation.** The selected, calibrated model is refit on the full training portion and scored once on the untouched test set (metrics + Brier score).

## 7a. Handling Class Imbalance

Churn is 26.5% positive — mild imbalance, but enough to make accuracy misleading and to skew untreated probabilities. Three strategies were evaluated **as fair competing candidates on the same cross-validation**, never assumed:

| Strategy | Mechanism | Applied to |
| --- | --- | --- |
| Cost-sensitive learning | Re-weight the loss (`class_weight="balanced"`; `scale_pos_weight` for XGBoost) | all 5 families |
| SMOTE oversampling | Synthesize minority rows inside each training fold (`imblearn` pipeline) | LogReg, XGBoost |
| Random undersampling | Drop majority rows inside each training fold | LogReg, XGBoost |

Guardrails: `scale_pos_weight` is computed from `y_train` only (a previous version used the full `y`, leaking the test-set class ratio); resamplers sit inside an `imblearn.pipeline.Pipeline` so they only ever see the training portion of each fold.

**Finding: the three strategies are a statistical tie on this dataset.** Best cost-sensitive candidate (LogReg, class weights) = 0.6661 CV PR-AUC; best resampling candidate (LogReg, undersampling) = 0.6668 — within one standard deviation (±0.016). For imbalance this mild, the choice of imbalance technique is not the lever that matters; the honest report is that it made no measurable difference. What did matter: the PR-AUC selection metric (which re-ranked the candidates) and isotonic calibration (held-out Brier score 0.1484 → 0.1349), which is what makes the dashboard's probability and risk-band output trustworthy.

## 8. Model Evaluation Results

### Cross-Validated Comparison (training data, 5-fold, mean ± std, post-tuning, ranked by PR-AUC, including the ensemble candidate)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Voting Ensemble (LogReg undersample + LogReg class weights)** | 0.782 ± 0.015 | 0.571 ± 0.022 | 0.721 ± 0.029 | 0.637 ± 0.025 | 0.849 ± 0.012 | **0.667 ± 0.014** |
| Logistic Regression (undersample) | 0.798 ± 0.013 | 0.615 ± 0.024 | 0.634 ± 0.030 | 0.624 ± 0.026 | 0.849 ± 0.012 | 0.667 ± 0.013 |
| Logistic Regression (class weights) | 0.755 ± 0.013 | 0.525 ± 0.016 | 0.799 ± 0.034 | 0.634 ± 0.021 | 0.849 ± 0.012 | 0.667 ± 0.016 |
| Logistic Regression (SMOTE) | 0.759 ± 0.017 | 0.532 ± 0.023 | 0.785 ± 0.037 | 0.634 ± 0.026 | 0.847 ± 0.012 | 0.665 ± 0.017 |
| XGBoost (scale_pos_weight) | 0.761 ± 0.006 | 0.535 ± 0.007 | 0.762 ± 0.026 | 0.629 ± 0.013 | 0.842 ± 0.008 | 0.656 ± 0.017 |
| XGBoost (SMOTE) | 0.797 ± 0.010 | 0.620 ± 0.021 | 0.608 ± 0.026 | 0.614 ± 0.019 | 0.842 ± 0.009 | 0.653 ± 0.022 |
| XGBoost (undersample) | 0.743 ± 0.008 | 0.510 ± 0.010 | 0.783 ± 0.019 | 0.617 ± 0.012 | 0.839 ± 0.010 | 0.645 ± 0.023 |
| Random Forest (class weights) | 0.782 ± 0.008 | 0.580 ± 0.013 | 0.652 ± 0.025 | 0.614 ± 0.018 | 0.828 ± 0.008 | 0.625 ± 0.027 |
| SVM (class weights) | 0.798 ± 0.015 | 0.626 ± 0.031 | 0.596 ± 0.029 | 0.610 ± 0.028 | 0.834 ± 0.012 | 0.612 ± 0.024 |
| Decision Tree (class weights) | 0.715 ± 0.015 | 0.479 ± 0.017 | 0.783 ± 0.033 | 0.594 ± 0.010 | 0.810 ± 0.006 | 0.595 ± 0.018 |

### Best Model

**Voting Ensemble** (soft voting) of tuned Logistic Regression (random undersampling) + tuned Logistic Regression (class weights), selected for the highest mean CV PR-AUC. Two caveats stated plainly:

- The top three rows (0.6670 / 0.6666 / 0.6666) are a statistical tie, all within ±0.014. The ensemble wins on the mean and is evaluated fairly, but a single tuned Logistic Regression would be an equally defensible final model. Its two members are both Logistic Regression, so its diversity — and its edge — is small.
- **Logistic Regression outranks XGBoost on PR-AUC here.** Once `avg_monthly_spend` / `charge_delta` are engineered, Telco churn is close to linearly separable, so a regularized linear model matches gradient boosting on ranking quality.

Isotonic calibration: held-out Brier score **0.1484 → 0.1349**. Tuned decision threshold: **0.33** (maximizes F1 on out-of-fold *calibrated* training probabilities — lower than a raw-score threshold because calibration has already pulled the scores toward the base rate).

### Final Held-Out Test Metrics (evaluated once, calibrated model)

- Accuracy: 0.7700
- Precision: 0.5502
- Recall: 0.7326
- F1 Score: 0.6284
- ROC-AUC: 0.8464
- PR-AUC: 0.6581
- Brier score: 0.1349

Confusion matrix:

```text
[[811 224]
 [100 274]]
```

The model catches 73% of actual churners (recall) — a deliberate choice: for a retention use case, missing an at-risk customer (false negative) is usually costlier than a wasted retention offer (false positive), so the threshold was tuned toward F1 rather than accuracy.

### What Actually Moved the Needle

- **The imbalance strategy did not.** Cost-sensitive learning, SMOTE, and random undersampling land within noise of each other (Section 7a). For imbalance this mild that is the expected result, and it is reported rather than hidden behind a claim that resampling "solved" anything.
- **Switching the selection metric from ROC-AUC to PR-AUC did.** It re-ranked the candidates and surfaced that Logistic Regression, not XGBoost, is the stronger ranker on this feature set — the previous round's ROC-AUC-based selection had put XGBoost second.
- **Isotonic calibration did.** It is the change that turns the dashboard's "churn probability" from a directional score into a number that matches observed frequencies (Brier 0.1484 → 0.1349; batch-scoring a fresh list now returns a mean probability ≈ the true 26% base rate).
- **Engineered features** (`avg_monthly_spend`, `charge_delta`, `is_new_customer`) still help the linear model most, since it cannot derive a ratio like `TotalCharges / tenure` on its own.

`avg_monthly_spend` is the single most important feature by mean |SHAP value| in the final model, ahead of monthly charges and fiber-optic internet — a good sign it is carrying real signal.

## 9. Model Explainability (SHAP)

Since the final model is a `VotingClassifier`, `train.py` and `app.py` share a `get_shap_explainer` helper (`features.py`) that dispatches to `TreeExplainer`/`LinearExplainer` per member (unwrapping resampling-pipeline members first) and averages their attributions weighted by voting weight — so explainability keeps working no matter which candidate wins a given training run. SHAP runs on the **uncalibrated** pipeline (`models/churn_model_base.joblib`); isotonic calibration is a monotonic squash of the score and would only force a slow model-agnostic explainer. `train.py` saves the resulting global feature-importance plot to `reports/shap_summary.png`. The top churn drivers by mean |SHAP value| are:

1. `avg_monthly_spend` (engineered feature — the single strongest driver)
2. MonthlyCharges
3. Fiber optic internet service
4. Tenure
5. `service_count`

These align with the independent t-test and chi-square hypothesis tests run in the notebook (tenure, MonthlyCharges, TotalCharges, and Contract are all significant at p < 0.001), which is a useful cross-check that the model learned something consistent with the underlying statistics rather than spurious correlations — and that `avg_monthly_spend` ranking first is a real signal, not noise. The dashboard also computes a per-prediction SHAP breakdown so a business user can see *why* a specific customer was flagged.

## 10. Customer Segmentation

K-Means clustering groups customers using tenure, MonthlyCharges, TotalCharges, and service_count. Rather than assuming a fixed number of segments, the number of clusters is chosen from the data:

- Silhouette score is scanned for k = 2 to 8. It's maximized at k=2, which is too coarse to be business-actionable.
- The search is restricted to a business-usable range (k = 3 to 6); the best silhouette score in that range picks **k = 3**.
- Segment names are derived from each cluster's statistics rather than assumed from cluster index (which is arbitrary and not stable across retrains): highest churn rate → "At Risk", lowest remaining tenure → "New", highest remaining spend → "High Value", and so on.

### Resulting Segments

| Segment | Size | Avg Tenure | Avg Monthly Charges | Avg Total Charges | Avg Services | Churn Rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| At Risk Customers | 2617 | 15.2 | $75.79 | $1,148.78 | 2.5 | 45.8% |
| New Customers | 2245 | 27.2 | $27.15 | $721.13 | 1.1 | 15.5% |
| High Value Customers | 2181 | 58.3 | $90.25 | $5,248.17 | 4.2 | 14.9% |

The At Risk segment's churn rate is roughly 3x the other two — a directly actionable finding for a retention team.

## 11. Dashboard Development

The dashboard has five pages: Overview (KPIs), Prediction (single-customer churn probability + SHAP explanation), Batch Prediction (CSV upload/download), Segmentation (live segment profile + segment lookup), and Model Performance (CV comparison table, SHAP plot, confusion matrix/ROC curve).

## 12. Tools and Technologies Used

Python, Pandas, NumPy, Scikit-learn, imbalanced-learn, XGBoost, SHAP, Streamlit, Matplotlib, Seaborn, Joblib.

## 13. A Bug Found and Fixed During This Project

The original segmentation exploration in the notebook accidentally included `customerID` in the clustering feature set (via an empty `id_cols` placeholder list). Since `customerID` is a unique string per row, one-hot encoding it produced one dummy column per customer — silently destroying any real cluster structure and pulling the naive silhouette analysis toward a degenerate k=2. The same placeholder bug also affected the churn model's feature set in the notebook. Both were fixed by explicitly excluding `customerID` from the feature set; `train.py`'s segmentation was unaffected since it only ever used the four numeric business features. This is a good example of why silhouette/cluster results should be sanity-checked against the actual feature list, not just accepted at face value.

## 14. Business Insights

- The At Risk segment (37% of customers) churns at 3x the rate of the other segments — the clearest lever for a retention campaign.
- Spend level (the engineered `avg_monthly_spend`, plus raw MonthlyCharges), fiber-optic internet, and short tenure are the dominant churn drivers (SHAP + independent hypothesis tests agree); month-to-month contracts follow.
- The dashboard supports both single-customer lookups and bulk CSV scoring, so it's usable for both individual retention outreach and portfolio-wide risk review.

## 15. Limitations

- The dashboard's per-prediction SHAP explanation is fast for tree-based/linear models; a different winning model type could require a slower generic explainer.
- The SMOTE candidates run on the one-hot-encoded matrix, so SMOTE interpolates binary dummy columns into fractional values — a common approximation rather than the categorically-correct `SMOTENC`. It is kept because it uses the same uniform pipeline structure as every other candidate and synthetic oversampling was not competitive on this dataset regardless; random undersampling (drops rows only) has no such caveat.
- The winning ensemble combines two Logistic Regression variants and is close to redundant — its CV margin over a single tuned Logistic Regression is within noise. It is kept per the project's "the ensemble competes and wins on the metric" rule; a reviewer could reasonably prefer the simpler single model.
- The full-dataset confusion matrix/ROC curve on the Model Performance page are for a quick visual check only — the cross-validated table is the honest generalization estimate.
- The project is not yet deployed to a public URL.

## 16. Future Scope

- Deploy on Streamlit Cloud, Render, or Hugging Face Spaces.
- Add downloadable prediction reports (PDF/HTML).
- Periodically retrain and monitor for model drift if the underlying data is refreshed.

## 17. Conclusion

This project builds an end-to-end, leakage-free machine learning solution for customer churn prediction and customer segmentation. Cross-validation, an imbalance-aware selection metric (PR-AUC), a fair head-to-head of cost-sensitive learning against resampling, isotonic probability calibration, out-of-fold threshold tuning, light hyperparameter search, SHAP explainability, and data-driven segmentation together make this a substantially more rigorous pipeline than a single train/test split with accuracy-based selection. Part of that rigor is reporting negative results honestly — the imbalance-technique comparison found no measurable winner, and that is stated rather than dressed up — the kind of methodology that holds up under scrutiny in a Data Science interview.
