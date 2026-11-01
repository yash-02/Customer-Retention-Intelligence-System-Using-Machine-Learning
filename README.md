# Customer Churn Prediction and Segmentation Dashboard

## Project Overview

This project is an end-to-end machine learning dashboard designed to analyze telecom customer behavior, predict customer churn, and group customers into meaningful business segments. It uses the Telco Customer Churn dataset and combines supervised learning for churn prediction with unsupervised learning for customer segmentation.

The application is built using Streamlit, allowing users to interact with the trained machine learning models through a simple web dashboard. The project is useful for understanding which customers are likely to leave a telecom service and how different customer groups behave based on tenure, charges, and service usage.

## Problem Statement

Customer churn is a major challenge for telecom companies because acquiring new customers is often more expensive than retaining existing ones. By identifying customers who are likely to churn, businesses can take proactive actions such as offering discounts, improving customer support, or recommending better service plans.

This project aims to answer two key business questions:

- Which customers are most likely to churn, and why (in terms a business user can act on)?
- What customer segments exist based on billing, tenure, and service usage patterns?

## Objectives

- Analyze telecom customer data to understand churn behavior.
- Build a machine learning model to predict customer churn, evaluated in a way that avoids overstating its accuracy.
- Create data-driven customer segments using clustering techniques.
- Develop an interactive Streamlit dashboard for business users, including batch scoring and model explainability.
- Provide churn risk labels such as low, medium, and high risk.
- Save trained models for reuse in the dashboard.

## Dataset

The project uses the Telco Customer Churn dataset:

```text
WA_Fn-UseC_-Telco-Customer-Churn.csv
```

The dataset contains information about telecom customers, including:

- Demographic details such as gender, senior citizen status, partner, and dependents.
- Account information such as tenure, contract type, payment method, and billing preference.
- Services used such as phone service, internet service, online security, tech support, streaming TV, and streaming movies.
- Billing information such as monthly charges and total charges.
- Churn status indicating whether the customer left the company.

## Project Workflow

1. Load the Telco Customer Churn dataset.
2. Clean and preprocess the data, and convert `TotalCharges` and `Churn` into usable numeric/binary form.
3. Create a new feature called `service_count`.
4. Separate numerical and categorical features, and build a preprocessing pipeline (imputation, scaling, one-hot encoding).
5. Hold out a stratified test set (20%) that is untouched until the final evaluation.
6. Compare 9 classification candidates (5 model families × cost-sensitive / SMOTE / undersampling imbalance strategies) using 5-fold cross-validation on the training portion only, ranked by PR-AUC.
7. Run a light hyperparameter search (`RandomizedSearchCV`, scored on average precision) on the top 2 candidates.
8. Build a soft-voting ensemble of the top 2 and cross-validate it as one more candidate; select the overall winner by CV PR-AUC.
9. Wrap the selected model in isotonic probability calibration, then tune the decision threshold on out-of-fold *calibrated* predictions (no leakage) and refit on the full training portion.
10. Evaluate the final calibrated model exactly once on the held-out test set (metrics + Brier score).
11. Generate SHAP explainability artifacts (from the uncalibrated pipeline) for the selected model.
12. Select the number of segmentation clusters via silhouette analysis, fit K-Means, and derive business-friendly segment names from each cluster's statistics.
13. Save the trained models, tuned threshold, evaluation results, and segment profile to `models/` and `reports/`.
14. Serve everything through an interactive Streamlit dashboard.

## Machine Learning Approach

### Churn Prediction

The churn prediction workflow avoids a common pitfall in projects like this one: picking a decision threshold and reporting accuracy using the *same* test set (which silently inflates the headline number). Instead:

- **Cross-validation, not a single split.** All candidates are compared with 5-fold `StratifiedKFold` cross-validation on the training portion (80% of the data). The held-out test set (20%) is never touched until the very end.
- **Selection metric: PR-AUC** (average precision), not raw accuracy or ROC-AUC. With churn at ~26.5% positive, accuracy rewards a model that just predicts "no churn" for everyone; ROC-AUC is threshold-independent but can look healthy while minority-class precision/recall is poor. PR-AUC tracks minority-class ranking quality directly. ROC-AUC and four other metrics are still reported for context.
- **Engineered features.** Beyond `service_count`, `avg_monthly_spend` (`TotalCharges / (tenure + 1)`, a smoothed historical charge) and `charge_delta` (how far current billing has drifted from that average) and `is_new_customer` (tenure ≤ 3 months, the single strongest predictor per the notebook's t-test) are added before the preprocessing pipeline. `avg_monthly_spend` turned out to be the single most important feature by SHAP, ahead of monthly charges and fiber-optic internet.
- **Light hyperparameter tuning.** `RandomizedSearchCV` (scored on average precision) searches a grid for the top 2 CV candidates.
- **Soft-voting ensemble as a fair competitor.** A `VotingClassifier` combining the two tuned top candidates is cross-validated the same way as every other candidate — it only becomes the final model if it actually beats the best single model on mean CV PR-AUC, not by default.
- **Probability calibration.** The selected model is wrapped in isotonic calibration (`CalibratedClassifierCV`, fit with its own internal CV on the training split) so the dashboard's "churn probability" and Low/Medium/High risk bands mean what they say — see [Handling class imbalance](#handling-class-imbalance) below.
- **Leakage-free threshold tuning.** The decision threshold is chosen by maximizing F1 on *out-of-fold, calibrated* probabilities (`cross_val_predict`) from the training data only — never on the test set.
- **One final evaluation.** The selected, refit, calibrated model is scored on the untouched test set exactly once, at the tuned threshold.

The base candidates compared are:

- Logistic Regression — cost-sensitive (`class_weight="balanced"`), SMOTE oversampling, and random undersampling
- Decision Tree — cost-sensitive
- Random Forest — cost-sensitive
- Support Vector Machine (RBF kernel, calibrated for probabilities) — cost-sensitive
- XGBoost — cost-sensitive (`scale_pos_weight`), SMOTE oversampling, and random undersampling

The pipeline includes median imputation + standard scaling for numeric features, most-frequent imputation + one-hot encoding for categorical features. How each candidate handles the imbalanced target is itself part of the comparison, described next.

### Handling class imbalance

Churn is ~26.5% positive — mild imbalance, but enough that accuracy is a misleading headline and untreated probabilities skew. Three strategies are evaluated **as fair competing candidates on the same cross-validation**, not assumed:

| Strategy | How | Applied to |
| --- | --- | --- |
| Cost-sensitive learning | Re-weight the loss (`class_weight="balanced"`, or `scale_pos_weight` for XGBoost) | LogReg, Decision Tree, Random Forest, SVM, XGBoost |
| SMOTE oversampling | Synthesize new minority rows inside each training fold (`imblearn` pipeline) | LogReg, XGBoost |
| Random undersampling | Drop majority rows inside each training fold | LogReg, XGBoost |

Rules that keep this honest:

- **No leakage.** `scale_pos_weight` is computed from `y_train` only (a previous version used the full `y`, leaking the test set's class ratio). Resamplers live inside an `imblearn.pipeline.Pipeline`, so they only ever touch the training portion of each CV fold — never the validation fold, never the held-out test set.
- **Result: cost-sensitive learning and resampling are statistically tied here.** Best cost-sensitive candidate (LogReg, class weights) = 0.6661 CV PR-AUC; best resampling candidate (LogReg, undersampling) = 0.6668 — well within one standard deviation (±0.016). SMOTE gave no edge, partly because interpolating one-hot columns produces fractional "categories" (see [Limitations](#limitations)). This is the expected outcome for imbalance this mild, and it's worth stating plainly rather than claiming resampling "fixed" anything.
- **Threshold, not just weights.** The 0.5 default is never assumed; the operating point is tuned on out-of-fold predictions to maximize F1 (missing an at-risk customer is treated as roughly as costly as a wasted retention offer).
- **Calibrated probabilities.** Any imbalance treatment distorts predicted probabilities toward the positive class. Isotonic calibration on the training split corrects this: held-out **Brier score improved from 0.148 → 0.135**, and batch-scoring a fresh customer list now returns a mean probability (~0.26) that matches the true base rate.

### Achieved Results

5-fold cross-validation on the training split (mean ± std), ranked by PR-AUC, after hyperparameter tuning of the top 2 candidates and adding the voting ensemble as a competing candidate:

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

**Selected model: Voting Ensemble** of tuned Logistic Regression (random undersampling) + tuned Logistic Regression (class weights), soft voting. Two honest caveats:

- The top three rows are a **statistical tie** (0.6670 vs 0.6666 vs 0.6666 CV PR-AUC, all within ±0.014). The ensemble wins on the mean and is evaluated fairly as a candidate — but on a dataset this size, a single tuned Logistic Regression with either imbalance treatment would be a defensible choice too. The ensemble members are both Logistic Regression, so the diversity (and the gain) is limited; it's kept because the project's rule is "the ensemble competes and wins on the metric," and it did.
- **XGBoost ranks below Logistic Regression on PR-AUC** here. Telco churn is close to linearly separable once `avg_monthly_spend` / `charge_delta` are engineered, so a regularized linear model competes with gradient boosting on ranking quality.

**Tuned threshold:** 0.33 (maximizes F1 on out-of-fold *calibrated* training probabilities — lower than a raw-probability threshold because isotonic calibration has already pulled the scores back toward the true base rate).

**Probability calibration:** held-out Brier score **0.1484 → 0.1349** after isotonic calibration of the selected model.

**Final held-out test set metrics** (evaluated once, at that threshold, on the calibrated model):

| Metric | Value |
| --- | ---: |
| Accuracy | 0.7700 |
| Precision | 0.5502 |
| Recall | 0.7326 |
| F1 Score | 0.6284 |
| ROC-AUC | 0.8464 |
| PR-AUC | 0.6581 |

Confusion matrix (test set):

```text
[[811 224]
 [100 274]]
```

The full CV comparison plus the final test row are saved to `reports/cv_results.csv`, and are also viewable live on the dashboard's **Model Performance** page.

**What actually moved the needle:** honestly, not the imbalance strategy — class weights, SMOTE, and undersampling all land within noise of each other (see [Handling class imbalance](#handling-class-imbalance)). The two changes that mattered: (1) switching the selection metric from ROC-AUC to **PR-AUC**, which reshuffled the ranking and surfaced that Logistic Regression, not XGBoost, is the stronger ranker on this feature set; and (2) **isotonic calibration**, which is what makes the dashboard's churn-probability numbers and risk bands trustworthy rather than just directionally correct. The engineered `avg_monthly_spend` / `charge_delta` features still help the linear model most (it can't derive a ratio on its own), and remain the top SHAP drivers.

### Model Explainability (SHAP)

`train.py` generates a SHAP global feature-importance plot for the selected model (`reports/shap_summary.png`), computed on the **uncalibrated** pipeline (`models/churn_model_base.joblib`) — isotonic calibration is a monotonic squash of the score and would only force a slow model-agnostic explainer without changing the attribution story. The strongest churn drivers are the engineered `avg_monthly_spend` feature, monthly charges, fiber optic internet service, tenure, and `service_count` — consistent with the notebook's t-test and chi-square hypothesis tests, and a good sign the new feature is capturing real signal rather than noise. Since the final model is a soft-voting ensemble, the dashboard's per-customer SHAP breakdown is computed as a voting-weight-averaged blend of each member model's attributions (`features.get_shap_explainer`, which also unwraps resampling-pipeline members), so explainability keeps working regardless of which candidate wins during retraining.

### Customer Segmentation

Segmentation uses `KMeans` on tenure, monthly charges, total charges, and service count. Rather than hardcoding 4 clusters, the number of clusters is chosen data-first:

- Silhouette score is computed for k = 2 through 8. It's maximized at a trivial k=2 ("everyone vs everyone else"), which isn't actionable for a business team.
- The search is restricted to a business-usable range (k = 3 to 6), and the best silhouette score within that range is used. On this dataset, **k = 3**.
- Segment names are derived programmatically from each cluster's statistics (highest churn rate → "At Risk", lowest tenure among the rest → "New", highest spend → "High Value", remainder → "Loyal", with a descriptive fallback label for any cluster beyond those four archetypes) — not assumed from cluster index, which isn't stable across retrains.

Resulting segments:

| Segment | Size | Avg Tenure | Avg Monthly Charges | Avg Total Charges | Avg Services | Churn Rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| At Risk Customers | 2617 | 15.2 | $75.79 | $1,148.78 | 2.5 | 45.8% |
| New Customers | 2245 | 27.2 | $27.15 | $721.13 | 1.1 | 15.5% |
| High Value Customers | 2181 | 58.3 | $90.25 | $5,248.17 | 4.2 | 14.9% |

The At Risk segment's churn rate (46%) is roughly 3x the other two segments — a clear, actionable signal for retention targeting. The full profile is saved to `reports/segment_profile.csv` and shown on the dashboard's Segmentation page.

## Dashboard Features

The Streamlit dashboard has five sections.

### 1. Overview

- Total number of customers, overall churn rate, average tenure, average monthly charge.
- Churn distribution visualization.

### 2. Prediction

Enter a customer's details (demographics, services, billing, contract) to get:

- Churn probability.
- Churn / stay prediction at the tuned threshold.
- Low / medium / high risk label.
- A SHAP-based explanation of which features pushed this specific prediction up or down.

### 3. Batch Prediction

Upload a CSV of customers (same schema as the Telco dataset) to score them all at once, preview the results, and download the scored CSV.

### 4. Segmentation

- View the live segment profile table (size, avg tenure/charges/services, churn rate per segment).
- Enter a customer's tenure, charges, and service count to see which segment they'd fall into.

### 5. Model Performance

- The cross-validated model comparison table (and the final held-out test row).
- The SHAP global feature-importance plot.
- A confusion matrix and ROC curve computed over the full dataset, for a quick visual sanity check (the honest generalization estimate is the CV table above, not this).

## Project Structure

```text
Customer Segmentation Project/
|-- app.py
|-- train.py
|-- features.py
|-- customer_segmentation.ipynb
|-- requirements.txt
|-- README.md
|-- WA_Fn-UseC_-Telco-Customer-Churn.csv
|-- models/
|   |-- churn_model.joblib          # isotonic-calibrated pipeline (probabilities)
|   |-- churn_model_base.joblib      # same pipeline, uncalibrated (SHAP)
|   |-- churn_threshold.joblib
|   |-- kmeans.joblib
|   |-- preprocessor.joblib
|   |-- segment_names.joblib
|   `-- shap_background.joblib
`-- reports/
    |-- cv_results.csv
    |-- segment_profile.csv
    |-- shap_summary.png
    |-- PROJECT_REPORT.md
    `-- PROJECT_REPORT.docx
```

## File Description

- `app.py`: Streamlit dashboard — overview KPIs, churn prediction (with SHAP explanations), batch prediction, segmentation, and model performance.
- `train.py`: Training script — CV comparison of imbalance strategies (cost-sensitive vs SMOTE vs undersampling), hyperparameter tuning, an ensembled candidate, isotonic probability calibration, leakage-free threshold tuning, final test evaluation (incl. Brier score), SHAP artifact generation, and data-driven segmentation. The sole script that writes to `models/` and `reports/`.
- `features.py`: Shared feature engineering (`add_engineered_features`) and SHAP explainer dispatch (`get_shap_explainer`, handles single models, the voting ensemble, and resampling-pipeline ensemble members) — imported by both `train.py` and `app.py` so the two can't drift out of sync.
- `customer_segmentation.ipynb`: Jupyter notebook for EDA, hypothesis testing (t-test, chi-square), and exploratory prototyping — the model comparison in `train.py` is the productionized version of this exploration.
- `requirements.txt`: Python dependencies (includes `imbalanced-learn` for the resampling candidates).
- `WA_Fn-UseC_-Telco-Customer-Churn.csv`: Dataset used for both the notebook and `train.py`.
- `models/churn_model.joblib`: The selected, tuned churn pipeline wrapped in isotonic calibration — used for all probability/prediction output.
- `models/churn_model_base.joblib`: The same selected pipeline without the calibration wrapper — used only to compute SHAP attributions.
- `models/churn_threshold.joblib`: Tuned decision threshold (on calibrated probabilities).
- `models/kmeans.joblib` / `models/preprocessor.joblib`: Segmentation model and its scaler.
- `models/segment_names.joblib`: Cluster ID → business-friendly segment name mapping.
- `models/shap_background.joblib`: Background sample used to build the SHAP explainer in the dashboard.
- `reports/cv_results.csv`: Cross-validated model comparison + final held-out test metrics.
- `reports/segment_profile.csv`: Per-segment statistics (size, averages, churn rate).
- `reports/shap_summary.png`: Global feature-importance plot.

## Technologies Used

- Python
- Streamlit
- Pandas / NumPy
- Scikit-learn
- imbalanced-learn
- XGBoost
- SHAP
- Matplotlib / Seaborn
- Joblib

## Installation

Clone or download the project, then open the project folder.

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the virtual environment on Windows:

```bash
.venv\Scripts\activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Model Training

To train or retrain the machine learning models, run:

```bash
python train.py
```

This runs the full cross-validated comparison of imbalance strategies, hyperparameter search, probability calibration, threshold tuning, final test evaluation, SHAP artifact generation, and segmentation, then writes:

- `models/churn_model.joblib` (calibrated), `models/churn_model_base.joblib` (uncalibrated, for SHAP), `models/churn_threshold.joblib`
- `models/kmeans.joblib`, `models/preprocessor.joblib`, `models/segment_names.joblib`
- `models/shap_background.joblib`
- `reports/cv_results.csv`, `reports/segment_profile.csv`, `reports/shap_summary.png`

## Running the Application

Start the Streamlit dashboard with:

```bash
streamlit run app.py
```

After running the command, Streamlit will open the application in the browser.

## Business Use Cases

This project can help telecom companies:

- Identify customers who are likely to leave, and understand *why* via SHAP.
- Prioritize retention campaigns for the At Risk segment (45.8% churn rate vs ~15% elsewhere).
- Score an entire customer list at once via CSV upload.
- Design targeted offers for different customer segments.
- Reduce revenue loss caused by customer churn.

## Key Learnings

This project demonstrates:

- Avoiding threshold/metric leakage in a churn-prediction pipeline (including computing `scale_pos_weight` from the training split only).
- Choosing an imbalance-aware model-selection metric (PR-AUC) instead of raw accuracy.
- Treating class imbalance as an empirical question — cost-sensitive learning, SMOTE, and undersampling compared as fair CV candidates rather than one assumed to be "the fix" — and reporting honestly that they tied.
- Calibrating predicted probabilities so a dashboard that shows a "churn probability" and risk bands is actually telling the truth (Brier score, not just AUC).
- Cross-validation, light hyperparameter tuning, and evaluating an ensemble as a fair competing candidate rather than assuming it wins.
- Grounding feature engineering in the project's own hypothesis tests instead of throwing in ungrounded features.
- Data-driven cluster count selection instead of an assumed number of segments.
- Deriving business labels from cluster statistics instead of hardcoding cluster index → name.
- Model explainability with SHAP, at both the global and per-prediction level.
- Building a multi-page Streamlit dashboard with batch scoring.

## Limitations

- SHAP explainability in the dashboard is fastest for tree-based and linear models; a non-tree/linear model (e.g. the calibrated SVM, if it were selected) would fall back to a slower generic explainer.
- The SMOTE candidates run on the one-hot-encoded matrix, so SMOTE interpolates binary dummy columns into fractional values — a common approximation rather than the categorically-correct `SMOTENC`. It's kept because it exercises the same uniform pipeline structure as every other candidate and, on this dataset, synthetic oversampling was not competitive anyway. Random undersampling (which only drops rows) has no such caveat.
- The winning ensemble combines two Logistic Regression variants, so it is close to redundant — the CV margin over a single tuned Logistic Regression is well within noise. It is kept per the project's "the ensemble competes and wins on the metric" rule; a reviewer could reasonably prefer the simpler single model.
- The "Model Performance" confusion matrix/ROC curve shown in the dashboard run over the full dataset for a quick visual check — the CV table above them is the actual honest generalization estimate.
- The project is not yet deployed to a public URL.

## Future Improvements

- Deploy the application on Streamlit Cloud, Render, or Hugging Face Spaces.
- Add downloadable PDF/HTML prediction reports.
- Track model performance over time if the underlying data is refreshed periodically.

## How to Present This Project on a Resume

Example resume description:

```text
Built an end-to-end customer churn prediction and segmentation dashboard using Python, Streamlit, Scikit-learn, imbalanced-learn, XGBoost, and SHAP. Designed a leakage-free evaluation pipeline (cross-validation, out-of-fold threshold tuning, single held-out test evaluation) selecting models by PR-AUC under class imbalance; compared cost-sensitive learning against SMOTE and random undersampling as fair CV candidates, added isotonic probability calibration (improving held-out Brier score), engineered domain-grounded features, evaluated a soft-voting ensemble as a fair competing candidate, and added SHAP-based per-prediction explainability (working across single models and ensembles), data-driven customer segmentation (silhouette-selected k, statistically-derived segment labels), and batch CSV scoring.
```

## Conclusion

This project provides a practical machine learning solution for customer churn analysis and customer segmentation. It combines leakage-free evaluation, predictive modeling, explainability, clustering, model persistence, and dashboard deployment into one complete workflow — the kind of end-to-end rigor that holds up well under interview scrutiny.
