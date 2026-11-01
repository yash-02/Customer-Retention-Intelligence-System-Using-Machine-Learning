import pandas as pd
import numpy as np
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from pathlib import Path

from sklearn.base import clone
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate,
    cross_val_predict,
    RandomizedSearchCV,
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
)
from xgboost import XGBClassifier

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

from features import add_engineered_features, get_shap_explainer, compute_clv_metrics

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42

# PR-AUC (average precision) is the model-selection metric. With churn at ~27%
# positive, accuracy rewards "predict nobody churns" and ROC-AUC can look
# healthy while the minority-class precision/recall trade-off is poor; PR-AUC
# is the metric that actually tracks minority-class ranking quality. ROC-AUC
# and four other metrics are still reported for context.
SELECTION_METRIC = "pr_auc"

# =====================
# Load & clean
# =====================

df = pd.read_csv(BASE_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv")
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# =====================
# Feature Engineering
# =====================

df = add_engineered_features(df)

X = df.drop(
    columns=[
        "customerID",
        "Churn"
    ]
)

y = df["Churn"]

num_cols = X.select_dtypes(
    include=np.number
).columns.tolist()

cat_cols = X.select_dtypes(
    exclude=np.number
).columns.tolist()


def build_preprocessor():
    # sparse_output=False so the imbalanced-learn resamplers downstream always
    # receive a dense array (SMOTE's k-NN and RandomUnderSampler are simplest
    # and most predictable on dense input).
    return ColumnTransformer([
        (
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]),
            num_cols
        ),
        (
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            ]),
            cat_cols
        )
    ])


# =====================
# Train / test split (test set held out until the final evaluation)
# =====================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y
)

# ---------------------------------------------------------------------------
# Class imbalance is handled entirely inside cross-validation, and every part
# of it is derived from the TRAINING split only -- the held-out test set never
# informs the class ratio, the resampling, the calibration, or the threshold.
#   * scale_pos_weight is measured on y_train (previously it was measured on
#     the full y, which leaked the test set's class ratio into XGBoost).
#   * SMOTE / RandomUnderSampler live inside the imbalanced-learn pipeline, so
#     each CV fold only ever resamples its own training portion.
# ---------------------------------------------------------------------------

train_neg = int((y_train == 0).sum())
train_pos = int((y_train == 1).sum())
scale_pos_weight = train_neg / train_pos
print(
    f"Training-split class balance: {train_pos} churn / {train_neg} retain "
    f"({train_pos / (train_pos + train_neg):.1%} positive) "
    f"-> scale_pos_weight = {scale_pos_weight:.3f}"
)

# =====================
# Candidate models: cost-sensitive learning vs resampling, compared fairly
# =====================


def cost_sensitive_pipeline(classifier):
    return Pipeline([
        ("preprocessor", build_preprocessor()),
        ("model", classifier),
    ])


def resampled_pipeline(resampler, classifier):
    # imblearn Pipeline: `resampler` runs on fit() only, never on
    # predict()/transform(), so there is no train/serve skew and no leakage
    # into the validation fold.
    return ImbPipeline([
        ("preprocessor", build_preprocessor()),
        ("resampler", resampler),
        ("model", classifier),
    ])


def make_xgb(**overrides):
    params = dict(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    params.update(overrides)
    return XGBClassifier(**params)


candidate_pipelines = {
    # ---- cost-sensitive learning (re-weight the loss, keep all rows) ----
    "Logistic Regression (class weights)": cost_sensitive_pipeline(
        LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
        )
    ),
    "Decision Tree (class weights)": cost_sensitive_pipeline(
        DecisionTreeClassifier(
            max_depth=6, class_weight="balanced", random_state=RANDOM_STATE
        )
    ),
    "Random Forest (class weights)": cost_sensitive_pipeline(
        RandomForestClassifier(
            n_estimators=300, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1,
        )
    ),
    "SVM (class weights)": cost_sensitive_pipeline(
        CalibratedClassifierCV(
            estimator=SVC(
                kernel="rbf", class_weight="balanced", random_state=RANDOM_STATE
            ),
            cv=3,
        )
    ),
    "XGBoost (scale_pos_weight)": cost_sensitive_pipeline(
        make_xgb(scale_pos_weight=scale_pos_weight)
    ),
    # ---- resampling (change the training distribution) ----
    # Evaluated as fair competitors on the same CV, not assumed to help.
    "Logistic Regression (SMOTE)": resampled_pipeline(
        SMOTE(random_state=RANDOM_STATE),
        LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    ),
    "Logistic Regression (undersample)": resampled_pipeline(
        RandomUnderSampler(random_state=RANDOM_STATE),
        LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    ),
    "XGBoost (SMOTE)": resampled_pipeline(
        SMOTE(random_state=RANDOM_STATE),
        make_xgb(),
    ),
    "XGBoost (undersample)": resampled_pipeline(
        RandomUnderSampler(random_state=RANDOM_STATE),
        make_xgb(),
    ),
}

RESAMPLING_TAGS = ("SMOTE", "undersample")


def is_resampling(model_name):
    return any(tag in model_name for tag in RESAMPLING_TAGS)


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
}

cv_rows = []
cv_pipelines = {}

print(
    "\nCross-validated comparison of imbalance strategies "
    "(5-fold, training data only, ranked by PR-AUC)\n"
)
for model_name, pipeline in candidate_pipelines.items():
    cv_scores = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=-1
    )

    row = {"Model": model_name}
    for metric in scoring:
        values = cv_scores[f"test_{metric}"]
        row[f"{metric}_mean"] = values.mean()
        row[f"{metric}_std"] = values.std()
    cv_rows.append(row)
    cv_pipelines[model_name] = pipeline

    print(
        f"{model_name:37s} "
        f"PR-AUC {row['pr_auc_mean']:.4f}+/-{row['pr_auc_std']:.4f} | "
        f"ROC-AUC {row['roc_auc_mean']:.4f} | "
        f"F1 {row['f1_mean']:.4f} | "
        f"Recall {row['recall_mean']:.4f} | "
        f"Accuracy {row['accuracy_mean']:.4f}"
    )

cv_results_df = (
    pd.DataFrame(cv_rows)
    .sort_values(by=f"{SELECTION_METRIC}_mean", ascending=False)
    .reset_index(drop=True)
)


def best_in_group(results_df, names):
    subset = results_df[results_df["Model"].isin(names)]
    if subset.empty:
        return None
    top = subset.sort_values(
        f"{SELECTION_METRIC}_mean", ascending=False
    ).iloc[0]
    return top["Model"], top[f"{SELECTION_METRIC}_mean"]


cost_sensitive_best = best_in_group(
    cv_results_df,
    [n for n in candidate_pipelines if not is_resampling(n)],
)
resampling_best = best_in_group(
    cv_results_df,
    [n for n in candidate_pipelines if is_resampling(n)],
)
print(
    f"\nBest cost-sensitive candidate: {cost_sensitive_best[0]} "
    f"(PR-AUC {cost_sensitive_best[1]:.4f})"
)
print(
    f"Best resampling candidate:     {resampling_best[0]} "
    f"(PR-AUC {resampling_best[1]:.4f})"
)

# =====================
# Light hyperparameter tuning on the top 2 candidates by CV PR-AUC
# =====================

param_distributions = {
    "Logistic Regression (class weights)": {
        "model__C": [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30],
    },
    "Decision Tree (class weights)": {
        "model__max_depth": [3, 4, 5, 6, 8, 10, None],
        "model__min_samples_leaf": [1, 2, 5, 10, 20],
    },
    "Random Forest (class weights)": {
        "model__n_estimators": [200, 300, 400, 600],
        "model__max_depth": [None, 6, 10, 14],
        "model__min_samples_leaf": [1, 2, 5, 10],
    },
    "SVM (class weights)": {
        "model__estimator__C": [0.1, 0.5, 1, 2, 5],
        "model__estimator__gamma": ["scale", "auto"],
    },
    "XGBoost (scale_pos_weight)": {
        "model__n_estimators": [200, 300, 400, 600, 800],
        "model__max_depth": [3, 4, 5, 6, 7],
        "model__learning_rate": [0.01, 0.02, 0.05, 0.08, 0.1],
        "model__subsample": [0.6, 0.8, 1.0],
        "model__colsample_bytree": [0.6, 0.8, 1.0],
        "model__min_child_weight": [1, 3, 5],
        "model__reg_lambda": [0.5, 1.0, 2.0, 5.0],
    },
    "Logistic Regression (SMOTE)": {
        "model__C": [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30],
        "resampler__sampling_strategy": [0.4, 0.6, 0.8, 1.0],
    },
    "Logistic Regression (undersample)": {
        "model__C": [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30],
        "resampler__sampling_strategy": [0.5, 0.7, 0.85, 1.0],
    },
    "XGBoost (SMOTE)": {
        "model__n_estimators": [200, 300, 400, 600],
        "model__max_depth": [3, 4, 5, 6],
        "model__learning_rate": [0.02, 0.05, 0.08, 0.1],
        "model__subsample": [0.6, 0.8, 1.0],
        "model__min_child_weight": [1, 3, 5],
        "resampler__sampling_strategy": [0.4, 0.6, 0.8, 1.0],
    },
    "XGBoost (undersample)": {
        "model__n_estimators": [200, 300, 400, 600],
        "model__max_depth": [3, 4, 5, 6],
        "model__learning_rate": [0.02, 0.05, 0.08, 0.1],
        "model__subsample": [0.6, 0.8, 1.0],
        "model__min_child_weight": [1, 3, 5],
        "resampler__sampling_strategy": [0.5, 0.7, 0.85, 1.0],
    },
}


def grid_size(grid):
    size = 1
    for values in grid.values():
        size *= len(values)
    return size


top_candidates = cv_results_df.head(2)["Model"].tolist()
print(
    f"\nTuning hyperparameters for top-2 candidates by CV "
    f"{SELECTION_METRIC.upper()}: {top_candidates}"
)

tuned_pipelines = dict(cv_pipelines)

for model_name in top_candidates:
    grid = param_distributions[model_name]
    search = RandomizedSearchCV(
        cv_pipelines[model_name],
        param_distributions=grid,
        n_iter=min(25, grid_size(grid)),
        scoring="average_precision",
        cv=3,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    tuned_pipelines[model_name] = search.best_estimator_
    print(
        f"{model_name}: best CV PR-AUC after tuning = {search.best_score_:.4f} "
        f"(params: {search.best_params_})"
    )

# Re-score the tuned top candidates with the same 5-fold CV so the comparison
# table reflects post-tuning performance.
for model_name in top_candidates:
    cv_scores = cross_validate(
        tuned_pipelines[model_name],
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=-1
    )
    row_idx = cv_results_df.index[cv_results_df["Model"] == model_name][0]
    for metric in scoring:
        values = cv_scores[f"test_{metric}"]
        cv_results_df.loc[row_idx, f"{metric}_mean"] = values.mean()
        cv_results_df.loc[row_idx, f"{metric}_std"] = values.std()

# =====================
# Soft-voting ensemble of the top-2 tuned models, evaluated as an
# additional candidate (only wins if it actually beats the best single
# model on mean CV PR-AUC -- not assumed to win by default).
# =====================


def ensemble_member(tuned_pipeline):
    """Reduce a tuned candidate to something a VotingClassifier can hold behind
    one shared preprocessor: the bare classifier, or (resampler + classifier)
    so a resampling candidate keeps resampling inside the ensemble too.
    """
    steps = tuned_pipeline.named_steps
    if "resampler" in steps:
        return ImbPipeline([
            ("resampler", clone(steps["resampler"])),
            ("model", clone(steps["model"])),
        ])
    return clone(steps["model"])


ensemble_name = "Voting Ensemble (" + " + ".join(top_candidates) + ")"
ensemble_pipeline = Pipeline([
    ("preprocessor", build_preprocessor()),
    ("model", VotingClassifier(
        estimators=[
            (name, ensemble_member(tuned_pipelines[name]))
            for name in top_candidates
        ],
        voting="soft",
        n_jobs=-1,
    )),
])

ensemble_cv_scores = cross_validate(
    ensemble_pipeline,
    X_train,
    y_train,
    cv=cv,
    scoring=scoring,
    n_jobs=-1
)

ensemble_row = {"Model": ensemble_name}
for metric in scoring:
    values = ensemble_cv_scores[f"test_{metric}"]
    ensemble_row[f"{metric}_mean"] = values.mean()
    ensemble_row[f"{metric}_std"] = values.std()

cv_results_df = pd.concat(
    [cv_results_df, pd.DataFrame([ensemble_row])],
    ignore_index=True
)
tuned_pipelines[ensemble_name] = ensemble_pipeline

print(
    f"{ensemble_name}: "
    f"PR-AUC {ensemble_row['pr_auc_mean']:.4f}+/-{ensemble_row['pr_auc_std']:.4f} | "
    f"ROC-AUC {ensemble_row['roc_auc_mean']:.4f} | "
    f"F1 {ensemble_row['f1_mean']:.4f}"
)

cv_results_df = cv_results_df.sort_values(
    by=f"{SELECTION_METRIC}_mean", ascending=False
).reset_index(drop=True)

print("\nFinal cross-validated comparison (post-tuning, including ensemble)")
print(
    cv_results_df.to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}"
    )
)

# =====================
# Select final model by CV PR-AUC
# =====================

best_model_name = cv_results_df.loc[0, "Model"]
best_pipeline = tuned_pipelines[best_model_name]
print(
    f"\nSelected model: {best_model_name} "
    f"(highest mean CV {SELECTION_METRIC.upper()})"
)

# =====================
# Probability calibration
# =====================
# class_weight / scale_pos_weight / resampling all buy minority-class recall by
# distorting the predicted probabilities (they push them toward the positive
# class). That matters here because the dashboard reports a churn *probability*
# and buckets it into Low/Medium/High risk -- an uncalibrated 0.8 that really
# means 0.55 would mislabel customers. So the selected pipeline is wrapped in
# isotonic calibration, fit with its own internal CV on the training split
# only. A separate uncalibrated copy is kept for SHAP (calibration is just a
# monotonic squash of the score; explaining the calibrated wrapper would only
# force a slow model-agnostic explainer).

CALIBRATION_METHOD = "isotonic"
calibrated_model = CalibratedClassifierCV(
    clone(best_pipeline),
    method=CALIBRATION_METHOD,
    cv=3,
)

# =====================
# Leakage-free threshold tuning: out-of-fold *calibrated* probabilities on the
# training data only, thresholds swept to maximize F1.
# =====================

oof_proba = cross_val_predict(
    calibrated_model,
    X_train,
    y_train,
    cv=cv,
    method="predict_proba",
    n_jobs=-1
)[:, 1]

thresholds = np.arange(0.05, 0.96, 0.01)
threshold_scores = [
    (t, f1_score(y_train, (oof_proba >= t).astype(int)))
    for t in thresholds
]
best_threshold, best_oof_f1 = max(threshold_scores, key=lambda s: s[1])
print(
    f"Tuned threshold (maximizes OOF F1 on calibrated training probabilities): "
    f"{best_threshold:.2f} (F1={best_oof_f1:.4f})"
)

# Refit: calibrated model for probabilities/predictions, uncalibrated copy for
# SHAP.
calibrated_model.fit(X_train, y_train)
base_pipeline = clone(best_pipeline)
base_pipeline.fit(X_train, y_train)

# =====================
# Single, final evaluation on the untouched test set
# =====================

test_proba = calibrated_model.predict_proba(X_test)[:, 1]
test_pred = (test_proba >= best_threshold).astype(int)

uncalibrated_test_proba = base_pipeline.predict_proba(X_test)[:, 1]
brier_uncalibrated = brier_score_loss(y_test, uncalibrated_test_proba)
brier_calibrated = brier_score_loss(y_test, test_proba)
print(
    f"\nBrier score on held-out test (lower = better-calibrated probabilities): "
    f"{brier_uncalibrated:.4f} -> {brier_calibrated:.4f} after {CALIBRATION_METHOD} calibration"
)

final_test_metrics = {
    "Model": f"{best_model_name} (final, calibrated, held-out test)",
    "accuracy_mean": accuracy_score(y_test, test_pred),
    "accuracy_std": np.nan,
    "precision_mean": precision_score(y_test, test_pred),
    "precision_std": np.nan,
    "recall_mean": recall_score(y_test, test_pred),
    "recall_std": np.nan,
    "f1_mean": f1_score(y_test, test_pred),
    "f1_std": np.nan,
    "roc_auc_mean": roc_auc_score(y_test, test_proba),
    "roc_auc_std": np.nan,
    "pr_auc_mean": average_precision_score(y_test, test_proba),
    "pr_auc_std": np.nan,
}

cv_results_df = pd.concat(
    [cv_results_df, pd.DataFrame([final_test_metrics])],
    ignore_index=True
)

test_cm = confusion_matrix(y_test, test_pred)

print(f"\nFinal held-out test metrics for {best_model_name} @ threshold {best_threshold:.2f}")
print(f"Accuracy:  {final_test_metrics['accuracy_mean']:.4f}")
print(f"Precision: {final_test_metrics['precision_mean']:.4f}")
print(f"Recall:    {final_test_metrics['recall_mean']:.4f}")
print(f"F1:        {final_test_metrics['f1_mean']:.4f}")
print(f"ROC-AUC:   {final_test_metrics['roc_auc_mean']:.4f}")
print(f"PR-AUC:    {final_test_metrics['pr_auc_mean']:.4f}")
print("Confusion matrix (test set):")
print(test_cm)

cv_results_df.to_csv(
    REPORTS_DIR / "cv_results.csv",
    index=False
)
print("\nCross-validation + final test results saved to reports/cv_results.csv")

# =====================
# Business metrics for the full customer base: churn probability, expected
# future revenue, and customer lifetime value (CLV). Scored with the same
# X_train-fitted calibrated_model that app.py loads for every other page, so
# these numbers stay consistent with the model actually deployed -- not a
# separate refit on the full data.
# =====================

df["churn_probability"] = calibrated_model.predict_proba(X)[:, 1]
df["expected_future_revenue"], df["clv"] = compute_clv_metrics(
    monthly_charges=df["MonthlyCharges"],
    total_charges=df["TotalCharges"],
    churn_probability=df["churn_probability"],
)

joblib.dump(calibrated_model, MODELS_DIR / "churn_model.joblib")
joblib.dump(base_pipeline, MODELS_DIR / "churn_model_base.joblib")
joblib.dump(best_threshold, MODELS_DIR / "churn_threshold.joblib")

# =====================
# SHAP explainability artifacts (computed on the uncalibrated pipeline)
# =====================

preprocessor = base_pipeline.named_steps["preprocessor"]
estimator = base_pipeline.named_steps["model"]
feature_names = preprocessor.get_feature_names_out()

background_sample = X_train.sample(
    n=min(100, len(X_train)), random_state=RANDOM_STATE
)
background_transformed = preprocessor.transform(background_sample)
if hasattr(background_transformed, "toarray"):
    background_transformed = background_transformed.toarray()

explain_sample = X_test.sample(
    n=min(200, len(X_test)), random_state=RANDOM_STATE
)
explain_transformed = preprocessor.transform(explain_sample)
if hasattr(explain_transformed, "toarray"):
    explain_transformed = explain_transformed.toarray()

try:
    explain = get_shap_explainer(estimator, background_transformed, feature_names)
    shap_values = explain(explain_transformed)

    plt.figure()
    shap.plots.bar(shap_values, show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "shap_summary.png", dpi=150)
    plt.close()
    print("SHAP global feature-importance plot saved to reports/shap_summary.png")

    joblib.dump(background_sample, MODELS_DIR / "shap_background.joblib")
except Exception as exc:
    print(f"SHAP explainability skipped ({exc})")

# =====================
# Segmentation: benchmark multiple unsupervised algorithms and let the data
# pick the winner, instead of assuming KMeans is the right fit. Silhouette
# score is the selection metric (same rule the original KMeans-only version
# used, kept consistent so the "business range k=3-6" logic still applies);
# Davies-Bouldin and Calinski-Harabasz are reported alongside for context,
# since silhouette alone can be misleading for non-convex or unequal-density
# clusters.
# =====================

segment_cols = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "service_count"
]

seg_df = df[segment_cols].copy()
seg_df = seg_df.fillna(seg_df.median())

seg_scaler = StandardScaler()
seg_scaled = seg_scaler.fit_transform(seg_df)

k_range = range(3, 7)

# Ward linkage is computed once and cut at each k with fcluster, instead of
# refitting AgglomerativeClustering per k -- the merge tree doesn't change
# with k, only where it gets cut, so this avoids redoing the expensive part
# four times.
ward_linkage = linkage(seg_scaled, method="ward")

labels_by_candidate = {}
for k in k_range:
    labels_by_candidate[("KMeans", k)] = KMeans(
        n_clusters=k, random_state=RANDOM_STATE, n_init=10
    ).fit_predict(seg_scaled)
    labels_by_candidate[("Gaussian Mixture", k)] = GaussianMixture(
        n_components=k, random_state=RANDOM_STATE, n_init=5
    ).fit_predict(seg_scaled)
    labels_by_candidate[("Agglomerative (Ward)", k)] = (
        fcluster(ward_linkage, t=k, criterion="maxclust") - 1
    )

segmentation_rows = []
for (algorithm_name, k), labels in labels_by_candidate.items():
    sil = silhouette_score(seg_scaled, labels)
    db = davies_bouldin_score(seg_scaled, labels)
    ch = calinski_harabasz_score(seg_scaled, labels)
    segmentation_rows.append({
        "algorithm": algorithm_name,
        "k": k,
        "silhouette": sil,
        "davies_bouldin": db,
        "calinski_harabasz": ch,
    })
    print(
        f"{algorithm_name:22s} k={k}: silhouette={sil:.4f}  "
        f"davies_bouldin={db:.4f} (lower better)  "
        f"calinski_harabasz={ch:.1f} (higher better)"
    )

segmentation_results_df = pd.DataFrame(segmentation_rows).sort_values(
    "silhouette", ascending=False
).reset_index(drop=True)

print(
    "\nUnsupervised algorithm comparison "
    "(business range k=3-6, ranked by silhouette):"
)
print(
    segmentation_results_df.to_string(
        index=False, float_format=lambda v: f"{v:.4f}"
    )
)
segmentation_results_df.to_csv(
    REPORTS_DIR / "segmentation_algorithm_comparison.csv", index=False
)

best_row = segmentation_results_df.iloc[0]
best_algorithm, best_k = best_row["algorithm"], int(best_row["k"])
print(
    f"\nSelected segmentation algorithm: {best_algorithm} "
    f"(k={best_k}, silhouette={best_row['silhouette']:.4f})"
)

cluster_labels = labels_by_candidate[(best_algorithm, best_k)]

profile = seg_df.copy()
profile["cluster"] = cluster_labels
profile["churn"] = df["Churn"].values
# Not fed into KMeans itself (segment_cols above stays as-is): these are
# derived from tenure/MonthlyCharges/TotalCharges, which already drive the
# clustering, so adding them as inputs too would just double-count that
# signal. They're joined in here purely for per-segment reporting.
profile["churn_probability"] = df["churn_probability"].values
profile["expected_future_revenue"] = df["expected_future_revenue"].values
profile["clv"] = df["clv"].values

cluster_profile = (
    profile.groupby("cluster")
    .agg(
        size=("cluster", "size"),
        avg_tenure=("tenure", "mean"),
        avg_monthly_charges=("MonthlyCharges", "mean"),
        avg_total_charges=("TotalCharges", "mean"),
        avg_service_count=("service_count", "mean"),
        churn_rate=("churn", "mean"),
        avg_churn_probability=("churn_probability", "mean"),
        avg_expected_future_revenue=("expected_future_revenue", "mean"),
        avg_clv=("clv", "mean"),
    )
    .reset_index()
)


def label_segments(cluster_profile_df):
    """Assign business-friendly names to clusters from their statistics.

    Highest churn rate -> At Risk, then lowest tenure -> New, then highest
    spend -> High Value, remainder -> Loyal. Any extra clusters beyond those
    four archetypes get a descriptive fallback label from their standout
    z-scored stat.
    """
    remaining = cluster_profile_df.copy()
    names = {}

    at_risk = remaining.sort_values("churn_rate", ascending=False).iloc[0]
    names[at_risk["cluster"]] = "At Risk Customers"
    remaining = remaining[remaining["cluster"] != at_risk["cluster"]]

    if len(remaining) > 0:
        new_cust = remaining.sort_values("avg_tenure", ascending=True).iloc[0]
        names[new_cust["cluster"]] = "New Customers"
        remaining = remaining[remaining["cluster"] != new_cust["cluster"]]

    if len(remaining) > 0:
        high_value = remaining.sort_values(
            "avg_total_charges", ascending=False
        ).iloc[0]
        names[high_value["cluster"]] = "High Value Customers"
        remaining = remaining[remaining["cluster"] != high_value["cluster"]]

    if len(remaining) > 0:
        loyal = remaining.sort_values("avg_tenure", ascending=False).iloc[0]
        names[loyal["cluster"]] = "Loyal Customers"
        remaining = remaining[remaining["cluster"] != loyal["cluster"]]

    stat_cols = [
        "avg_tenure", "avg_monthly_charges", "avg_total_charges", "avg_service_count"
    ]
    zscored = (
        cluster_profile_df.set_index("cluster")[stat_cols]
        - cluster_profile_df[stat_cols].mean()
    ) / cluster_profile_df[stat_cols].std()

    for _, row in remaining.iterrows():
        cid = row["cluster"]
        standout_col = zscored.loc[cid].abs().idxmax()
        direction = "High" if zscored.loc[cid, standout_col] > 0 else "Low"
        readable = standout_col.replace("avg_", "").replace("_", " ").title()
        names[cid] = f"{direction} {readable} Segment"

    return names


segment_names = label_segments(cluster_profile)
cluster_profile["segment_name"] = cluster_profile["cluster"].map(segment_names)

print("\nSegment profile:")
print(
    cluster_profile.to_string(
        index=False,
        float_format=lambda value: f"{value:.2f}"
    )
)

cluster_profile.to_csv(REPORTS_DIR / "segment_profile.csv", index=False)

customer_segments = df[[
    "customerID", "tenure", "MonthlyCharges", "TotalCharges",
    "churn_probability", "expected_future_revenue", "clv",
]].copy()
customer_segments["cluster"] = cluster_labels
customer_segments["segment_name"] = customer_segments["cluster"].map(segment_names)
customer_segments.to_csv(REPORTS_DIR / "customer_segments.csv", index=False)
print("Per-customer segment assignments saved to reports/customer_segments.csv")

# Cluster centroids (mean of each cluster in scaled space), computed the same
# way regardless of which algorithm won. KMeans and Gaussian Mixture can
# score a brand-new customer directly, but Agglomerative Clustering is
# transductive -- it has no predict() for points outside the training set --
# so app.py assigns new customers to their nearest centroid instead, which
# works uniformly no matter which algorithm was selected.
seg_scaled_df = pd.DataFrame(seg_scaled, columns=segment_cols)
seg_scaled_df["cluster"] = cluster_labels
segment_centroids = seg_scaled_df.groupby("cluster")[segment_cols].mean()

segmentation_meta = {
    "algorithm": best_algorithm,
    "k": best_k,
    "silhouette": float(best_row["silhouette"]),
}

joblib.dump(seg_scaler, MODELS_DIR / "preprocessor.joblib")
joblib.dump(segment_centroids, MODELS_DIR / "segment_centroids.joblib")
joblib.dump(segmentation_meta, MODELS_DIR / "segmentation_meta.joblib")
joblib.dump(segment_names, MODELS_DIR / "segment_names.joblib")

# =====================
# Repo hygiene: drop orphaned artifacts from earlier notebook experiments
# that app.py never loads.
# =====================

for stale_file in [
    "customer_segmentation_kmeans.joblib",
    "segmentation_preprocessor.joblib",
    "kmeans.joblib",
]:
    stale_path = MODELS_DIR / stale_file
    if stale_path.exists():
        stale_path.unlink()
        print(f"Removed stale artifact: models/{stale_file}")

print("\nModels saved successfully")
