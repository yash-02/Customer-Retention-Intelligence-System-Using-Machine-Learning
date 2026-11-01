import numpy as np
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import shap

SERVICE_COLS = [
    "PhoneService",
    "OnlineSecurity",
    "OnlineBackup",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies"
]

TREE_BASED = (DecisionTreeClassifier, RandomForestClassifier, XGBClassifier)

# Longest tenure actually observed in this dataset (months) -- used to cap the
# revenue projection below so a near-zero churn probability doesn't extrapolate
# an unrealistic infinite-horizon lifetime value.
MAX_PROJECTION_MONTHS = 72


def compute_clv_metrics(monthly_charges, total_charges, churn_probability, max_months=MAX_PROJECTION_MONTHS):
    """Expected future revenue and customer lifetime value from a churn probability.

    Treats churn_probability as a constant monthly hazard rate p, so expected
    remaining tenure follows a geometric distribution with mean 1/p. That
    remaining-tenure estimate is capped at max_months. CLV combines revenue
    already collected (total_charges) with that projected future revenue, so
    it reflects the customer's whole relationship, not just what's ahead.
    Accepts scalars or array-likes (Series/ndarray) for all three inputs.
    """
    safe_prob = np.maximum(np.asarray(churn_probability, dtype=float), 1 / max_months)
    expected_remaining_months = np.minimum(1 / safe_prob, max_months)
    expected_future_revenue = np.asarray(monthly_charges, dtype=float) * expected_remaining_months
    historic_revenue = np.nan_to_num(np.asarray(total_charges, dtype=float), nan=0.0)
    clv = historic_revenue + expected_future_revenue
    return expected_future_revenue, clv


def add_engineered_features(df):
    """Adds service_count plus three tenure/billing-derived features.

    avg_monthly_spend and charge_delta are grounded in the project's own
    hypothesis tests (tenure/MonthlyCharges/TotalCharges are all highly
    significant); is_new_customer flags the single strongest churn predictor
    (tenure) explicitly for models that don't split continuous values well.
    """
    df = df.copy()
    df["service_count"] = df[SERVICE_COLS].eq("Yes").sum(axis=1)
    df["avg_monthly_spend"] = df["TotalCharges"] / (df["tenure"] + 1)
    df["charge_delta"] = df["MonthlyCharges"] - df["avg_monthly_spend"]
    df["is_new_customer"] = (df["tenure"] <= 3).astype(int)
    return df


def get_shap_explainer(estimator, background_transformed, feature_names):
    """Returns a callable: transformed_samples -> shap.Explanation.

    Dispatches to TreeExplainer/LinearExplainer for single models (fast),
    falls back to the generic Explainer otherwise, and averages member
    explanations (weighted by voting weight) for a soft-voting ensemble so
    explainability keeps working regardless of which candidate wins.
    """
    if isinstance(estimator, VotingClassifier):
        members = [_unwrap_member(member) for member in estimator.estimators_]
        weights = estimator.weights if estimator.weights is not None else [1] * len(members)
        weights = np.array(weights, dtype=float)
        weights = weights / weights.sum()

        member_explainers = [
            _single_explainer(member, background_transformed, feature_names)
            for member in members
        ]

        def explain(samples):
            values = None
            base_values = None
            for weight, explainer in zip(weights, member_explainers):
                member_result = explainer(samples)
                member_values = weight * member_result.values
                member_base = weight * member_result.base_values
                values = member_values if values is None else values + member_values
                base_values = member_base if base_values is None else base_values + member_base
            return shap.Explanation(
                values=values,
                base_values=base_values,
                data=samples,
                feature_names=feature_names,
            )

        return explain

    single = _single_explainer(estimator, background_transformed, feature_names)
    return single


def _unwrap_member(member):
    """A soft-voting member may be a bare classifier or a (resampling) pipeline.

    Resampling only happens at fit time, so for SHAP -- which explains the
    fitted classifier on already-transformed data -- the meaningful object is
    the final `model` step either way.
    """
    if hasattr(member, "named_steps") and "model" in member.named_steps:
        return member.named_steps["model"]
    return member


def _single_explainer(estimator, background_transformed, feature_names):
    if isinstance(estimator, TREE_BASED):
        return shap.TreeExplainer(
            estimator,
            feature_perturbation="tree_path_dependent",
            feature_names=feature_names
        )
    if isinstance(estimator, LogisticRegression):
        return shap.LinearExplainer(
            estimator,
            background_transformed,
            feature_names=feature_names
        )
    return shap.Explainer(
        estimator,
        background_transformed,
        feature_names=feature_names
    )
