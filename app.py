import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix, roc_curve, auc

from features import add_engineered_features, get_shap_explainer, compute_clv_metrics

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

st.set_page_config(
    page_title="Customer Churn Dashboard",
    layout="wide"
)

st.title(
    "📊 Customer Churn Intelligence Dashboard"
)

model = joblib.load(
    MODELS_DIR / "churn_model.joblib"
)

threshold_path = MODELS_DIR / "churn_threshold.joblib"
churn_threshold = (
    joblib.load(threshold_path)
    if threshold_path.exists()
    else 0.5
)

segment_centroids = joblib.load(
    MODELS_DIR / "segment_centroids.joblib"
)

segmentation_meta_path = MODELS_DIR / "segmentation_meta.joblib"
segmentation_meta = (
    joblib.load(segmentation_meta_path)
    if segmentation_meta_path.exists()
    else {}
)

seg_scaler = joblib.load(
    MODELS_DIR / "preprocessor.joblib"
)

segment_names_path = MODELS_DIR / "segment_names.joblib"
segment_names = (
    joblib.load(segment_names_path)
    if segment_names_path.exists()
    else {}
)

segment_profile_path = REPORTS_DIR / "segment_profile.csv"
segment_profile = (
    pd.read_csv(segment_profile_path)
    if segment_profile_path.exists()
    else None
)

df = pd.read_csv(
    BASE_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

df["Churn"] = df["Churn"].map({
    "Yes":1,
    "No":0
})

def category_options(column):
    return sorted(df[column].dropna().unique().tolist())


@st.cache_resource
def build_shap_explainer():
    # `churn_model.joblib` is the isotonic-calibrated model used for
    # probabilities; SHAP runs on the uncalibrated pipeline saved alongside it
    # (calibration is a monotonic squash of the score and only adds a slow
    # model-agnostic explainer path).
    base_model_path = MODELS_DIR / "churn_model_base.joblib"
    if not base_model_path.exists():
        return None, None, None
    base_model = joblib.load(base_model_path)

    preprocessor = base_model.named_steps["preprocessor"]
    estimator = base_model.named_steps["model"]
    feature_names = preprocessor.get_feature_names_out()

    background_path = MODELS_DIR / "shap_background.joblib"
    if not background_path.exists():
        return None, None, None

    background_sample = joblib.load(background_path)
    background_transformed = preprocessor.transform(background_sample)
    if hasattr(background_transformed, "toarray"):
        background_transformed = background_transformed.toarray()

    try:
        explain = get_shap_explainer(estimator, background_transformed, feature_names)
        return explain, preprocessor, feature_names
    except Exception:
        return None, None, None


page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Prediction",
        "Batch Prediction",
        "Segmentation",
        "Model Performance"
    ]
)

if page == "Overview":

    st.subheader(
        "Business KPIs"
    )

    c1,c2,c3,c4 = st.columns(4)

    c1.metric(
        "Customers",
        len(df)
    )

    c2.metric(
        "Churn Rate",
        f"{df['Churn'].mean()*100:.2f}%"
    )

    c3.metric(
        "Avg Tenure",
        f"{df['tenure'].mean():.1f}"
    )

    c4.metric(
        "Avg Monthly Charge",
        f"${df['MonthlyCharges'].mean():.2f}"
    )

    fig,ax = plt.subplots()

    sns.countplot(
        x="Churn",
        data=df,
        ax=ax
    )

    st.pyplot(fig)

elif page == "Prediction":

    st.header(
        "Predict Customer Churn"
    )

    gender = st.selectbox(
        "Gender",
        ["Male","Female"]
    )

    senior = st.selectbox(
        "Senior Citizen",
        [0,1]
    )

    partner = st.selectbox(
        "Partner",
        ["Yes","No"]
    )

    dependents = st.selectbox(
        "Dependents",
        ["Yes","No"]
    )

    tenure = st.slider(
        "Tenure",
        0,
        72,
        12
    )

    monthly = st.slider(
        "Monthly Charges",
        18,
        120,
        50
    )

    total = st.number_input(
        "Total Charges",
        min_value=0.0,
        value=float(tenure * monthly)
    )

    phone_service = st.selectbox(
        "Phone Service",
        category_options("PhoneService")
    )

    multiple_lines = st.selectbox(
        "Multiple Lines",
        category_options("MultipleLines")
    )

    internet_service = st.selectbox(
        "Internet Service",
        category_options("InternetService")
    )

    online_security = st.selectbox(
        "Online Security",
        category_options("OnlineSecurity")
    )

    online_backup = st.selectbox(
        "Online Backup",
        category_options("OnlineBackup")
    )

    device_protection = st.selectbox(
        "Device Protection",
        category_options("DeviceProtection")
    )

    tech_support = st.selectbox(
        "Tech Support",
        category_options("TechSupport")
    )

    streaming_tv = st.selectbox(
        "Streaming TV",
        category_options("StreamingTV")
    )

    streaming_movies = st.selectbox(
        "Streaming Movies",
        category_options("StreamingMovies")
    )

    contract = st.selectbox(
        "Contract",
        [
            "Month-to-month",
            "One year",
            "Two year"
        ]
    )

    paperless_billing = st.selectbox(
        "Paperless Billing",
        ["Yes","No"]
    )

    payment_method = st.selectbox(
        "Payment Method",
        category_options("PaymentMethod")
    )

    sample = pd.DataFrame({
        "gender":[gender],
        "SeniorCitizen":[senior],
        "Partner":[partner],
        "Dependents":[dependents],
        "tenure":[tenure],
        "PhoneService":[phone_service],
        "MultipleLines":[multiple_lines],
        "InternetService":[internet_service],
        "OnlineSecurity":[online_security],
        "OnlineBackup":[online_backup],
        "DeviceProtection":[device_protection],
        "TechSupport":[tech_support],
        "StreamingTV":[streaming_tv],
        "StreamingMovies":[streaming_movies],
        "Contract":[contract],
        "PaperlessBilling":[paperless_billing],
        "PaymentMethod":[payment_method],
        "MonthlyCharges":[monthly],
        "TotalCharges":[total],
    })
    sample = add_engineered_features(sample)

    if st.button(
        "Predict"
    ):

        prob = model.predict_proba(
            sample
        )[0][1]

        expected_future_revenue, clv = compute_clv_metrics(
            monthly_charges=monthly,
            total_charges=total,
            churn_probability=prob,
        )

        m1, m2, m3 = st.columns(3)
        m1.metric(
            "Churn Probability",
            f"{prob*100:.2f}%"
        )
        m2.metric(
            "Expected Future Revenue",
            f"${float(expected_future_revenue):,.2f}"
        )
        m3.metric(
            "Customer Lifetime Value",
            f"${float(clv):,.2f}"
        )

        if prob >= churn_threshold:
            st.error(
                "Prediction: Customer is likely to churn"
            )

        else:
            st.success(
                "Prediction: Customer is likely to stay"
            )

        if prob < 0.3:
            st.success(
                "Low Risk"
            )

        elif prob < 0.7:
            st.warning(
                "Medium Risk"
            )

        else:
            st.error(
                "High Risk"
            )

        explainer, preprocessor, feature_names = build_shap_explainer()

        if explainer is not None:
            st.subheader("Why this prediction?")
            try:
                transformed = preprocessor.transform(sample)
                if hasattr(transformed, "toarray"):
                    transformed = transformed.toarray()

                shap_values = explainer(transformed)
                values = shap_values.values[0]

                contributions = (
                    pd.Series(values, index=feature_names)
                    .sort_values(key=np.abs, ascending=False)
                    .head(8)
                    .sort_values()
                )

                fig, ax = plt.subplots(figsize=(6, 4))
                colors = ["#d62728" if v > 0 else "#2ca02c" for v in contributions]
                ax.barh(contributions.index, contributions.values, color=colors)
                ax.set_xlabel("Impact on churn probability (SHAP value)")
                ax.axvline(0, color="black", linewidth=0.8)
                st.pyplot(fig)
                st.caption(
                    "Red bars push this customer's churn probability up, "
                    "green bars push it down."
                )
            except Exception:
                st.info("Explainability isn't available for this model type.")
        else:
            st.info("Explainability isn't available for this model type.")

elif page == "Batch Prediction":

    st.header("Batch Churn Prediction")
    st.write(
        "Upload a CSV with the same columns as the Telco Customer Churn "
        "dataset (customerID and Churn columns are optional/ignored) to "
        "score many customers at once."
    )

    uploaded_file = st.file_uploader("Upload customer CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            batch_features = batch_df.drop(
                columns=["customerID", "Churn"], errors="ignore"
            )
            batch_features["TotalCharges"] = pd.to_numeric(
                batch_features["TotalCharges"], errors="coerce"
            )
            batch_features = add_engineered_features(batch_features)

            probabilities = model.predict_proba(batch_features)[:, 1]
            predictions = (probabilities >= churn_threshold).astype(int)
            expected_future_revenue, clv = compute_clv_metrics(
                monthly_charges=batch_features["MonthlyCharges"],
                total_charges=batch_features["TotalCharges"],
                churn_probability=probabilities,
            )

            results = batch_df.copy()
            results["churn_probability"] = probabilities
            results["predicted_churn"] = np.where(
                predictions == 1, "Yes", "No"
            )
            results["risk_level"] = pd.cut(
                probabilities,
                bins=[-0.01, 0.3, 0.7, 1.01],
                labels=["Low", "Medium", "High"]
            )
            results["expected_future_revenue"] = expected_future_revenue
            results["clv"] = clv

            st.success(f"Scored {len(results)} customers.")
            st.dataframe(results.head(50))

            st.download_button(
                "Download scored CSV",
                data=results.to_csv(index=False).encode("utf-8"),
                file_name="churn_predictions.csv",
                mime="text/csv"
            )
        except Exception as exc:
            st.error(f"Could not score this file: {exc}")

elif page == "Segmentation":

    st.header("Customer Segmentation")

    if segmentation_meta:
        st.caption(
            f"Segments generated by **{segmentation_meta.get('algorithm')}** "
            f"(k={segmentation_meta.get('k')}), chosen from KMeans, Gaussian "
            f"Mixture, and Agglomerative (Ward) by silhouette score "
            f"({segmentation_meta.get('silhouette', 0):.3f})."
        )

    if segment_profile is not None:
        st.subheader("Segment Profile (from training data)")
        display_profile = segment_profile.copy()
        display_profile["churn_rate"] = (
            display_profile["churn_rate"] * 100
        ).round(1).astype(str) + "%"
        display_columns = [
            "segment_name", "size", "avg_tenure", "avg_monthly_charges",
            "avg_total_charges", "avg_service_count", "churn_rate",
        ]
        if "avg_churn_probability" in display_profile.columns:
            display_profile["avg_churn_probability"] = (
                display_profile["avg_churn_probability"] * 100
            ).round(1).astype(str) + "%"
            display_columns += [
                "avg_churn_probability", "avg_expected_future_revenue", "avg_clv",
            ]
        st.dataframe(display_profile[display_columns])

    st.subheader("Find a Customer's Segment")

    tenure = st.slider(
        "Tenure",
        0,
        72,
        12
    )

    monthly = st.slider(
        "Monthly Charges",
        18,
        120,
        50
    )

    total = st.number_input(
        "Total Charges",
        value=1000.0
    )

    service_count = st.slider(
        "Services Used",
        0,
        6,
        2
    )
    seg = pd.DataFrame({
        "tenure":[tenure],
        "MonthlyCharges":[monthly],
        "TotalCharges":[total],
        "service_count":[service_count]
    })

    scaled = seg_scaler.transform(seg)[0]

    # Nearest-centroid assignment, not a model .predict(): Agglomerative
    # Clustering (a possible winner of the algorithm comparison in train.py)
    # has no way to score points outside its training set, so new customers
    # are matched to whichever segment's centroid they're closest to -- the
    # same rule works regardless of which algorithm actually won.
    distances = np.linalg.norm(segment_centroids.values - scaled, axis=1)
    cluster = segment_centroids.index[np.argmin(distances)]

    segment_label = segment_names.get(cluster, segment_names.get(str(cluster), f"Segment {cluster}"))

    st.metric(
        "Customer Segment",
        segment_label
    )

elif page == "Model Performance":

    st.header("Model Performance")

    cv_results_path = REPORTS_DIR / "cv_results.csv"
    if cv_results_path.exists():
        st.subheader("Cross-Validated Model Comparison")
        st.write(
            "5-fold cross-validation on the training split only, followed "
            "by hyperparameter tuning of the top 2 candidates and a single "
            "held-out test evaluation for the selected model."
        )
        st.dataframe(pd.read_csv(cv_results_path))
    else:
        st.info("Run train.py to generate the model comparison report.")

    shap_summary_path = REPORTS_DIR / "shap_summary.png"
    if shap_summary_path.exists():
        st.subheader("Global Feature Importance (SHAP)")
        st.image(str(shap_summary_path))

    st.subheader("Held-Out Test Diagnostics")
    id_cols = ["customerID"] if "customerID" in df.columns else []
    eval_df = df.dropna(subset=["Churn"]).copy()
    eval_features = eval_df.drop(columns=["Churn"] + id_cols, errors="ignore")
    eval_features = add_engineered_features(eval_features)
    y_true = eval_df["Churn"]

    probabilities = model.predict_proba(eval_features)[:, 1]
    predictions = (probabilities >= churn_threshold).astype(int)

    col1, col2 = st.columns(2)

    with col1:
        cm = confusion_matrix(y_true, predictions)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix (full dataset)")
        st.pyplot(fig)

    with col2:
        fpr, tpr, _ = roc_curve(y_true, probabilities)
        roc_auc = auc(fpr, tpr)
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve (full dataset)")
        ax.legend()
        st.pyplot(fig)

    st.caption(
        "These diagnostics run on the full dataset for a quick visual "
        "check; see the cross-validated table above and reports/cv_results.csv "
        "for the honest held-out generalization estimate."
    )
