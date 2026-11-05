"""
Task 1 — CLASSIFICATION
Dataset : Breast Cancer Wisconsin (sklearn.datasets.load_breast_cancer)
Model   : RandomForestClassifier (classic ML, no deep learning)
Run     : streamlit run classification_app.py --server.port 8501 --server.address 0.0.0.0
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)
PORT = 8501
SCALERS = {"StandardScaler": StandardScaler, "MinMaxScaler": MinMaxScaler, "RobustScaler": RobustScaler}
st.set_page_config(page_title="Classification — Breast Cancer", layout="wide")


@st.cache_data
def load_data():
    data = load_breast_cancer(as_frame=True)
    df = data.frame.copy()
    df["target_name"] = df["target"].map(dict(enumerate(data.target_names)))
    return df, list(data.feature_names), list(data.target_names)


@st.cache_resource
def train_model(X_train, y_train, n_estimators, max_depth, scaler_name):
    pipe = Pipeline(steps=[
        ("scaler", SCALERS[scaler_name]()),
        ("classifier", RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth, random_state=42
        )),
    ])
    pipe.fit(X_train, y_train)
    return pipe


df, feature_names, target_names = load_data()
X = df[feature_names]
y = df["target"]
mean_feats = [f for f in feature_names if f.startswith("mean")]

st.title("🧬 Breast Cancer Classification")
st.caption(f"Running on port {PORT} — malignant vs. benign tumor classification, RandomForestClassifier")
st.caption("Every control below — EDA filters, the preprocessing scaler, the model hyperparameters — "
           "re-runs the pipeline live. Nothing on this page is a frozen screenshot.")

tab_eda, tab_prep, tab_model, tab_predict = st.tabs(
    ["📊 EDA", "🧹 Preprocessing", "🤖 Model & Evaluation", "🔮 Live Prediction"]
)

# ---------------------------------------------------------------- EDA ----
with tab_eda:
    st.header("Exploratory Data Analysis")

    diagnosis_filter = st.multiselect(
        "Filter rows by diagnosis (affects every chart below)",
        target_names, default=target_names,
    )
    df_eda = df[df["target_name"].isin(diagnosis_filter)]

    c1, c2 = st.columns(2)
    c1.metric("Rows (after filter)", df_eda.shape[0])
    c2.metric("Features", len(feature_names))

    st.subheader("Sample rows")
    st.dataframe(df_eda.head(10))

    st.subheader("Class balance")
    fig, ax = plt.subplots(figsize=(4, 3))
    df_eda["target_name"].value_counts().plot(kind="bar", ax=ax, color=["#d62728", "#2ca02c"])
    ax.set_xlabel("Diagnosis")
    ax.set_ylabel("Count")
    st.pyplot(fig)

    st.subheader("Missing values")
    missing = df_eda[feature_names].isna().sum()
    st.write(f"Total missing values across all features: **{int(missing.sum())}**")

    st.subheader("Summary statistics")
    st.dataframe(df_eda[feature_names].describe().T)

    st.subheader("Feature distribution by diagnosis")
    col_a, col_b = st.columns(2)
    feat = col_a.selectbox("Choose a feature", feature_names, index=feature_names.index("mean radius"))
    bins = col_b.slider("Histogram bins", 10, 60, 25, 5)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for cls in df_eda["target_name"].unique():
        subset = df_eda[df_eda["target_name"] == cls][feat]
        ax.hist(subset, bins=bins, alpha=0.5, label=cls, density=True)
    ax.set_xlabel(feat)
    ax.legend()
    st.pyplot(fig)

    st.subheader("Correlation heatmap")
    corr_feats = st.multiselect("Features to correlate", feature_names, default=mean_feats)
    if len(corr_feats) >= 2:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(df_eda[corr_feats].corr(), annot=len(corr_feats) <= 8, fmt=".2f", cmap="coolwarm", ax=ax)
        st.pyplot(fig)
    else:
        st.info("Pick at least 2 features to draw a correlation heatmap.")

# --------------------------------------------------------- Preprocessing --
with tab_prep:
    st.header("Preprocessing")
    st.markdown("""
    1. **Train/test split** — stratified on the target so both classes stay balanced in each split.
    2. **Scaling** — pick a scaler below; it feeds straight into the model pipeline used in the
       next tab, so switching it changes the trained model and every metric/chart downstream.
    """)
    col1, col2 = st.columns(2)
    test_size = col1.slider("Test set size", 0.1, 0.4, 0.2, 0.05)
    scaler_name = col2.selectbox("Scaler", list(SCALERS.keys()), index=0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    st.write(f"Train shape: {X_train.shape} — Test shape: {X_test.shape}")

    st.subheader("Before vs. after scaling")
    preview_feat = st.selectbox("Preview feature", feature_names, index=feature_names.index("mean radius"))
    scaler_preview = SCALERS[scaler_name]().fit(X_train)
    idx = feature_names.index(preview_feat)
    before = X_train[preview_feat]
    after = scaler_preview.transform(X_train)[:, idx]
    comp = pd.DataFrame({"before": before.values, "after": after})
    st.dataframe(comp.describe().T)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    axes[0].hist(comp["before"], bins=30, color="#1f77b4")
    axes[0].set_title(f"{preview_feat} — before {scaler_name}")
    axes[1].hist(comp["after"], bins=30, color="#ff7f0e")
    axes[1].set_title(f"{preview_feat} — after {scaler_name}")
    st.pyplot(fig)

# --------------------------------------------------------------- Model ---
with tab_model:
    st.header("Model Training & Evaluation")
    st.caption(f"Using scaler **{scaler_name}** and test size **{test_size}** from the Preprocessing tab.")
    col1, col2 = st.columns(2)
    n_estimators = col1.slider("n_estimators", 10, 300, 100, 10)
    max_depth = col2.slider("max_depth", 2, 20, 6, 1)

    model = train_model(X_train, y_train, n_estimators, max_depth, scaler_name)
    y_pred = model.predict(X_test)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.3f}")
    m2.metric("Precision", f"{precision_score(y_test, y_pred):.3f}")
    m3.metric("Recall", f"{recall_score(y_test, y_pred):.3f}")
    m4.metric("F1", f"{f1_score(y_test, y_pred):.3f}")

    st.subheader("Confusion matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(4, 3.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=target_names, yticklabels=target_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    st.subheader("Classification report")
    st.text(classification_report(y_test, y_pred, target_names=target_names))

    st.subheader("Feature importance (top 10)")
    importances = pd.Series(
        model.named_steps["classifier"].feature_importances_, index=feature_names
    ).sort_values(ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(6, 4))
    importances.sort_values().plot(kind="barh", ax=ax, color="#1f77b4")
    st.pyplot(fig)

# ------------------------------------------------------------- Predict ---
with tab_predict:
    st.header("Live Prediction")
    st.caption("Adjust the key tumor measurements and classify in real time, using the model "
               "trained with the current Preprocessing/Model tab settings.")
    #cols = st.columns(3) is a Streamlit layout feature that splits the
    #main app interface horizontally into 3 columns of equal width.
    cols = st.columns(3)
    inputs = {}
    for i, feat in enumerate(mean_feats):
        default = float(X[feat].mean())
        lo, hi = float(X[feat].min()), float(X[feat].max())
        inputs[feat] = cols[i % 3].slider(feat, lo, hi, default)

    if st.button("Predict diagnosis"):
        row = X.mean().copy()
        for feat, val in inputs.items():
            row[feat] = val
        row_df = pd.DataFrame([row])[feature_names]
        pred = model.predict(row_df)[0]
        proba = model.predict_proba(row_df)[0]
        st.success(f"Predicted diagnosis: **{target_names[pred]}**")
        st.bar_chart(pd.Series(proba, index=target_names))
