import joblib
import numpy as np
import streamlit as st

st.set_page_config(page_title="Iris Predictor", page_icon="🌸", layout="centered")


@st.cache_resource
def load_model():
    return joblib.load("iris_pipeline.joblib")


model = load_model()
CLASS_NAMES = ["setosa", "versicolor", "virginica"]

st.title("🌸 Iris Species Predictor")
st.caption("Loads iris_pipeline.joblib directly — no API yet.")

if "history" not in st.session_state:
    st.session_state.history = []

#syntax st.slider("title", min, max, default)
sepal_length = st.slider("Sepal length (cm)", 4.0, 8.0, 5.1)
sepal_width = st.slider("Sepal width (cm)", 2.0, 4.5, 3.5)
petal_length = st.slider("Petal length (cm)", 1.0, 7.0, 1.4)
petal_width = st.slider("Petal width (cm)", 0.1, 2.5, 0.2)

if st.button("Predict species"):
    row = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    pred_class = int(model.predict(row)[0])
    pred_proba = model.predict_proba(row)[0]

    st.success(f"Predicted species: **{CLASS_NAMES[pred_class]}**")
    st.bar_chart(dict(zip(CLASS_NAMES, pred_proba)))

    st.session_state.history.append(CLASS_NAMES[pred_class])

if st.session_state.history:
    st.caption(f"Predictions this session: {', '.join(st.session_state.history)}")
