# app.py — Streamlit app to serve your SVR model
import streamlit as st
import numpy as np
import pandas as pd
import pickle, json, os

# -------- Configuration --------
ARTIFACTS_DIR = "deploy_artifacts"
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "best_model_final.pkl")
SCALER_PATH = os.path.join(ARTIFACTS_DIR, "scaler.pkl")
FEATURE_ORDER_PATH = os.path.join(ARTIFACTS_DIR, "feature_order.json")
STATS_PATH = os.path.join(ARTIFACTS_DIR, "training_stats.json")

st.set_page_config(page_title="Cases per 100k Predictor", layout="centered")
st.title("Cases per 100,000 Predictor — Demo")

st.markdown(
    "Enter the county attributes (same features used for training). "
    "The app returns predicted Cases per 100k and a simple local explanation."
)

# -------- Load artifacts --------
@st.cache_resource
def load_artifacts():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Place model pickle in repo under {ARTIFACTS_DIR}/")
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(f"Scaler not found at {SCALER_PATH}. Place scaler pickle in repo under {ARTIFACTS_DIR}/")
    if not os.path.exists(FEATURE_ORDER_PATH):
        raise FileNotFoundError(f"Feature order not found at {FEATURE_ORDER_PATH}. Place feature_order.json in repo under {ARTIFACTS_DIR}/")
    model = pickle.load(open(MODEL_PATH, "rb"))
    scaler = pickle.load(open(SCALER_PATH, "rb"))
    feature_order = json.load(open(FEATURE_ORDER_PATH, "r"))
    # training stats optional
    training_stats = {}
    if os.path.exists(STATS_PATH):
        training_stats = json.load(open(STATS_PATH, "r"))
    else:
        # default naive stats if file not present (0..1)
        training_stats = {f: {"min": 0.0, "max": 1.0, "mean": 0.5} for f in feature_order}
    return model, scaler, feature_order, training_stats

model, scaler, feature_order, training_stats = load_artifacts()

st.sidebar.header("Input features")
st.sidebar.markdown("Set feature values and press **Predict**")

# Build inputs
input_dict = {}
for feat in feature_order:
    s = training_stats.get(feat, {"min": 0.0, "max": 1.0, "mean": 0.5})
    lo, hi, mu = float(s.get("min", 0.0)), float(s.get("max", 1.0)), float(s.get("mean", 0.5))
    step = (hi - lo) / 100 if hi > lo else 0.01
    input_dict[feat] = st.sidebar.slider(feat, min_value=lo, max_value=hi, value=mu, step=step)

if st.sidebar.button("Predict"):
    X_input = pd.DataFrame([input_dict], columns=feature_order)
    # Scale using saved scaler (scaler expects same feature order)
    X_scaled = scaler.transform(X_input)
    pred = model.predict(X_scaled)[0]

    st.markdown("## Prediction")
    st.metric("Predicted Cases per 100,000", f"{pred:.2f}")

    # Quick local explanation: leave-one-out mean replacement
    st.markdown("### Local feature impacts (mean-replacement)")
    contributions = []
    for feat in feature_order:
        # create copy in original space, replace this feat with training mean
        X_repl = X_input.copy()
        X_repl.loc[0, feat] = training_stats.get(feat, {}).get("mean", X_repl.loc[0, feat])
        X_repl_scaled = scaler.transform(X_repl)
        pred_repl = model.predict(X_repl_scaled)[0]
        contributions.append((feat, pred - pred_repl))

    contributions = sorted(contributions, key=lambda x: abs(x[1]), reverse=True)
    contrib_df = pd.DataFrame(contributions, columns=["feature", "delta_pred"]).head(10)
    st.dataframe(contrib_df.style.format({"delta_pred": "{:.2f}"}))

    st.markdown("**Interpretation:** `delta_pred` shows how much the prediction changed when that feature was set to its training mean (positive means the current value increases the prediction).")

    if st.checkbox("Show raw inputs and scaled vector"):
        st.write("Input (original):")
        st.write(X_input.T)
        st.write("Scaled input:")
        st.write(pd.DataFrame(X_scaled, columns=feature_order).T)

# Footer: show model info if you want
st.sidebar.markdown("---")
st.sidebar.write("Model: SVR (RBF), C=10.0, epsilon=0.1")
