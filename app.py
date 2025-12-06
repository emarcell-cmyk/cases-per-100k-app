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
    # 1. Prepare Input
    # We must match the order: Features first, then target (as defined in training CSV)
    X_input = pd.DataFrame([input_dict], columns=feature_order)
    
    # 2. Add Dummy Target Column
    # The scaler expects 5 columns because it was trained on features + target.
    # We add "Cases/100,000" with a dummy value (0) to satisfy the scaler shape.
    X_input_for_scaling = X_input.copy()
    X_input_for_scaling["Cases/100,000"] = 0.0 
    
    # 3. Scale the Input
    # This returns a numpy array with 5 columns
    X_scaled_full = scaler.transform(X_input_for_scaling)
    
    # 4. Slice to keep only Features
    # The model was trained on X (4 columns), so we slice off the last column (the dummy target)
    X_scaled_features = X_scaled_full[:, :-1]
    
    # 5. Predict (Output is Scaled 0-1)
    pred_scaled = model.predict(X_scaled_features)[0]
    
    # 6. Inverse Transform Prediction
    # The prediction is in 0-1 range. We use the scaler to reverse this to get real cases.
    # We create a dummy row of 5 zeros, place our prediction in the last slot (target slot), and inverse transform.
    dummy_inverse_row = np.zeros((1, 5)) 
    dummy_inverse_row[0, -1] = pred_scaled
    
    pred_real = scaler.inverse_transform(dummy_inverse_row)[0, -1]

    st.markdown("## Prediction")
    st.metric("Predicted Cases per 100,000", f"{pred_real:.2f}")

    # -------- Local Explanation (Mean Replacement) --------
    st.markdown("### Local feature impacts")
    contributions = []
    
    # Base prediction (real scale)
    base_pred = pred_real
    
    for feat in feature_order:
        # Create copy of input
        X_repl = X_input.copy()
        
        # Replace current feature with its training mean
        mean_val = training_stats.get(feat, {}).get("mean", 0.5)
        X_repl.loc[0, feat] = mean_val
        
        # Add dummy target again for scaling
        X_repl_for_scaling = X_repl.copy()
        X_repl_for_scaling["Cases/100,000"] = 0.0
        
        # Scale
        X_repl_scaled_full = scaler.transform(X_repl_for_scaling)
        X_repl_features = X_repl_scaled_full[:, :-1]
        
        # Predict (Scaled)
        pred_repl_scaled = model.predict(X_repl_features)[0]
        
        # Inverse Transform (to get real units)
        dummy_inv_repl = np.zeros((1, 5))
        dummy_inv_repl[0, -1] = pred_repl_scaled
        pred_repl_real = scaler.inverse_transform(dummy_inv_repl)[0, -1]
        
        # Calculate Delta
        delta = base_pred - pred_repl_real
        contributions.append((feat, delta))

    contributions = sorted(contributions, key=lambda x: abs(x[1]), reverse=True)
    contrib_df = pd.DataFrame(contributions, columns=["feature", "impact_on_cases"]).head(10)
    
    st.dataframe(contrib_df.style.format({"impact_on_cases": "{:.2f}"}))
    st.markdown("**Interpretation:** `impact_on_cases` shows how much the prediction changes (in actual cases) compared to if that feature was just 'average'.")

    if st.checkbox("Show debug data"):
        st.write("Scaled Prediction (0-1):", pred_scaled)
        st.write("Unscaled Prediction (Real):", pred_real)
        st.write("Input Features:", X_input)

# Footer
st.sidebar.markdown("---")
st.sidebar.write("Model: SVR (RBF)")
