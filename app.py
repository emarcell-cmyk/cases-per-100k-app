# app.py — Streamlit app to serve your SVR model with AI Explanations
import streamlit as st
import numpy as np
import pandas as pd
import pickle, json, os

# Try to import OpenAI (handle case where it's not installed locally)
try:
    from openai import OpenAI
    has_openai = True
except ImportError:
    has_openai = False

# -------- Configuration --------
ARTIFACTS_DIR = "deploy_artifacts"
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "best_model_final.pkl")
SCALER_PATH = os.path.join(ARTIFACTS_DIR, "scaler.pkl")
FEATURE_ORDER_PATH = os.path.join(ARTIFACTS_DIR, "feature_order.json")
STATS_PATH = os.path.join(ARTIFACTS_DIR, "training_stats.json")

st.set_page_config(page_title="Cases per 100k Predictor", layout="centered")
st.title("Cases per 100,000 Predictor — Demo")

st.markdown(
    "Enter the county attributes below. "
    "The app returns predicted **Cases per 100k** and uses AI to explain the socioeconomic context."
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
    
    # Load training stats if available, else default
    training_stats = {}
    if os.path.exists(STATS_PATH):
        training_stats = json.load(open(STATS_PATH, "r"))
    else:
        training_stats = {f: {"min": 0.0, "max": 1.0, "mean": 0.5} for f in feature_order}
        
    return model, scaler, feature_order, training_stats

model, scaler, feature_order, training_stats = load_artifacts()

# -------- OpenAI Helper Function --------
def get_ai_explanation(api_key, inputs, prediction):
    if not has_openai:
        return "OpenAI library not installed."
        
    client = OpenAI(api_key=api_key)
    
    # The PROMPT: Injecting context about Reverse Causality
    system_prompt = """
    You are an expert Public Health Analyst. You are analyzing a machine learning model predicting HIV cases per 100k people.
    
    CRITICAL CONTEXT FOR YOUR ANALYSIS:
    1. The model often shows that HIGH "Percent With Prep Prescription" leads to HIGH "Cases". Explain that this is likely REVERSE CAUSALITY: PrEP resources are deployed most aggressively in areas that already have high epidemics.
    2. Poverty and Lack of HS Diploma are standard risk factors.
    3. Be concise (max 3-4 sentences).
    4. Speak to the user like a policymaker.
    """
    
    user_prompt = f"""
    Analyze this specific county scenario:
    INPUT DATA: {inputs}
    
    MODEL PREDICTION: {prediction:.2f} Cases per 100,000.
    
    Explain why these specific demographics might lead to this prediction based on socioeconomic factors.
    """
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo", 
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.7
    )
    return response.choices[0].message.content

# -------- UI: Sidebar --------
st.sidebar.header("Configuration")

# 1. API Key Logic (Secrets -> Sidebar -> None)
api_key = None
if "OPENAI_API_KEY" in st.secrets:
    api_key = st.secrets["OPENAI_API_KEY"]
else:
    api_key = st.sidebar.text_input("OpenAI API Key (Optional)", type="password", help="Enter key to enable AI explanations.")

st.sidebar.markdown("---")
st.sidebar.header("Input Features")

# 2. Build Inputs
input_dict = {}
for feat in feature_order:
    s = training_stats.get(feat, {"min": 0.0, "max": 1.0, "mean": 0.5})
    lo, hi, mu = float(s.get("min", 0.0)), float(s.get("max", 1.0)), float(s.get("mean", 0.5))
    step = (hi - lo) / 100 if hi > lo else 0.01
    input_dict[feat] = st.sidebar.slider(feat, min_value=lo, max_value=hi, value=mu, step=step)

# -------- Main Prediction Logic --------
if st.sidebar.button("Predict"):
    # A. Prepare Input
    X_input = pd.DataFrame([input_dict], columns=feature_order)
    
    # B. Add Dummy Target Column (Fix for Scaler Mismatch)
    # Scaler expects 5 cols (features + target), so we add a dummy 0.0 for target
    X_input_for_scaling = X_input.copy()
    X_input_for_scaling["Cases/100,000"] = 0.0 
    
    # C. Scale
    X_scaled_full = scaler.transform(X_input_for_scaling)
    X_scaled_features = X_scaled_full[:, :-1] # Slice off the dummy target
    
    # D. Predict (Result is Scaled 0-1)
    pred_scaled = model.predict(X_scaled_features)[0]
    
    # E. Inverse Transform Prediction (Get back to Real Cases)
    dummy_inverse_row = np.zeros((1, 5)) 
    dummy_inverse_row[0, -1] = pred_scaled
    pred_real = scaler.inverse_transform(dummy_inverse_row)[0, -1]

    st.markdown("## Prediction")
    st.metric("Predicted Cases per 100,000", f"{pred_real:.2f}")

    # -------- AI Explanation Section --------
    st.markdown("---")
    st.subheader("🤖 AI Analysis")
    
    if api_key:
        with st.spinner("Consulting the AI Analyst..."):
            try:
                explanation = get_ai_explanation(api_key, input_dict, pred_real)
                st.info(explanation)
            except Exception as e:
                st.error(f"AI Error: {e}")
    else:
        st.warning("To get an AI explanation, please add your OpenAI API Key in the sidebar or app secrets.")

    # -------- Local Explanation (Data Table) --------
    st.markdown("### Feature Impacts (Math)")
    contributions = []
    base_pred = pred_real
    
    # Calculate impact of each feature by comparing to "Average"
    for feat in feature_order:
        X_repl = X_input.copy()
        mean_val = training_stats.get(feat, {}).get("mean", 0.5)
        X_repl.loc[0, feat] = mean_val
        
        # Scale with dummy target
        X_repl_for_scaling = X_repl.copy()
        X_repl_for_scaling["Cases/100,000"] = 0.0
        X_repl_scaled_full = scaler.transform(X_repl_for_scaling)
        X_repl_features = X_repl_scaled_full[:, :-1]
        
        # Predict & Inverse Transform
        pred_repl_scaled = model.predict(X_repl_features)[0]
        dummy_inv_repl = np.zeros((1, 5))
        dummy_inv_repl[0, -1] = pred_repl_scaled
        pred_repl_real = scaler.inverse_transform(dummy_inv_repl)[0, -1]
        
        delta = base_pred - pred_repl_real
        contributions.append((feat, delta))

    contributions = sorted(contributions, key=lambda x: abs(x[1]), reverse=True)
    contrib_df = pd.DataFrame(contributions, columns=["feature", "impact_on_cases"]).head(10)
    
    st.dataframe(contrib_df.style.format({"impact_on_cases": "{:.2f}"}))
    st.caption("Positive `impact` means this feature value is increasing the predicted case count compared to the average county.")
