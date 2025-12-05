# Cases per 100k Predictor (Streamlit)

How to run locally:
1. python -m venv .venv
2. source .venv/bin/activate   # or .venv\Scripts\activate on Windows
3. pip install -r requirements.txt
4. streamlit run app.py

The 'deploy_artifacts/' folder must contain:
 - best_model_final.pkl
 - scaler.pkl
 - feature_order.json
 - training_stats.json

These are produced by the final Colab cell.
