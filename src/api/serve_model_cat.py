import pickle
import pandas as pd
from flask import Flask, request, jsonify
from pathlib import Path
import sys

# Add the src directory to the path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from feature.transform_model_ready_features import FeaturesTransformedModel

# Define paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "api" / "model_cat_engineered.pkl"

app = Flask(__name__)

# Load improved CatBoost model
with open(MODEL_PATH, "rb") as f:
    model, required_features, categorical_feature_info, scaler = pickle.load(f)

print(f"[Info] Loaded improved CatBoost model with {len(required_features)} features")

@app.route("/")
def home():
    return "<h2>HR Attrition Prediction API - CatBoost Improved</h2><p>POST JSON to /predict</p>"

@app.route("/predict", methods=["POST"])
def predict():
    """Accept JSON with HR features and return attrition prediction."""
    if not request.json:
        return jsonify({"error": "No JSON input provided"}), 400

    try:
        # Convert input to DataFrame
        input_df = pd.DataFrame([request.json])
        
        # Add missing fields
        defaults = {
            "Attrition": 0,
            "attrition_prob": 0.0
        }
        
        for col, default_val in defaults.items():
            if col not in input_df.columns:
                input_df[col] = default_val

        # Apply feature engineering (same as training)
        fe = FeaturesTransformedModel(input_df)
        X_engineered, _ = fe.prepare_features(return_df=True)
        
        # Align with model features
        X_final = X_engineered.reindex(columns=required_features, fill_value=0)
        
        # Predict
        pred_proba = model.predict_proba(X_final)[0][1]
        pred_class = 1 if pred_proba > 0.5 else 0
        
        label = "Will Leave" if pred_class == 1 else "Will Stay"

        # Risk levels
        if pred_proba >= 0.6:
            risk_level = "High"
            risk_description = "Immediate attention required"
        elif pred_proba >= 0.4:
            risk_level = "Medium-High" 
            risk_description = "Monitor closely"
        elif pred_proba >= 0.25:
            risk_level = "Medium"
            risk_description = "Standard monitoring"
        elif pred_proba >= 0.15:
            risk_level = "Low-Medium"
            risk_description = "Low concern"
        else:
            risk_level = "Low"
            risk_description = "Stable employee"

        # Key factors
        key_factors = {
            "job_satisfaction": request.json.get('JobSatisfaction', 'N/A'),
            "work_life_balance": request.json.get('WorkLifeBalance', 'N/A'),
            "overtime": request.json.get('OverTime', 'N/A'),
            "monthly_income": request.json.get('MonthlyIncome', 'N/A'),
            "years_at_company": request.json.get('YearsAtCompany', 'N/A'),
            "ninebox_score": request.json.get('NineBoxScore', 'N/A')
        }

        return jsonify({
            "model": "CatBoost-Improved",
            "prediction": int(pred_class),
            "prediction_label": label,
            "probability": round(float(pred_proba), 3),
            "risk_level": risk_level,
            "risk_description": risk_description,
            "probability_percentage": f"{round(float(pred_proba) * 100, 1)}%",
            "key_factors": key_factors
        })

    except Exception as e:
        print(f"[Error] Prediction failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002, debug=True)  # Port 5002