import pickle
import pandas as pd
from flask import Flask, request, jsonify
from pathlib import Path
import xgboost as xgb
import sys

# Add the src directory to the path to import feature engineering
sys.path.append(str(Path(__file__).resolve().parent.parent))
from feature.transform_model_ready_features import FeaturesTransformedModel

# Define paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "api" / "model_xgb.pkl"

app = Flask(__name__)

# Load XGBoost model
with open(MODEL_PATH, "rb") as f:
    model, required_features, categorical_feature_info, scaler = pickle.load(f)

print(f"[Info] Loaded XGBoost model with {len(required_features)} features")

@app.route("/")
def home():
    return "<h2>HR Attrition Prediction API - XGBoost</h2><p>POST JSON to /predict</p>"

@app.route("/predict", methods=["POST"])
def predict():
    """Accept JSON with HR features and return attrition prediction."""
    if not request.json:
        return jsonify({"error": "No JSON input provided"}), 400

    try:
        # Convert input to DataFrame
        input_df = pd.DataFrame([request.json])
        
        # Add missing fields with defaults if they don't exist
        defaults = {
            "Attrition": 0,  # We'll remove this later
            "attrition_prob": 0.0  # Remove leakage feature
        }
        
        for col, default_val in defaults.items():
            if col not in input_df.columns:
                input_df[col] = default_val

        print(f"[Debug] Input DataFrame shape: {input_df.shape}")
        print(f"[Debug] Input columns: {list(input_df.columns)}")

        # Apply the same feature engineering that was used during training
        fe = FeaturesTransformedModel(input_df)
        X_engineered, _ = fe.prepare_features(return_df=True)
        
        print(f"[Debug] After feature engineering shape: {X_engineered.shape}")
        print(f"[Debug] Engineered columns: {list(X_engineered.columns)}")

        # Align with model's expected features
        X_final = X_engineered.reindex(columns=required_features, fill_value=0)
        
        print(f"[Debug] Final aligned shape: {X_final.shape}")
        print(f"[Debug] Missing features filled with 0")

        # Create DMatrix for XGBoost prediction
        dtest = xgb.DMatrix(X_final, feature_names=required_features)
        
        # Predict
        pred_proba = model.predict(dtest)[0]
        pred_class = 1 if pred_proba > 0.5 else 0
        
        label = "Will Leave" if pred_class == 1 else "Will Stay"

        # Updated risk levels with more realistic thresholds for XGBoost
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

        # Extract key factors from original input
        key_factors = {
            "job_satisfaction": request.json.get('JobSatisfaction', 'N/A'),
            "work_life_balance": request.json.get('WorkLifeBalance', 'N/A'),
            "overtime": request.json.get('OverTime', 'N/A'),
            "monthly_income": request.json.get('MonthlyIncome', 'N/A'),
            "years_at_company": request.json.get('YearsAtCompany', 'N/A'),
            "business_travel": request.json.get('BusinessTravel', 'N/A'),
            "ninebox_score": request.json.get('NineBoxScore', 'N/A')
        }

        return jsonify({
            "model": "XGBoost",
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

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "model": "XGBoost",
        "model_features": len(required_features)
    })

@app.route("/required-features", methods=["GET"])
def get_required_features():
    """Return the list of required raw features (before engineering)."""
    # These are the features your JSON should contain
    raw_features = [
        "EmployeeID", "FirstName", "LastName", "Email", "Age", "City", "Race", "Ethnicity",
        "Department", "JobRole", "MonthlyIncome", "YearsAtCompany", "NumProjects", "TeamSize",
        "JobSatisfaction", "EnvironmentSatisfaction", "WorkLifeBalance", "DistanceFromHome",
        "NumCompaniesWorked", "PercentSalaryHike", "PerformanceRating", "OverTime", 
        "BusinessTravel", "MaritalStatus", "EducationLevel", "TotalYearsExperience",
        "PreviousIndustry", "TownhallAttendance", "VoluntaryContributions", "LeadershipRoles",
        "InternalJobApplications", "TrainingHoursLastYear", "StockOptionLevel", 
        "CurrentManager", "PreviousManager", "YearsWithCurrManager", "YearsWithPrevManager",
        "HierarchyLevel", "GlassdoorRating", "JobMarketIndex", "TeamCohesion", 
        "NineBox", "NineBoxScore"
    ]
    
    return jsonify({
        "raw_features_needed": raw_features,
        "engineered_features_count": len(required_features),
        "note": "Send raw features in JSON, feature engineering will be applied automatically"
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)