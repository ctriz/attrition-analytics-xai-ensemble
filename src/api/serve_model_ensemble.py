# src/api/serve_model_ensemble.py

#########################################
# Flask API to serve ensemble model predictions
# Combines XGBoost and CatBoost predictions
#########################################

import pickle
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from pathlib import Path
import sys
import xgboost as xgb

# Add the src directory to the path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from feature.transform_model_ready_features import FeaturesTransformedModel

# Define paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "api" / "model_ensemble.pkl"

app = Flask(__name__)

# Load ensemble model
with open(MODEL_PATH, "rb") as f:
    ensemble_data = pickle.load(f)

print(f"[Info] Loaded ensemble model using {ensemble_data['strategy']} strategy")
print(f"[Info] Available base models: {list(ensemble_data['base_models'].keys())}")

class EnsemblePredictor:
    """Handle ensemble predictions with different strategies"""
    
    def __init__(self, ensemble_data):
        self.ensemble_data = ensemble_data
        self.strategy = ensemble_data['strategy']
        self.base_models = ensemble_data['base_models']
        self.required_features = ensemble_data['required_features']
        
    def predict_proba(self, X):
        """Make ensemble predictions"""
        
        # Get base model predictions
        predictions = {}
        
        # XGBoost prediction
        if 'xgboost' in self.base_models:
            dmatrix = xgb.DMatrix(X, feature_names=self.required_features)
            predictions['xgboost'] = self.base_models['xgboost']['model'].predict(dmatrix)
            
        # CatBoost prediction  
        if 'catboost' in self.base_models:
            predictions['catboost'] = self.base_models['catboost']['model'].predict_proba(X)[:, 1]
            
        # Ensemble prediction based on strategy
        if self.strategy == 'simple_average':
            ensemble_prob = (predictions['xgboost'] + predictions['catboost']) / 2
            
        elif self.strategy == 'weighted_average':
            weights = self.ensemble_data['weights']
            ensemble_prob = (
                predictions['xgboost'] * weights['xgboost'] + 
                predictions['catboost'] * weights['catboost']
            )
            
        elif self.strategy == 'meta_learner':
            meta_features = np.column_stack([predictions['xgboost'], predictions['catboost']])
            meta_learner = self.ensemble_data['meta_learner']
            ensemble_prob = meta_learner.predict_proba(meta_features)[:, 1]
            
        else:
            # Fallback to simple average
            ensemble_prob = (predictions['xgboost'] + predictions['catboost']) / 2
            
        return ensemble_prob, predictions

# Initialize predictor
predictor = EnsemblePredictor(ensemble_data)

@app.route("/")
def home():
    return f"""
    <h2>HR Attrition Prediction API - Ensemble Model</h2>
    <p>Strategy: {ensemble_data['strategy'].title()}</p>
    <p>Base Models: {', '.join(ensemble_data['base_models'].keys())}</p>
    <p>POST JSON to /predict</p>
    """

@app.route("/predict", methods=["POST"])
def predict():
    """Accept JSON with HR features and return ensemble attrition prediction."""
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
        
        # Remove leakage features
        if "attrition_prob" in X_engineered.columns:
            X_engineered = X_engineered.drop(columns=["attrition_prob"])
        
        # Align with required features
        X_final = X_engineered.reindex(columns=predictor.required_features, fill_value=0)
        
        # Get ensemble prediction and base model predictions
        ensemble_prob, base_predictions = predictor.predict_proba(X_final)
        ensemble_prob = ensemble_prob[0]  # Get scalar value
        
        pred_class = 1 if ensemble_prob > 0.5 else 0
        label = "Will Leave" if pred_class == 1 else "Will Stay"

        # Risk levels with updated thresholds for ensemble
        if ensemble_prob >= 0.65:
            risk_level = "High"
            risk_description = "Immediate attention required"
        elif ensemble_prob >= 0.5:
            risk_level = "Medium-High" 
            risk_description = "Monitor closely"
        elif ensemble_prob >= 0.35:
            risk_level = "Medium"
            risk_description = "Standard monitoring"
        elif ensemble_prob >= 0.2:
            risk_level = "Low-Medium"
            risk_description = "Low concern"
        else:
            risk_level = "Low"
            risk_description = "Stable employee"

        # Key factors from original input
        key_factors = {
            "job_satisfaction": request.json.get('JobSatisfaction', 'N/A'),
            "work_life_balance": request.json.get('WorkLifeBalance', 'N/A'),
            "overtime": request.json.get('OverTime', 'N/A'),
            "monthly_income": request.json.get('MonthlyIncome', 'N/A'),
            "years_at_company": request.json.get('YearsAtCompany', 'N/A'),
            "ninebox_score": request.json.get('NineBoxScore', 'N/A'),
            "performance_rating": request.json.get('PerformanceRating', 'N/A')
        }

        # Base model predictions for transparency
        base_model_predictions = {
            "xgboost": {
                "probability": round(float(base_predictions['xgboost'][0]), 3),
                "prediction": 1 if base_predictions['xgboost'][0] > 0.5 else 0
            },
            "catboost": {
                "probability": round(float(base_predictions['catboost'][0]), 3),
                "prediction": 1 if base_predictions['catboost'][0] > 0.5 else 0
            }
        }

        # Add ensemble weights if applicable
        ensemble_info = {
            "strategy": ensemble_data['strategy'],
            "description": {
                "simple_average": "Equal weight to both models",
                "weighted_average": "Performance-based weighting", 
                "meta_learner": "Logistic regression meta-learner"
            }.get(ensemble_data['strategy'], "Unknown strategy")
        }
        
        if ensemble_data['strategy'] == 'weighted_average' and 'weights' in ensemble_data:
            ensemble_info['weights'] = ensemble_data['weights']

        return jsonify({
            "model": "Ensemble",
            "ensemble_info": ensemble_info,
            "prediction": int(pred_class),
            "prediction_label": label,
            "probability": round(float(ensemble_prob), 3),
            "risk_level": risk_level,
            "risk_description": risk_description,
            "probability_percentage": f"{round(float(ensemble_prob) * 100, 1)}%",
            "key_factors": key_factors,
            "base_model_predictions": base_model_predictions,
            "ensemble_advantage": f"Combines strengths of {len(base_predictions)} models"
        })

    except Exception as e:
        print(f"[Error] Prediction failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/model-info", methods=["GET"])
def model_info():
    """Return detailed information about the ensemble model"""
    
    info = {
        "ensemble_strategy": ensemble_data['strategy'],
        "base_models": list(ensemble_data['base_models'].keys()),
        "total_features": len(predictor.required_features),
        "strategy_description": {
            "simple_average": "Averages predictions from all base models equally",
            "weighted_average": "Weights base models by their individual performance",
            "meta_learner": "Uses logistic regression to optimally combine base model predictions"
        }.get(ensemble_data['strategy'], "Unknown")
    }
    
    if ensemble_data['strategy'] == 'weighted_average' and 'weights' in ensemble_data:
        info['model_weights'] = ensemble_data['weights']
        
    return jsonify(info)

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model": "Ensemble",
        "strategy": ensemble_data['strategy'],
        "base_models": len(ensemble_data['base_models'])
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5003, debug=True)  # Port 5003 for ensemble model