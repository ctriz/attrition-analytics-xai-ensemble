# /src/app.py

# =============================================================================
# HR Attrition Analysis - using XGBoost & CatBoost - Refactored
# =============================================================================

import os, sys
from pathlib import Path
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Define the file path 
script_path = Path(__file__).resolve()
project_dir = script_path.parent.parent

print(f'script_path: {script_path}')
print(f'project_dir: {project_dir}')

FILE_PATH = project_dir / 'data' / 'raw' / 'hrdb_sim.csv'
ENRICHED_FILE_PATH = project_dir / 'data' / 'raw' / 'hrdb_enriched.csv'

from data.datagen_base_emp_record import DataGenEmpBaseRecord
from data.datagen_org_ext_record import DataGenOrgExternalData
from analysis.data_analysis_enhanced import AdvancedEDA
from feature.transform_model_ready_features import FeaturesTransformedModel
from modelling.xgboost_enhanced import XGBoostModelEval
from modelling.catboost_enhanced import CatBoostModelEval
from analysis.shap_analysis import ShapExplainer

# 1. Generate employees
# gen_data = DataGenEmpBaseRecord(n=5000, seed=42).save_to_csv()

# 2. Enrich with synthetic external/org data
# gen_org_ext_data = DataGenOrgExternalData().run_pipeline()

# 3. Analyse data (Optional - comment this for debug)
# eda = AdvancedEDA(str(ENRICHED_FILE_PATH)).run_full_eda()

# 4. Apply feature engineering for both models
df_enriched = pd.read_csv(ENRICHED_FILE_PATH)
fe = FeaturesTransformedModel(df_enriched)
X_df, y = fe.prepare_features(return_df=True)

# Safe distribution print
print("Target distribution:", pd.Series(y).value_counts(normalize=True))

'''
# 5A. Model Training & Eval - XGBoost uses Engineered features
print("=" * 50)
print("Training XGBoost Model")
print("=" * 50)

xgb_eval = XGBoostModelEval()   
xgb_results = xgb_eval.train_and_evaluate(X_df, y, save_model=True)

print(f"XGBoost AUC: {xgb_results['roc_auc']:.4f}")

-----------

# 5B. Model Training & Eval - CatBoost with SAME engineered features
print("=" * 50)
print("Training CatBoost Model (with engineered features)")
print("=" * 50)

cat_model_fixed = CatBoostModelEvalFixed()
cat_results_fixed = cat_model_fixed.train_and_evaluate_with_raw_data(
    df_enriched, 
    target_col="Attrition", 
    save_model=True
)

print(f"CatBoost (Engineered) AUC: {cat_results_fixed['roc_auc']:.4f}")

-----------

'''
"""
------------

# 6. SHAP Analysis
print("=" * 50)
print("SHAP Analysis")
print("=" * 50)

# Get engineered features for SHAP (same as used for training)
fe_shap = FeaturesTransformedModel(df_enriched)
X_engineered, y_shap = fe_shap.prepare_features(return_df=True)

# Remove leakage feature
if "attrition_prob" in X_engineered.columns:
    X_engineered = X_engineered.drop(columns=["attrition_prob"])

   # Safe distribution print
print("Target distribution for SHAP:", pd.Series(y_shap).value_counts(normalize=True))

print(f"SHAP analysis on {X_engineered.shape[1]} engineered features")
# XGBoost SHAP (using the engineered model)
try:
    shap_xgb = ShapExplainer(xgb_results["model"], X_engineered, reports_dir="reports", model_type="xgboost")
    shap_xgb.create_shap_plots()
    shap_xgb.save_summary_report()
    print("XGBoost SHAP analysis completed!")
except Exception as e:
    print(f"XGBoost SHAP analysis failed: {e}")



# CatBoost SHAP (using the original model)
try:
    shap_cat = ShapExplainer(cat_results_fixed["model"], X_engineered, reports_dir="reports", model_type="catboost")
    shap_cat.create_shap_plots() 
    shap_cat.save_summary_report()
    print("CatBoost SHAP analysis completed!")
except Exception as e:
    print(f"CatBoost SHAP analysis failed: {e}")
    
print("=" * 50)   
-----------
"""


# 7. Ensemble Model Training (Optional)
print("=" * 50)
print("Training Ensemble Model")
print("=" * 50)

from modelling.ensemble_model import EnsembleModelEval

ensemble_eval = EnsembleModelEval()
ensemble_results = ensemble_eval.create_ensemble_with_raw_data(
    df_enriched, 
    target_col="Attrition", 
    save_model=True
)

print(f"Best Ensemble Strategy: {ensemble_results['best_strategy']}")
print(f"Ensemble AUC: {ensemble_results['results'][ensemble_results['best_strategy']]['roc_auc']:.4f}")

# 6. Enhanced SHAP Analysis (including ensemble)
print("=" * 50)
print("Enhanced SHAP Analysis (with Ensemble)")
print("=" * 50)

from analysis.shap_analysis_ensemble import EnsembleShapExplainer

BASE_DIR = project_dir / 'src'
X_engineered, y_shap = fe.prepare_features(return_df=True)

# Run ensemble SHAP analysis
ensemble_shap = EnsembleShapExplainer(
    BASE_DIR / "api" / "model_ensemble.pkl",
    X_engineered,
    reports_dir="reports"
)
ensemble_shap.run_full_analysis()

print("=" * 50)
print("Enhanced SHAP Analysis (with Ensemble)")
print("=" * 50)


################################
# End of pipeline
################################

print("=" * 50)
print("Pipeline execution completed successfully!")
print("=" * 50)
print(f"Models saved to: {project_dir / 'src' / 'api'}")
print("Available models:")
print("- model_xgb.pkl (XGBoost)")
print("- model_cat_engineered.pkl (CatBoost with engineered features)")