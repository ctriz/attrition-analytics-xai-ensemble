# src/modelling/ensemble_model.py


# =============================================================================
# Ensemble model that combines XGBoost and CatBoost predictions
# Uses multiple strategies: averaging, voting, and meta-learning
# =============================================================================

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
import pickle
import sys

# Add feature engineering import
sys.path.append(str(Path(__file__).resolve().parent.parent))
from feature.transform_model_ready_features import FeaturesTransformedModel

# Define the file path 
script_path = Path(__file__).resolve()
pickle_dir = script_path.parent.parent / 'api'

print(f'ensemble script_path: {script_path}')
print(f'ensemble pickle_dir: {pickle_dir}')

class EnsembleModelEval:
    """
    Ensemble model that combines XGBoost and CatBoost predictions
    Uses multiple strategies: averaging, voting, and meta-learning
    """
    
    def __init__(self, random_state=42, reports_dir="reports"):
        self.rs = random_state
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.ensemble_model = None
        self.base_models = {}
        self.meta_learner = None
        
    def load_base_models(self):
        """Load pre-trained XGBoost and CatBoost models"""
        
        # Load XGBoost model
        xgb_path = pickle_dir / "model_xgb.pkl"
        if xgb_path.exists():
            with open(xgb_path, "rb") as f:
                xgb_model, xgb_features, xgb_cat_info, xgb_scaler = pickle.load(f)
            self.base_models['xgboost'] = {
                'model': xgb_model,
                'features': xgb_features,
                'cat_info': xgb_cat_info,
                'scaler': xgb_scaler
            }
            print(f"[Info] Loaded XGBoost model with {len(xgb_features)} features")
        else:
            raise FileNotFoundError(f"XGBoost model not found at {xgb_path}")
            
        # Load CatBoost model
        cat_path = pickle_dir / "model_cat_engineered.pkl"
        if cat_path.exists():
            with open(cat_path, "rb") as f:
                cat_model, cat_features, cat_cat_info, cat_scaler = pickle.load(f)
            self.base_models['catboost'] = {
                'model': cat_model,
                'features': cat_features,
                'cat_info': cat_cat_info,
                'scaler': cat_scaler
            }
            print(f"[Info] Loaded CatBoost model with {len(cat_features)} features")
        else:
            raise FileNotFoundError(f"CatBoost model not found at {cat_path}")

    def create_ensemble_with_raw_data(self, df_raw, target_col="Attrition", save_model=True):
        """
        Create ensemble using raw data and feature engineering
                
        Parameters:
        -----------
        df_raw : pd.DataFrame
            Raw enriched dataset
        target_col : str
            Target column name
        save_model : bool
            Whether to save the ensemble model
        """
        
        print(f"[Info] Creating ensemble model from raw data")
        print(f"[Debug] Raw data shape: {df_raw.shape}")
        
        # Load base models
        self.load_base_models()
        
        # Apply feature engineering
        fe = FeaturesTransformedModel(df_raw, target_col=target_col)
        X_engineered, y = fe.prepare_features(return_df=True)
        
        # Remove leakage features
        if "attrition_prob" in X_engineered.columns:
            X_engineered = X_engineered.drop(columns=["attrition_prob"])
            
        print(f"[Debug] After feature engineering: {X_engineered.shape}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_engineered, y, test_size=0.2, stratify=y, random_state=self.rs
        )
        
        # Get predictions from base models
        print("[Info] Getting base model predictions...")
        
        # XGBoost predictions
        import xgboost as xgb
        print("[Debug] Preparing DMatrix for XGBoost...")
        dtest_train = xgb.DMatrix(X_train, feature_names=X_engineered.columns.tolist())
        dtest_test = xgb.DMatrix(X_test, feature_names=X_engineered.columns.tolist())
        
        print("[Debug] Making predictions with XGBoost...")
        xgb_train_proba = self.base_models['xgboost']['model'].predict(dtest_train)
        xgb_test_proba = self.base_models['xgboost']['model'].predict(dtest_test)
        
        # CatBoost predictions
        print("[Debug] Making predictions with CatBoost...")
        cat_train_proba = self.base_models['catboost']['model'].predict_proba(X_train)[:, 1]
        cat_test_proba = self.base_models['catboost']['model'].predict_proba(X_test)[:, 1]
        
        print(f"[Debug] XGBoost train proba range: {xgb_train_proba.min():.3f} - {xgb_train_proba.max():.3f}")
        print(f"[Debug] CatBoost train proba range: {cat_train_proba.min():.3f} - {cat_train_proba.max():.3f}")
        
        # Create ensemble predictions using different strategies
        ensemble_results = {}
        
        # Strategy 1: Simple Average
        # Combine predictions by averaging probabilities
        print("[Info] Evaluating Simple Average ensemble...")

        avg_train_proba = (xgb_train_proba + cat_train_proba) / 2
        avg_test_proba = (xgb_test_proba + cat_test_proba) / 2
        avg_test_pred = (avg_test_proba >= 0.5).astype(int)
        
        avg_roc = roc_auc_score(y_test, avg_test_proba)
        avg_pr = average_precision_score(y_test, avg_test_proba)

        print(f"[Debug] Simple average train proba range: {avg_train_proba.min():.3f} - {avg_train_proba.max():.3f}")
        print(f"[Debug] Simple average test proba range: {avg_test_proba.min():.3f} - {avg_test_proba.max():.3f}")
        print(f"[Debug] Simple average test predictions: {avg_test_pred[:5]}")

        ensemble_results['simple_average'] = {
            'roc_auc': avg_roc,
            'pr_auc': avg_pr,
            'predictions': avg_test_proba
        }
        
        print(f"[Info] Simple Average ROC AUC: {avg_roc:.6f}, PR AUC: {avg_pr:.6f}")
        
        
        # Strategy 2: Weighted Average (favor the better performing model)
        # Calculate individual model performance on validation set
        print("[Info] Evaluating Weighted Average ensemble...")

        xgb_roc = roc_auc_score(y_test, xgb_test_proba)
        cat_roc = roc_auc_score(y_test, cat_test_proba)
        
        # Weight based on performance
        print(f"[Debug] XGBoost ROC AUC: {xgb_roc:.6f}, CatBoost ROC AUC: {cat_roc:.6f}")

        total_roc = xgb_roc + cat_roc
        xgb_weight = xgb_roc / total_roc
        cat_weight = cat_roc / total_roc
                
        weighted_train_proba = (xgb_train_proba * xgb_weight) + (cat_train_proba * cat_weight)
        weighted_test_proba = (xgb_test_proba * xgb_weight) + (cat_test_proba * cat_weight)
        
        weighted_roc = roc_auc_score(y_test, weighted_test_proba)
        weighted_pr = average_precision_score(y_test, weighted_test_proba)
        
        ensemble_results['weighted_average'] = {
            'roc_auc': weighted_roc,
            'pr_auc': weighted_pr,
            'predictions': weighted_test_proba,
            'weights': {'xgboost': xgb_weight, 'catboost': cat_weight}
        }
        
        # Strategy 3: Meta-learner (Logistic Regression)
        # Logistic regression on base model predictions
        print("[Info] Evaluating Meta-Learner ensemble...")
        print("[Debug] Creating meta-features for Meta-Learner...")
        meta_train_X = np.column_stack([xgb_train_proba, cat_train_proba])
        meta_test_X = np.column_stack([xgb_test_proba, cat_test_proba])
        
        # Train meta-learner
        print("[Debug] Training meta-learner (Logistic Regression)...")

        meta_learner = LogisticRegression(random_state=self.rs)
        meta_learner.fit(meta_train_X, y_train)
        
        meta_test_proba = meta_learner.predict_proba(meta_test_X)[:, 1]
        meta_roc = roc_auc_score(y_test, meta_test_proba)
        meta_pr = average_precision_score(y_test, meta_test_proba)
        
        ensemble_results['meta_learner'] = {
            'roc_auc': meta_roc,
            'pr_auc': meta_pr,
            'predictions': meta_test_proba,
            'model': meta_learner
        }
        
        # Choose best strategy
        print("[Info] Selecting best ensemble strategy...")
        best_strategy = max(ensemble_results.keys(), 
                          key=lambda k: ensemble_results[k]['roc_auc'])
        
        print(f"\n=== Ensemble Results ===")
        print(f"Individual Model Performance:")
        print(f"  XGBoost ROC AUC: {xgb_roc:.6f}")
        print(f"  CatBoost ROC AUC: {cat_roc:.6f}")
        print(f"\nEnsemble Strategies:")
        for strategy, results in ensemble_results.items():
            print(f"  {strategy.title()}: ROC AUC: {results['roc_auc']:.6f}, PR AUC: {results['pr_auc']:.6f}")
            
        print(f"\nBest Strategy: {best_strategy.title()}")
        print(f"Best ROC AUC: {ensemble_results[best_strategy]['roc_auc']:.6f}")
        
        # Classification report for best strategy
        best_pred = (ensemble_results[best_strategy]['predictions'] >= 0.5).astype(int)
        print(f"\nClassification Report (Best Strategy - {best_strategy.title()}):")
        print(classification_report(y_test, best_pred, digits=3))
        
        # Save ensemble model
        if save_model:
            self.ensemble_model = {
                'strategy': best_strategy,
                'base_models': self.base_models,
                'meta_learner': ensemble_results.get('meta_learner', {}).get('model', None),
                'weights': ensemble_results.get('weighted_average', {}).get('weights', None),
                'required_features': X_engineered.columns.tolist(),
                'feature_engineering': fe  # Save the feature engineering pipeline
            }
            
            # Save ensemble package
            Path(pickle_dir).mkdir(parents=True, exist_ok=True)
            with open(Path(pickle_dir) / "model_ensemble.pkl", "wb") as f:
                pickle.dump(self.ensemble_model, f)
            
            print(f"[Model] Saved ensemble model to model_ensemble.pkl")
            print(f"[Model] Strategy: {best_strategy}")
            
        # Save detailed results
        results_df = pd.DataFrame({
            'Strategy': list(ensemble_results.keys()),
            'ROC_AUC': [r['roc_auc'] for r in ensemble_results.values()],
            'PR_AUC': [r['pr_auc'] for r in ensemble_results.values()]
        }).sort_values('ROC_AUC', ascending=False)
        
        results_df.to_csv(self.reports_dir / "ensemble_comparison.csv", index=False)
        print(f"[Report] Ensemble comparison saved to ensemble_comparison.csv")
        
        return {
            'ensemble_model': self.ensemble_model,
            'best_strategy': best_strategy,
            'results': ensemble_results,
            'individual_performance': {'xgboost': xgb_roc, 'catboost': cat_roc}
        }
    

"""
# Example usage in app.py
# 7. Ensemble Model Training (Optional)
ensemble_model = EnsembleModelTraining()
ensemble_results = ensemble_model.train_and_evaluate(X_train, y_train, X_test, y_test, save_model=True)
print(f"Ensemble Model (Optional) AUC: {ensemble_results['best_strategy']['roc_auc']:.4f}")
try:
    shap_cat = ShapExplainer(
        BASE_DIR / "api" / "model_cat_engineered.pkl",
        X_engineered,
        reports_dir="reports"
    )
    shap_cat.run_full_analysis()
    print("CatBoost SHAP analysis completed!")
except Exception as e:
    print(f"CatBoost SHAP analysis failed: {e}")
    shap_cat = None
print("=" * 50)
"""

