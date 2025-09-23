# src/modelling/catboost_enhanced.py

################################
# CatBoost Model Evaluation with Engineered Features
################################


from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
from catboost import CatBoostClassifier, Pool
import pickle
import sys

# Add feature engineering import
sys.path.append(str(Path(__file__).resolve().parent.parent))
from feature.transform_model_ready_features import FeaturesTransformedModel

# Define the file path 
script_path = Path(__file__).resolve()
pickle_dir = script_path.parent.parent / 'api'

class CatBoostModelEval:
    """
    Fixed CatBoost training that uses the SAME engineered features as XGBoost
    """
    print("Initialized CatBoostModelEval with engineered features")
    def __init__(self, random_state=42, reports_dir="reports"):
        self.rs = random_state
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.model = None
    
    def train_and_evaluate_with_raw_data(self, df_raw, target_col="Attrition", save_model=True):
        """
        Train CatBoost using the SAME feature engineering pipeline as XGBoost
        
        Parameters:
        -----------
        df_raw : pd.DataFrame
            Raw enriched dataset (same as used for XGBoost)
        target_col : str
            Target column name
        save_model : bool
            Whether to save the model
        """
        
        print(f"[Info] Training CatBoost with engineered features (like XGBoost)")
        print(f"[Debug] Raw data shape: {df_raw.shape}")
        
        # Apply the SAME feature engineering as XGBoost
        fe = FeaturesTransformedModel(df_raw, target_col=target_col)
        X_engineered, y = fe.prepare_features(return_df=True)
        
        print(f"[Debug] After feature engineering: {X_engineered.shape}")
        print(f"[Debug] Engineered features: {list(X_engineered.columns)}")
        
        # Remove any leakage features
        if "attrition_prob" in X_engineered.columns:
            X_engineered = X_engineered.drop(columns=["attrition_prob"])
            print(f"[Debug] Removed attrition_prob, final shape: {X_engineered.shape}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_engineered, y, test_size=0.2, stratify=y, random_state=self.rs
        )
        
        print(f"[Debug] Train shape: {X_train.shape}, Test shape: {X_test.shape}")
        
        # Since all features are now numeric (one-hot encoded), treat as numeric
        # No categorical features for CatBoost to handle specially
        train_pool = Pool(X_train, y_train, feature_names=X_engineered.columns.tolist())
        test_pool = Pool(X_test, y_test, feature_names=X_engineered.columns.tolist())
        
        # Train CatBoost model (no categorical features specified)
        model = CatBoostClassifier(
            iterations=1000,
            learning_rate=0.05,
            depth=6,
            eval_metric="AUC",
            random_seed=self.rs,
            verbose=100,
            early_stopping_rounds=50,
            auto_class_weights='Balanced',
            boosting_type='Plain',
            l2_leaf_reg=3
        )
        
        print("[Info] Training CatBoost model with engineered features...")
        model.fit(train_pool, eval_set=test_pool, use_best_model=True, verbose=True)
        
        # Predictions
        y_pred_proba = model.predict_proba(test_pool)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        # Evaluation
        roc = roc_auc_score(y_test, y_pred_proba)
        pr = average_precision_score(y_test, y_pred_proba)
        
        print(f"\n=== CatBoost Results (With Engineered Features) ===")
        print(f"ROC AUC: {roc:.6f}")
        print(f"PR AUC (AP): {pr:.6f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, digits=3))
        
        # Feature importances
        feat_names = model.feature_names_
        importance_vals = model.get_feature_importance()
        imp_df = pd.DataFrame({
            "Feature": feat_names, 
            "Importance": importance_vals
        }).sort_values("Importance", ascending=False)
        
        # Save reports
        imp_df.to_csv(self.reports_dir / "catboost_engineered_feature_importances.csv", index=False)
        print(f"[Report] Feature importances saved")
        
        # Top 10 features
        print(f"\nTop 10 Most Important Features:")
        print(imp_df.head(10).to_string(index=False))
        
        # Save model
        if save_model:
            self.model = model
            required_features = X_engineered.columns.tolist()
            
            # No categorical features since everything is engineered
            categorical_feature_info = {'names': [], 'indices': []}
            
            # No scaler needed since feature engineering handles normalization
            scaler = None
            
            # Save package
            Path(pickle_dir).mkdir(parents=True, exist_ok=True)
            package = (self.model, required_features, categorical_feature_info, scaler)
            
            with open(Path(pickle_dir) / "model_cat_engineered.pkl", "wb") as f:
                pickle.dump(package, f)
            
            print(f"[Model] Saved CatBoost model with engineered features to model_cat_engineered.pkl")
        
        return {
            "model": model,
            "roc_auc": roc,
            "pr_auc": pr,
            "importances": imp_df
        }
"""
# Example usage in app.py
# 5B. Model Training & Eval - CatBoost with SAME engineered features
cat_model = CatBoostModelEval()
cat_results = cat_model.train_and_evaluate_with_engineered_features(X_df, y, save_model=True)
print(f"CatBoost (Engineered) AUC: {cat_results['roc_auc']:.4f}")
""" 

