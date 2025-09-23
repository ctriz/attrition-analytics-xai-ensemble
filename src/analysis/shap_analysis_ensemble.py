# src/analysis/shap_analysis_ensemble.py

########################################
# SHAP analysis for ensemble models
# Provides explanations for both individual models and ensemble predictions
########################################

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from pathlib import Path
import pickle

class EnsembleShapExplainer:
    """
    SHAP analysis for ensemble models
    Provides explanations for both individual models and ensemble predictions
    """
    
    def __init__(self, ensemble_model_path, X_sample, reports_dir="reports", sample_size=100):
        print("=" * 50)
        print("Initializing Ensemble SHAP Explainer")
        self.ensemble_path = ensemble_model_path
        self.X_sample = X_sample.sample(n=min(sample_size, len(X_sample)), random_state=42)
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.feature_names = list(X_sample.columns)
        
        # Load ensemble model
        print(f"[Info] Loading ensemble model from {ensemble_model_path}")
        with open(ensemble_model_path, "rb") as f:
            self.ensemble_data = pickle.load(f)
        
        self.explainers = {}
        self.shap_values = {}
        
    def create_explainers(self):
        """Create SHAP explainers for individual models"""
        
        print("[Info] Creating SHAP explainers for ensemble components...")
        
        # XGBoost explainer
        if 'xgboost' in self.ensemble_data['base_models']:
            print("[Info] Creating XGBoost explainer...")
            import xgboost as xgb
            
            xgb_model = self.ensemble_data['base_models']['xgboost']['model']
            self.explainers['xgboost'] = shap.TreeExplainer(xgb_model)
            
            # Convert to DMatrix for prediction
            dmatrix = xgb.DMatrix(self.X_sample, feature_names=self.feature_names)
            self.shap_values['xgboost'] = self.explainers['xgboost'].shap_values(self.X_sample)
            
        # CatBoost explainer  
        if 'catboost' in self.ensemble_data['base_models']:
            print("[Info] Creating CatBoost explainer...")
            
            cat_model = self.ensemble_data['base_models']['catboost']['model']
            self.explainers['catboost'] = shap.TreeExplainer(cat_model)
            self.shap_values['catboost'] = self.explainers['catboost'].shap_values(self.X_sample)
        
        print(f"[Info] Created explainers for {len(self.explainers)} models")
        
    def create_ensemble_shap_values(self):
        """Create ensemble SHAP values based on the ensemble strategy"""
        print("[Info] Creating ensemble SHAP values...")
        
        strategy = self.ensemble_data['strategy']
        
        if strategy == 'simple_average':
            # Average SHAP values
            print("[Info] Using simple average for ensemble SHAP values")
            self.shap_values['ensemble'] = (
                self.shap_values['xgboost'] + self.shap_values['catboost']
            ) / 2
            
        elif strategy == 'weighted_average':
            # Weighted average of SHAP values
            print("[Info] Using weighted average for ensemble SHAP values")
            weights = self.ensemble_data['weights']
            self.shap_values['ensemble'] = (
                self.shap_values['xgboost'] * weights['xgboost'] + 
                self.shap_values['catboost'] * weights['catboost']
            )
            
        elif strategy == 'meta_learner':
            # For meta-learner, create SHAP values showing contribution of each base model
            print("[Info] Using meta-learner for ensemble SHAP values")
            xgb_model = self.ensemble_data['base_models']['xgboost']['model']
            cat_model = self.ensemble_data['base_models']['catboost']['model']
            
            # Get base model predictions
            import xgboost as xgb
            dmatrix = xgb.DMatrix(self.X_sample, feature_names=self.feature_names)
            xgb_pred = xgb_model.predict(dmatrix)
            cat_pred = cat_model.predict_proba(self.X_sample)[:, 1]
            
            # Meta-features
            meta_features = np.column_stack([xgb_pred, cat_pred])
            
            # Meta-learner SHAP (shows how much each base model contributes)
            meta_learner = self.ensemble_data['meta_learner']
            meta_explainer = shap.LinearExplainer(meta_learner, meta_features)
            meta_shap = meta_explainer.shap_values(meta_features)
            
            # Combine with base model SHAP values weighted by meta-learner contributions
            self.shap_values['ensemble'] = (
                self.shap_values['xgboost'] * meta_shap[:, 0:1] + 
                self.shap_values['catboost'] * meta_shap[:, 1:2]
            )
            
            # Also save meta-SHAP values
            self.meta_shap_values = meta_shap
            self.meta_feature_names = ['XGBoost_Contribution', 'CatBoost_Contribution']
        
        print(f"[Info] Created ensemble SHAP values using {strategy} strategy")
        
    def create_shap_plots(self):
        """Create comprehensive SHAP plots"""
        
        # Individual model plots
        for model_name, shap_vals in self.shap_values.items():
            if model_name == 'ensemble':
                continue
                
            # Summary plot
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_vals, self.X_sample, 
                            feature_names=self.feature_names,
                            show=False, max_display=15)
            plt.title(f'{model_name.upper()} SHAP Summary')
            plt.tight_layout()
            plt.savefig(self.reports_dir / f"{model_name}_shap_summary.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
        # Ensemble plot
        if 'ensemble' in self.shap_values:
            plt.figure(figsize=(12, 10))
            shap.summary_plot(self.shap_values['ensemble'], self.X_sample,
                            feature_names=self.feature_names,
                            show=False, max_display=20)
            plt.title('Ensemble SHAP Summary')
            plt.tight_layout()
            plt.savefig(self.reports_dir / "ensemble_shap_summary.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
        # Meta-learner plot (if applicable)
        if hasattr(self, 'meta_shap_values'):
            plt.figure(figsize=(8, 6))
            shap.summary_plot(self.meta_shap_values, 
                            np.column_stack([np.zeros(len(self.meta_shap_values)), 
                                           np.ones(len(self.meta_shap_values))]),
                            feature_names=self.meta_feature_names,
                            show=False)
            plt.title('Meta-Learner: Base Model Contributions')
            plt.tight_layout()
            plt.savefig(self.reports_dir / "meta_learner_contributions.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
        print("[Info] SHAP plots saved successfully")
        
    def save_summary_report(self):
        """Save comprehensive SHAP analysis report"""
        
        # Feature importance for each model
        for model_name, shap_vals in self.shap_values.items():
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'mean_abs_shap': np.abs(shap_vals).mean(axis=0)
            }).sort_values('mean_abs_shap', ascending=False)
            
            importance_df.to_csv(
                self.reports_dir / f"{model_name}_shap_importance.csv", 
                index=False
            )
            
        # Model comparison
        if len(self.shap_values) > 2:  # Has individual models + ensemble
            comparison_data = []
            for model_name, shap_vals in self.shap_values.items():
                if model_name == 'ensemble':
                    continue
                feature_importance = np.abs(shap_vals).mean(axis=0)
                for i, feat in enumerate(self.feature_names):
                    comparison_data.append({
                        'model': model_name,
                        'feature': feat,
                        'importance': feature_importance[i]
                    })
                    
            comparison_df = pd.DataFrame(comparison_data)
            comparison_pivot = comparison_df.pivot(index='feature', columns='model', values='importance')
            comparison_pivot.to_csv(self.reports_dir / "model_feature_comparison.csv")
            
        print("[Info] SHAP analysis reports saved")
        
    def run_full_analysis(self):
        """Run complete SHAP analysis pipeline"""
        
        print("=" * 50)
        print("Ensemble SHAP Analysis")
        print("=" * 50)
        
        self.create_explainers()
        self.create_ensemble_shap_values()
        self.create_shap_plots()
        self.save_summary_report()
        
        print("=" * 50)
        print("Ensemble SHAP Analysis Completed!")
        print(f"Reports saved to: {self.reports_dir}")
        print("=" * 50)