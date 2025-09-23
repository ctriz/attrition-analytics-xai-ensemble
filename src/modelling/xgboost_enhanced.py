# src/modelling/xgboost_enhanced.py
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import pickle

# Define the file path 
script_path = Path(__file__).resolve()
# script is in src/modelling
# pickle_dir should be src/api
pickle_dir = script_path.parent.parent / 'api'

print(f'script_path', script_path)
print(f'pickle_dir', pickle_dir)

class XGBoostModelEval:
    """
    Train & evaluate XGBoost on prepared features X (DataFrame) and y (array/Series).
    Produces:
      - ROC AUC, PR AUC
      - Classification report (default threshold 0.5)
      - Feature importance CSV
      - Separate "synthetic features" importance block
      Handles class imbalance (scale_pos_weight), saves feature importance reports, extracts synthetic feature importances.
    """

    synthetic_features = ["GlassdoorRating", "JobMarketIndex", "TeamCohesion", "NineBox", "NineBoxScore"]

    def __init__(self, random_state=42, reports_dir="reports"):
        self.rs = random_state
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.model = None

    def _ensure_frame(self, X):
        if isinstance(X, np.ndarray):
            # create generic column names
            cols = [f"f{i}" for i in range(X.shape[1])]
            return pd.DataFrame(X, columns=cols)
        if isinstance(X, pd.DataFrame):
            return X.copy()
        raise ValueError("X must be numpy array or pandas DataFrame")

    def train_and_evaluate(self, X, y,
                           test_size=0.2,
                           params=None,
                           num_boost_round=200,
                           early_stopping_rounds=30,
                           verbose_eval=False,
                           save_model=True,
                           model_dir=pickle_dir,
                           categorical_features=None):
        X = self._ensure_frame(X)
        y = np.asarray(y).ravel()

        # Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=self.rs)

        # Ensure train/test splits are DataFrames
        X_train = self._ensure_frame(X_train)
        X_test = self._ensure_frame(X_test)

        # Default params
        if params is None:
            params = {
                "objective": "binary:logistic",
                "eval_metric": "auc",
                "tree_method": "hist",
                "learning_rate": 0.05,
                "max_depth": 6,
                "seed": self.rs,
                "verbosity": 0
            }

        # handle class imbalance (scale_pos_weight)
        pos = (y_train == 1).sum()
        neg = (y_train == 0).sum()
        if pos > 0:
            params["scale_pos_weight"] = float(neg) / max(1.0, float(pos))

        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=X.columns.tolist())
        dtest = xgb.DMatrix(X_test, label=y_test, feature_names=X.columns.tolist())

        evals = [(dtrain, "train"), (dtest, "test")]
        self.model = xgb.train(params, dtrain, num_boost_round=num_boost_round,
                               evals=evals, early_stopping_rounds=early_stopping_rounds,
                               verbose_eval=verbose_eval)

        # Predict probabilities
        preds_proba = self.model.predict(dtest)
        preds = (preds_proba >= 0.5).astype(int)

        # Metrics
        roc = roc_auc_score(y_test, preds_proba)
        pr = average_precision_score(y_test, preds_proba)

        print("\nXGBoost Results")
        print(f"AUC: {roc:.6f}")
        print(f"PR AUC (AP): {pr:.6f}")
        print("\nClassification Report:")
        print(classification_report(y_test, preds, digits=3))

        # Feature importances
        importances = self.model.get_score(importance_type="weight")  # dict {feat: score}
        # convert to DataFrame and keep all features
        imp_df = (pd.DataFrame({
            "Feature": list(importances.keys()),
            "Importance": list(importances.values())
        })
        .set_index("Feature")
        .reindex(X.columns)   # align with all training features
        .fillna(0)
        .rename_axis("Feature")   # keep the index name consistent
        .reset_index()            # ensures we always have "Feature" column
        )
        imp_df = imp_df.sort_values("Importance", ascending=False).reset_index(drop=True)

        # Save full importances
        xgb_out = self.reports_dir / "xgb_feature_importances.csv"
        imp_df.to_csv(xgb_out, index=False)
        print(f"[Report] XGBoost feature importances written to {xgb_out}")

        # Save model and preprocessing artifacts
        if save_model:
            # Prepare required features
            required_features = X.columns.tolist()
            
            # Handle categorical features info (for consistency with CatBoost API)
            if categorical_features is None:
                categorical_features = []
                # Auto-detect categorical features (object/category dtypes)
                for col in X.columns:
                    if X[col].dtype in ['object', 'category']:
                        categorical_features.append(col)
            
            categorical_feature_info = {
                'names': categorical_features,
                'indices': [X.columns.get_loc(col) for col in categorical_features if col in X.columns]
            }
            
            # Create and fit scaler on numeric features only
            numeric_features = X_train.select_dtypes(include=[np.number]).columns
            scaler = StandardScaler()
            if len(numeric_features) > 0:
                scaler.fit(X_train[numeric_features])
            else:
                scaler = None
            
            # Save model package
            Path(model_dir).mkdir(parents=True, exist_ok=True)
            package = (self.model, required_features, categorical_feature_info, scaler)
            
            with open(Path(model_dir) / "model_xgb.pkl", "wb") as f:
                pickle.dump(package, f)
            
            print(f"[Model] Saved XGBoost model and preprocessing to {Path(model_dir)/'model_xgb.pkl'}")

        # Synthetic features block (match substrings for one-hot columns too)
        syn_mask = imp_df["Feature"].apply(lambda f: any(s in f for s in self.synthetic_features))
        syn_df = imp_df[syn_mask]
        if not syn_df.empty:
            print("\nTop Synthetic Feature Importances:")
            print(syn_df.sort_values("Importance", ascending=False).to_string(index=False))
            syn_out = self.reports_dir / "xgb_synthetic_feature_importances.csv"
            syn_df.to_csv(syn_out, index=False)
            print(f"[Report] Synthetic feature importances written to {syn_out}")
        else:
            print("\nNo synthetic features found among X columns.")

        # Save evaluation summary
        summary = {
            "roc_auc": roc,
            "pr_auc": pr,
            "n_train": len(y_train),
            "n_test": len(y_test),
            "scale_pos_weight": params.get("scale_pos_weight", None)
        }
        pd.Series(summary).to_csv(self.reports_dir / "xgb_eval_summary.csv")

        return {
            "model": self.model,
            "roc_auc": roc,
            "pr_auc": pr,
            "importances": imp_df,
            "synthetic_importances": syn_df
        }