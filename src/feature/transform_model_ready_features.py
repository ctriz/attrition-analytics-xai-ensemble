# src/feature/feature_engg_enhanced.py

# =========================================================================================================
# Transform enriched dataset into model-ready features (X) and target (y).
# =========================================================================================================

"""
# Splits X and Y
# Encodes Categorical Features
# Handles Synthetic & Interactive Features
# Returns both pd and np
Steps:
Normalize target (Attrition → 0/1).
Handle missing values.
Encode categoricals (one-hot or target encoding).
Add engineered interaction features:
    LowCohesion_LowNineBox
    Glassdoor_x_JobMarket
    Tenure_x_JobMarket
    LowSalary_LowNineBox
Drop leakage columns (attrition_prob).
Return clean X and y.
Produces reports like feature_effects.csv.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import chi2_contingency


class FeaturesTransformedModel:
    """
    Enhanced feature engineering for HR attrition dataset.
    Includes synthetic features (GlassdoorRating, JobMarketIndex, TeamCohesion, NineBox, NineBoxScore).
    Produces effect size reports with a flag for synthetic features.
    """

    def __init__(self, df, target_col="Attrition", reports_dir="reports", high_card_thresh=50):
        self.df = df.copy()
        self.target_col = target_col
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.high_card_thresh = high_card_thresh
        self.ignore_cols = ["FirstName", "LastName", "Email", "EmployeeID"]

        # Mark synthetic features explicitly (base + engineered interactions)
        self.synthetic_features = [
            "GlassdoorRating",
            "JobMarketIndex",
            "TeamCohesion",
            "NineBox",
            "NineBoxScore",
            "LowCohesion_LowNineBox",
            "Glassdoor_x_JobMarket",
            "Tenure_x_JobMarket",
            "LowSalary_LowNineBox"
        ]

    def _normalize_target(self):
        if self.target_col not in self.df.columns:
            raise ValueError(f"Target column '{self.target_col}' not found in DataFrame")

        col = self.df[self.target_col]

        if pd.api.types.is_numeric_dtype(col):
            # Already numeric (0/1), just enforce integer dtype
            self.df[self.target_col] = col.astype(int)
            print(f"[Info] Target '{self.target_col}' detected as numeric 0/1. Keeping as is.")
        else:
            # Normalize string labels to YES/NO → 1/0
            self.df[self.target_col] = (
                col.astype(str)
                .str.strip()
                .str.upper()
                .replace({"Y": "YES", "N": "NO", "YE": "YES"})
            )
            self.df[self.target_col] = (self.df[self.target_col] == "YES").astype(int)
            print(f"[Info] Target '{self.target_col}' normalized from string labels to numeric 0/1.")


    # -------------------------
    # Effect size helpers
    # -------------------------
    def _cohens_d(self, x, y):
        if len(np.unique(y)) != 2:
            return np.nan
        group1, group2 = x[y == 0], x[y == 1]
        if group1.std() == 0 and group2.std() == 0:
            return 0.0
        nx, ny = len(group1), len(group2)
        pooled_std = np.sqrt(((nx - 1) * group1.std() ** 2 + (ny - 1) * group2.std() ** 2) / (nx + ny - 2))
        if pooled_std == 0:
            return 0.0
        return (group1.mean() - group2.mean()) / pooled_std

    def _cramers_v(self, confusion_matrix):
        chi2, _, _, _ = chi2_contingency(confusion_matrix)
        n = confusion_matrix.sum().sum()
        phi2 = chi2 / n
        r, k = confusion_matrix.shape
        return np.sqrt(phi2 / min(k - 1, r - 1))

    def _compute_effect_sizes(self, X, y):
        """Compute Cohen’s d (numeric) and Cramér’s V (categorical)."""
        effects = []
        for col in X.columns:
            try:
                if pd.api.types.is_numeric_dtype(X[col]):
                    d = self._cohens_d(X[col].values, y)
                    f_type = "Numeric"
                    effect_val = d
                else:
                    cm = pd.crosstab(X[col], y)
                    v = self._cramers_v(cm)
                    f_type = "Categorical"
                    effect_val = v

                effects.append({
                    "Feature": col,
                    "Type": f_type,
                    "EffectSize": effect_val,
                    "Synthetic": col.split("_")[0] in self.synthetic_features or col in self.synthetic_features
                })
            except Exception:
                effects.append({
                    "Feature": col,
                    "Type": "Unknown",
                    "EffectSize": np.nan,
                    "Synthetic": col in self.synthetic_features
                })

        effects_df = pd.DataFrame(effects).sort_values("EffectSize", key=np.abs, ascending=False)
        effects_df.to_csv(self.reports_dir / "feature_effects.csv", index=False)
        print(f"[Report] Feature effect sizes written to {self.reports_dir/'feature_effects.csv'}")

    # -------------------------
    # Main method
    # -------------------------

    """
    It takes the raw enriched HR dataset and prepares:

        X → the features matrix (numeric + encoded categoricals)

        y → the target (attrition, 0/1)
    
    Copy and normalize target
        Makes sure Attrition is consistently numeric (0 = stay, 1 = leave).
    Separate numeric and categorical features
        Numeric = continuous features (e.g., age, salary).
        Categoricals split into low-cardinality (e.g., Gender, Department) vs high-cardinality (e.g., JobTitle).
    Encode categoricals
        Low-cardinality → one-hot encode (dummies).
        High-cardinality → target encode (replace with attrition rate per category).
    Build feature matrix (X)
        Combine numerics + encoded categoricals into one DataFrame.
    Extract target (y)
        Handle both numeric and string forms safely.
    Prevent leakage
        Drop Attrition from features if it was mistakenly included.
    Diagnostics
        Print out feature counts, dataset shape, and target distribution.
    Effect size computation
        Quantify how strongly each feature relates to attrition (saved to CSV).
    Return formats
        Return X and y either as pandas objects (return_df=True) or numpy arrays (return_df=False).
    """
    def prepare_features(self, return_df=False):
        # Make a working copy of the DataFrame
        df = self.df.copy()
        # Normalize target column (convert YES/NO or numeric into consistent 0/1)
        self._normalize_target()
        # -------------------------------------------------------------------
        # 1. Identify numeric and categorical columns (excluding ignored ones)
        # -------------------------------------------------------------------
        numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in self.ignore_cols]
        cat_candidates = [c for c in df.select_dtypes(include=["object", "category"]).columns if c not in self.ignore_cols]

        # If target column is categorical, remove it from the candidates
        if self.target_col in cat_candidates:
            cat_candidates.remove(self.target_col)

        
        # -------------------------------------------------------------------
        # 2. Handle missing values in numeric columns
        # -------------------------------------------------------------------
        for col in numeric_cols:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())

        # Split categorical features by cardinality
        low_card_cols = [c for c in cat_candidates if df[c].nunique() <= self.high_card_thresh]
        high_card_cols = [c for c in cat_candidates if df[c].nunique() > self.high_card_thresh]

        # -------------------------------------------------------------------
        # 3. Encode categorical features
        # -------------------------------------------------------------------
        
        # (a) One-hot encode low-cardinality categoricals
        df_ohe = pd.get_dummies(df[low_card_cols].astype(str), drop_first=True, dummy_na=False) if low_card_cols else pd.DataFrame(index=df.index)

        # (b) Target encode high-cardinality categoricals
        df_te = pd.DataFrame(index=df.index)
        if high_card_cols and self.target_col in df.columns:
            for col in high_card_cols:
                # Compute mean attrition rate per category
                te_map = df.groupby(col)[self.target_col].apply(lambda x: (x == "YES").mean())
                # Map category to that mean, fill unknowns with global mean
                df_te[col + "_TE"] = df[col].map(te_map).fillna(te_map.mean())

        # -------------------------------
        # 3A. Add engineered interaction features
        # -------------------------------

        # Binary interaction: Low Cohesion × Low NineBox
        df['LowCohesion_LowNineBox'] = ((df['TeamCohesion'] < 0.4) & (df['NineBoxScore'] <= 3)).astype(int)

        # Continuous interaction: Glassdoor × JobMarketIndex
        df['Glassdoor_x_JobMarket'] = df['GlassdoorRating'] * df['JobMarketIndex']

        # Short-tenured employees are more likely to leave when the external job market is hot
        df['Tenure_x_JobMarket'] = df['YearsAtCompany'] * df['JobMarketIndex']

        #Underpaid employees with poor career progression signals (low NineBoxScore) are flight risks.
        df['LowSalary_LowNineBox'] = ((df['MonthlyIncome'] < df['MonthlyIncome'].median()) & (df['NineBoxScore'] <= 3)).astype(int)



        # -------------------------------------------------------------------
        # 4. Combine all feature groups into X
        # -------------------------------------------------------------------
        
        # Ensure engineered interaction features are included with numeric columns
        engineered_cols = [
            "LowCohesion_LowNineBox",
            "Glassdoor_x_JobMarket",
            "Tenure_x_JobMarket",
            "LowSalary_LowNineBox"
        ]

        X = pd.concat([df[numeric_cols + engineered_cols].reset_index(drop=True),
                       df_ohe.reset_index(drop=True),
                       df_te.reset_index(drop=True)], axis=1)
        
        if "attrition_prob" in X.columns:
            print("[Warning] Dropping leakage column 'attrition_prob'")
            X = X.drop(columns=["attrition_prob"])
        # -------------------------------------------------------------------
        # 5. Extract target (y)
        # ------------------------------------------------------------------- 

        if self.target_col in df.columns:
            if pd.api.types.is_numeric_dtype(df[self.target_col]):
                # Already numeric 0/1
                y = df[self.target_col].astype(int).copy()
                print(f"[Debug] Using numeric target with {y.sum()} positives out of {len(y)}")
            else:
                # String case:Convert YES/NO strings to 1/0
                y = (df[self.target_col].astype(str).str.upper() == "YES").astype(int)
                print(f"[Debug] Converted string target to numeric with {y.sum()} positives out of {len(y)}")
        else:
            raise ValueError(f"Target column '{self.target_col}' not found in DataFrame")

        # -------------------------------------------------------------------
        # 6. Prevent leakage: drop target column if it snuck into X
        # -------------------------------------------------------------------    

        if self.target_col in X.columns:
            print(f"[Warning] Dropping target '{self.target_col}' from features to avoid leakage.")
            X = X.drop(columns=[self.target_col])
        # Drop columns with all-NaN
        X = X.loc[:, X.notna().any(axis=0)]

        # -------------------------------------------------------------------
        # 7. Diagnostics
        # -------------------------------------------------------------------
        print(f"Using {len(numeric_cols)} numeric and {len(low_card_cols)+len(high_card_cols)} categorical features")
        print(f"Prepared features shape: {X.shape}, Target shape: {y.shape}")
        
        # Show attrition balance summary
        # Show target distribution (count and proportion)
        print("\nTarget distribution (Attrition):")
        print(y.value_counts())
        print(y.value_counts(normalize=True).round(3))

        # -------------------------------------------------------------------
        # 8. Effect size report (Cohen’s d, Cramér’s V)
        # -------------------------------------------------------------------
        self._compute_effect_sizes(X, y)

        # -------------------------------------------------------------------
        # 09 Return data in requested format
        # -------------------------------------------------------------------

        if return_df:
            return X, y  # DataFrame + Series
        else:
            return X.values, y.values # Keep numpy option for backward compatibility
