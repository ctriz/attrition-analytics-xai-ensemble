# data_analysis_enhanced.py

# =========================================================================================================
# Analyze the dataset for EDA, Bias, Correlation
# =========================================================================================================

import os
import sys
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# To import modules from the src directory
script_path = Path(__file__).resolve()
src_path = script_path.parent.parent
sys.path.append(str(src_path))


# Define file paths
script_path = Path(__file__).resolve()
project_dir = script_path.parent.parent.parent

# NOTE: point to the enriched CSV that contains org/external synthetic features
FILE_PATH = project_dir / 'data' / 'raw' / 'hrdb_enriched.csv'

# Helper Functions
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

def cramers_v(confusion_matrix):
    """Calculate Cramér's V statistic for categorical-categorical association."""
    chi2, p, dof, ex = chi2_contingency(confusion_matrix)
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    # bias correction
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    denom = min((kcorr-1), (rcorr-1))
    if denom <= 0:
        return 0.0
    return np.sqrt(phi2corr / denom)

def cohens_d(x1, x2):
    """Calculate Cohen's d for numeric differences between two groups."""
    nx1, nx2 = len(x1), len(x2)
    if nx1 < 2 or nx2 < 2:
        return np.nan
    pooled_std = np.sqrt(((nx1 - 1)*np.var(x1, ddof=1) + (nx2 - 1)*np.var(x2, ddof=1)) / (nx1 + nx2 - 2))
    if pooled_std == 0:
        return np.nan
    return (np.mean(x1) - np.mean(x2)) / pooled_std


class AdvancedEDA:
    """
    Advanced Exploratory Data Analysis class for HR dataset exploration.
    All results are printed as text/tables, with no plotting.
    Chi-square tests are limited to relevant categorical variables vs Attrition.
    """
    
    def __init__(self, data_path=None, df=None):
        """
        Initialize EDA with either a file path or DataFrame.
        
        Parameters:
        data_path (str): Path to the dataset file.
        df (pd.DataFrame): Pre-loaded DataFrame.
        """
        if df is not None:
            self.df = df.copy()
        elif data_path is not None:
            self.df = self.load_data(data_path)
        else:
            raise ValueError("Either data_path or df must be provided")
        if self.df is None:
            raise ValueError("Data loading failed. Check the file path and file contents.")

        # Ignore columns with high-cardinality or identifiers
        self.ignore_cols = ['FirstName', 'LastName', 'Email', 'EmployeeID']
        
        # Ensure Attrition is normalized to consistent labels (YES / NO)
        if 'Attrition' in self.df.columns:
            self.df['Attrition'] = self.df['Attrition'].astype(str).str.strip()
            # normalize common variants
            self.df['Attrition'] = self.df['Attrition'].str.upper().replace({'YE': 'YES', 'Y': 'YES', 'N': 'NO'})
            # keep 'Yes'/'No' style for downstream grouping if needed - but above is uppercase
            # We'll keep uppercase 'YES'/'NO' internally
    
        # Build numeric/categorical lists excluding ignore columns
        # Some synthetic features may be floats but stored as object - attempt conversion where safe
        # Force numeric conversion for the synthetic numeric features if possible
        for col in ['GlassdoorRating', 'JobMarketIndex', 'MedianSalaryGroup', 'TeamCohesion', 'NineBoxScore']:
            if col in self.df.columns:
                try:
                    self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
                except Exception:
                    pass

        self.numeric_cols = [c for c in self.df.select_dtypes(include=[np.number]).columns if c not in self.ignore_cols]
        self.categorical_cols = [c for c in self.df.select_dtypes(include=['object', 'category']).columns if c not in self.ignore_cols]
        self.datetime_cols = [c for c in self.df.select_dtypes(include=['datetime64']).columns if c not in self.ignore_cols]
       
    def load_data(self, data_path):
        """
        Load data from various file formats.
        """
        try:
            if str(data_path).endswith('.csv'):
                return pd.read_csv(data_path)
            elif str(data_path).endswith('.xlsx') or str(data_path).endswith('.xls'):
                return pd.read_excel(data_path)
            elif str(data_path).endswith('.json'):
                return pd.read_json(data_path)
            elif str(data_path).endswith('.parquet'):
                return pd.read_parquet(data_path)
            else:
                raise ValueError("Unsupported file format")
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def dataset_overview(self):
        """Comprehensive dataset overview."""
        print("="*60)
        print("DATASET OVERVIEW")
        print("="*60)
        
        print(f"Dataset Shape: {self.df.shape[0]} rows × {self.df.shape[1]} columns")
        print(f"Memory Usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        print()
        
        print("Column Information:")
        print("-" * 40)
        for i, column in enumerate(self.df.columns, 1):
            dtype = self.df[column].dtype
            null_count = self.df[column].isnull().sum()
            null_pct = (null_count / len(self.df)) * 100
            unique_count = self.df[column].nunique()
            
            print(f"{i:2d}. {column:<25} | {str(dtype):<12} | "
                  f"Nulls: {null_count:>5} ({null_pct:>5.1f}%) | "
                  f"Unique: {unique_count:>5}")
        
        print()
        print(f"Numeric columns ({len(self.numeric_cols)}): {', '.join(self.numeric_cols)}")
        print(f"Categorical columns ({len(self.categorical_cols)}): {', '.join(self.categorical_cols)}")
        if self.datetime_cols:
            print(f"DateTime columns ({len(self.datetime_cols)}): {', '.join(self.datetime_cols)}")
        else:
            print("DateTime columns (0): None")
        print()

        # Summary statistics for numeric columns
        if self.numeric_cols:
            print("Numeric Columns Summary Statistics:")
            print(self.df[self.numeric_cols].describe().round(2).to_string())
            print()
        else:
            print("No numeric columns found!")
            print()
        # Value counts for categorical columns
        print("Categorical Columns Value Counts (Top 5):")
        if self.categorical_cols:
            for col in self.categorical_cols:
                print(f"\n{col}:")
                try:
                    print(self.df[col].value_counts().head(5).to_string())
                except Exception:
                    print("[Could not compute value counts]")
            print()
        else:
            print("No categorical columns found!")
            print()
        
        # Data quality checks
        print("Data Quality Checks:")
        if self.df.columns.duplicated().any():
            dup_cols = self.df.columns[self.df.columns.duplicated()].tolist()
            print(f"Warning: Duplicate column names found: {dup_cols}")
        else:
            print("No duplicate column names found.")
        mixed_type_cols = [col for col in self.df.columns if self.df[col].apply(type).nunique() > 1]
        if mixed_type_cols:
            print(f"Warning: Columns with mixed data types: {mixed_type_cols}")
        else:
            print("No columns with mixed data types found.")
        single_value_cols = [col for col in self.df.columns if self.df[col].nunique() == 1]
        if single_value_cols:
            print(f"Columns with a single unique value: {single_value_cols}")
        else:
            print("No columns with a single unique value.")
        no_missing_cols = [col for col in self.df.columns if self.df[col].isnull().sum() == 0]
        if no_missing_cols:
            print(f"Columns with no missing data: {no_missing_cols}")
        else:
            print("No columns with complete data (no missing values).")
        complete_missing_cols = [col for col in self.df.columns if self.df[col].isnull().sum() == len(self.df)]
        if complete_missing_cols:
            print(f"Columns with complete missing data: {complete_missing_cols}")
        else:
            print("No columns with complete missing data.")
        potential_numeric = []
        for col in self.categorical_cols:
            try:
                pd.to_numeric(self.df[col], errors='raise')
                potential_numeric.append(col)
            except:
                pass
        if potential_numeric:
            print(f"Categorical columns that might be numeric: {', '.join(potential_numeric)}")
        else:
            print("No categorical columns appear to be numeric.")
        
        # High cardinality categorical columns
        print("High Cardinality Categorical Columns:")
        high_cardinality_cols = []
        for col in self.categorical_cols:
            cardinality_ratio = self.df[col].nunique() / len(self.df) if len(self.df) > 0 else 0
            if cardinality_ratio > 0.9:
                high_cardinality_cols.append(f"{col} ({cardinality_ratio:.1%})")
        if high_cardinality_cols:
            print(f"High cardinality categorical columns: {', '.join(high_cardinality_cols)}")
        else:
            print("No high cardinality categorical columns found.")
        
        # Numeric Data Range Summary
        print("\nNumeric Data Range Summary:")
        for col in self.numeric_cols:
            try:
                col_min = self.df[col].min()
                col_max = self.df[col].max()
                print(f"{col}: Min = {col_min}, Max = {col_max}")
            except Exception:
                pass
        
        # Check for Negative Values in Columns like age and salary
        range_issues = [col for col in self.numeric_cols if self.df[col].min() < 0]
        if range_issues:
            print(f"Warning: Numeric columns with negative values: {range_issues}")

        # Detect skewed category variables 
        skewed_cats = []
        for col in self.categorical_cols:
            try:
                top_freq = self.df[col].value_counts(normalize=True).iloc[0]
                if top_freq > 0.95:
                    skewed_cats.append(f"{col} ({top_freq:.1%} in '{self.df[col].mode()[0]}')")
            except Exception:
                pass
        if skewed_cats:
            print(f"Highly imbalanced categorical columns: {', '.join(skewed_cats)}")

        # Automatically detect identifier-like columns (e.g., EmployeeID, Email) where nunique ≈ nrows.
        id_like = [col for col in self.df.columns if self.df[col].nunique() == len(self.df)]
        if id_like:
            print(f"Potential identifier columns (unique per row): {id_like}")

        # Quick check: Numeric columns vs Attrition (robust label detection)
        if 'Attrition' in self.df.columns:
            attr_vals = [str(v).upper() for v in self.df['Attrition'].dropna().unique()]
            if any(v.startswith('Y') for v in attr_vals) and any(v.startswith('N') for v in attr_vals):
                print("\nQuick Correlation with Attrition (Mean Differences):")
                for col in self.numeric_cols:
                    try:
                        means = self.df.groupby(self.df['Attrition'].str.upper())[col].mean()
                        if 'YES' in means.index and 'NO' in means.index:
                            diff = abs(means['YES'] - means['NO'])
                            avg_val = self.df[col].mean()
                            if avg_val != 0 and diff > 0.1 * abs(avg_val):  # >10% difference
                                print(f"- {col}: mean differs by {diff:.2f} between YES/NO (avg={avg_val:.2f})")
                    except Exception:
                        pass

        # Final summary and recommendations
        print()
        print("Initial Data Cleaning Recommendations:")
        print("- Remove or impute columns with high missing data (>50%)")
        print("- Address columns with single unique values")
        print("- Investigate columns with mixed data types")
        print("- Convert potential numeric categorical columns to numeric")
        print("- Handle high cardinality categorical columns appropriately")
        print("- Parse DateTime columns if needed")
        print("- Check for duplicates and inconsistent entries")
        print("Dataset overview complete.")
        print("="*60)
    
    def missing_data_analysis(self):
        """Analyze missing data patterns."""
        print("\n" + "="*60)
        print("MISSING DATA ANALYSIS")
        print("="*60)
        
        missing_data = self.df.isnull().sum()
        missing_percent = (missing_data / len(self.df)) * 100
        
        missing_df = pd.DataFrame({
            'Column': missing_data.index,
            'Missing_Count': missing_data.values,
            'Missing_Percentage': missing_percent.values
        }).sort_values('Missing_Count', ascending=False)
        
        missing_df = missing_df[missing_df['Missing_Count'] > 0]
        
        if len(missing_df) > 0:
            print("Columns with missing values:")
            print(missing_df.to_string(index=False))
            high_missing_cols = missing_df[missing_df['Missing_Percentage'] > 50]
            if not high_missing_cols.empty:
                print("\nColumns with >50% missing values (consider removing):")
                print(high_missing_cols.to_string(index=False))
        else:
            print("No missing values found in the dataset!")

    def numeric_analysis(self):
        """Analyze numeric columns."""
        if not self.numeric_cols:
            print("\nNo numeric columns found!")
            return
            
        print("\n" + "="*60)
        print("NUMERIC COLUMNS ANALYSIS")
        print("="*60)
        
        # Descriptive statistics
        print("Descriptive Statistics:")
        desc_stats = self.df[self.numeric_cols].describe()
        print(desc_stats.to_string())
        
        # Additional statistics
        print("\nAdditional Statistics:")
        additional_stats = pd.DataFrame({
            'Skewness': self.df[self.numeric_cols].skew(),
            'Kurtosis': self.df[self.numeric_cols].kurtosis(),
            'CV (%)': (self.df[self.numeric_cols].std() / self.df[self.numeric_cols].mean()) * 100
        })
        print(additional_stats.to_string())
        
        # Outlier detection using IQR, by Department
        print("\nOutlier Analysis (IQR Method) by Department:")
        outlier_summary = []
        for col in self.numeric_cols:
            if 'Department' not in self.df.columns:
                break
            for dept in self.df['Department'].unique():
                try:
                    dept_data = self.df[self.df['Department'] == dept][col].dropna()
                    if len(dept_data) < 4:
                        continue
                    Q1 = dept_data.quantile(0.25)
                    Q3 = dept_data.quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    outliers = dept_data[(dept_data < lower_bound) | (dept_data > upper_bound)]
                    outlier_summary.append({
                        'Column': col,
                        'Department': dept,
                        'Outlier_Count': len(outliers),
                        'Outlier_Percentage': (len(outliers) / len(dept_data)) * 100,
                        'Lower_Bound': round(lower_bound, 2),
                        'Upper_Bound': round(upper_bound, 2)
                    })
                except Exception:
                    pass
        
        if outlier_summary:
            outlier_df = pd.DataFrame(outlier_summary)
            print(outlier_df.sort_values('Outlier_Percentage', ascending=False).to_string(index=False))
        else:
            print("No outlier summary available.")

    def categorical_analysis(self):
        """Analyze categorical columns and show attrition % by category."""
        if not self.categorical_cols:
            print("\nNo categorical columns found!")
            return
            
        print("\n" + "="*60)
        print("CATEGORICAL COLUMNS ANALYSIS")
        print("="*60)
        
        for col in self.categorical_cols:
            print(f"\n{col}:")
            print("-" * (len(col) + 1))
            
            try:
                value_counts = self.df[col].value_counts(dropna=False)
                print(f"Unique values: {self.df[col].nunique()}")
                print(f"Most frequent: '{value_counts.index[0]}' ({value_counts.iloc[0]} times)")
                
                if len(value_counts) <= 20:
                    print("\nValue counts:")
                    for idx, (value, count) in enumerate(value_counts.head(10).items()):
                        percentage = (count / len(self.df)) * 100
                        print(f"  {value}: {count} ({percentage:.1f}%)")
                    if len(value_counts) > 10:
                        print(f"  ... and {len(value_counts) - 10} more categories")
                else:
                    print(f"Too many categories to display ({len(value_counts)} unique values)")
            except Exception as e:
                print(f"Could not compute counts for {col}: {e}")

            # Attrition % by category (if Attrition column exists)
            if 'Attrition' in self.df.columns and col != 'Attrition':
                try:
                    # use upper-case YES/NO internal representation
                    attr_rates = (
                        self.df.groupby(col)['Attrition']
                        .apply(lambda x: (x.str.upper() == 'YES').mean() * 100)
                        .round(2)
                    )
                    print("\nAttrition Rate by Category (%):")
                    print(attr_rates.to_string())
                except Exception as e:
                    print(f"Could not calculate attrition rate for {col}: {e}")
    
    def attrition_analysis(self):
        """Analyze relationships between Attrition and selected variables including effect sizes."""
        print("\n" + "="*60)
        print("ATTRITION ANALYSIS")
        print("="*60)

        # robustly find yes/no labels (we use uppercase 'YES'/'NO' internally)
        if 'Attrition' not in self.df.columns:
            print("Attrition column not found.")
            return

        # --- Numeric variables vs Attrition ---
        print("\nNumeric Variables vs Attrition (Mean/Median + Cohen's d):")
        numeric_stats = []
        for col in self.numeric_cols:
            try:
                grouped = self.df.groupby(self.df['Attrition'].str.upper())[col].agg(['mean', 'median'])
                yes_vals = self.df[self.df['Attrition'].str.upper() == 'YES'][col].dropna()
                no_vals  = self.df[self.df['Attrition'].str.upper() == 'NO'][col].dropna()
                d = cohens_d(yes_vals, no_vals)
                numeric_stats.append({
                    'Column': col,
                    'Mean_Yes': round(grouped.loc['YES', 'mean'], 2) if 'YES' in grouped.index else np.nan,
                    'Mean_No': round(grouped.loc['NO', 'mean'], 2) if 'NO' in grouped.index else np.nan,
                    'Median_Yes': round(grouped.loc['YES', 'median'], 2) if 'YES' in grouped.index else np.nan,
                    'Median_No': round(grouped.loc['NO', 'median'], 2) if 'NO' in grouped.index else np.nan,
                    "Cohen's d": round(d, 3) if not np.isnan(d) else np.nan
                })
            except Exception:
                pass
        if numeric_stats:
            print(pd.DataFrame(numeric_stats).to_string(index=False))

        # --- Categorical variables vs Attrition (Chi-square + Cramér's V) ---
        relevant_cats = [
            'Department', 'JobRole', 'Gender', 'MaritalStatus', 'EducationLevel',
            'OverTime', 'BusinessTravel', 'LeadershipRoles', 'NineBox'
        ]
        print("\nSelected Categorical Variables vs Attrition (Chi-square + Cramér's V):")
        chi2_results = []
        for col in relevant_cats:
            if col in self.df.columns and col in self.categorical_cols and self.df[col].nunique() > 1:
                try:
                    contingency_table = pd.crosstab(self.df[col], self.df['Attrition'].str.upper())
                    if contingency_table.shape[0] < 2 or contingency_table.shape[1] < 2:
                        continue
                    chi2, p, _, _ = stats.chi2_contingency(contingency_table)
                    cv = cramers_v(contingency_table)
                    chi2_results.append({
                        'Column': col,
                        'Chi2_Stat': round(chi2, 2),
                        'P_Value': round(p, 3),
                        "Cramér's V": round(cv, 3)
                    })
                    print(f"\nContingency Table for {col} vs Attrition:")
                    print(contingency_table.to_string())
                except Exception as e:
                    print(f"Skipping chi-square test for {col}: {e}")

        if chi2_results:
            print("\nChi-square + Cramér's V Summary:")
            print(pd.DataFrame(chi2_results).to_string(index=False))
    
    
    def manager_hierarchy_analysis(self):
        """Analyze manager-employee relationships and hierarchy."""
        print("\n" + "="*60)
        print("MANAGER AND HIERARCHY ANALYSIS")
        print("="*60)
        
        if 'HierarchyLevel' in self.df.columns:
            print("\nDistribution of Hierarchy Levels:")
            hierarchy_counts = self.df['HierarchyLevel'].value_counts().sort_index()
            for level, count in hierarchy_counts.items():
                print(f"Level {level}: {count} ({count / len(self.df) * 100:.1f}%)")
        
        # Attrition rate by CurrentManager (top 5 managers with highest attrition)
        if 'CurrentManager' in self.df.columns:
            print("\nAttrition Rate by CurrentManager (Top 5):")
            manager_attrition = self.df[self.df['CurrentManager'] != 'None'].groupby('CurrentManager').agg({
                'Attrition': lambda x: (x.str.upper() == 'YES').mean() * 100,
                'EmployeeID': 'count'
            }).rename(columns={'Attrition': 'Attrition_Rate (%)', 'EmployeeID': 'Team_Size'})
            manager_attrition = manager_attrition[manager_attrition['Team_Size'] > 5]
            if not manager_attrition.empty:
                print(manager_attrition.sort_values('Attrition_Rate (%)', ascending=False).head().to_string())
        
        if 'YearsWithCurrManager' in self.df.columns:
            print("\nYearsWithCurrManager vs Attrition (Mean):")
            try:
                years_attrition = self.df.groupby(self.df['Attrition'].str.upper())['YearsWithCurrManager'].agg(['mean', 'median'])
                print(years_attrition.to_string())
            except Exception:
                pass
    
    def group_based_analysis(self):
        """
        Analyze data by key groups (Department, City, HierarchyLevel).
        Adds:
        - Attrition odds ratio vs reference group (for Department)
        - Weighted satisfaction averages (weighted by TeamSize)
        """
        print("\n" + "="*60)
        print("GROUP-BASED ANALYSIS")
        print("="*60)

        group_cols = ['Department', 'City', 'HierarchyLevel']
        for group_col in group_cols:
            if group_col not in self.df.columns:
                continue
            print(f"\nGroup Analysis by {group_col}:")
            grouped_stats = self.df.groupby(group_col).agg({
                'MonthlyIncome': ['mean', 'median'] if 'MonthlyIncome' in self.df.columns else (lambda x: np.nan),
                'JobSatisfaction': ['mean', 'median'] if 'JobSatisfaction' in self.df.columns else (lambda x: np.nan),
                'Attrition': lambda x: (x.str.upper() == 'YES').mean() * 100
            }).round(2)
            # fix columns if aggregation returned functions
            # Build column names carefully
            grouped_stats.columns = [
                'Income_Mean', 'Income_Median',
                'Satisfaction_Mean', 'Satisfaction_Median',
                'Attrition_Rate (%)'
            ][: grouped_stats.shape[1]]
            print(grouped_stats.to_string())

            # ------------------------------------------------------------------
            # (1) Attrition Odds Ratios - only meaningful for Department
            # ------------------------------------------------------------------
            if group_col == 'Department' and 'Attrition' in self.df.columns:
                ref_group = self.df['Department'].mode()[0]  # use most common dept as reference
                print(f"\nAttrition Odds Ratios by Department (ref: {ref_group})")
                odds_ratios = []
                for dept, dept_df in self.df.groupby('Department'):
                    if dept == ref_group:
                        continue
                    try:
                        p_yes_dept = (dept_df['Attrition'].str.upper() == 'YES').mean()
                        p_no_dept = (dept_df['Attrition'].str.upper() == 'NO').mean()
                        p_yes_ref = (self.df[self.df['Department'] == ref_group]['Attrition'].str.upper() == 'YES').mean()
                        p_no_ref = (self.df[self.df['Department'] == ref_group]['Attrition'].str.upper() == 'NO').mean()
                        if p_no_dept == 0 or p_no_ref == 0:
                            OR = None
                        else:
                            OR = (p_yes_dept / p_no_dept) / (p_yes_ref / p_no_ref)
                        odds_ratios.append((dept, round(OR, 3) if OR is not None else None))
                    except Exception:
                        odds_ratios.append((dept, None))
                for dept, or_val in odds_ratios:
                    print(f"{dept}: Odds Ratio = {or_val}")

            # ------------------------------------------------------------------
            # (2) Weighted Satisfaction Scores - weighted by TeamSize
            # ------------------------------------------------------------------
            if group_col == 'Department' and 'TeamSize' in self.df.columns:
                sat_cols = [c for c in self.df.columns if 'Satisfaction' in c]
                if sat_cols:
                    print("\nWeighted Average Satisfaction Scores by Department (weighted by TeamSize):")
                    weighted_avgs = {}
                    for col in sat_cols:
                        try:
                            weighted_avgs[col] = (
                                self.df.groupby('Department')
                                .apply(lambda g: np.average(g[col], weights=g['TeamSize']))
                            )
                        except Exception:
                            weighted_avgs[col] = self.df.groupby('Department')[col].mean()
                    weighted_df = pd.DataFrame(weighted_avgs)
                    print(weighted_df.round(2).to_string())


    def interaction_analysis(self):
        """Analyze key feature interactions with Attrition."""
        print("\n" + "="*60)
        print("FEATURE INTERACTION ANALYSIS")
        print("="*60)
        
        if 'TrainingHoursLastYear' in self.df.columns and 'JobSatisfaction' in self.df.columns:
            print("\nJobSatisfaction and TrainingHoursLastYear vs Attrition:")
            try:
                self.df['TrainingHours_Bin'] = pd.cut(self.df['TrainingHoursLastYear'], bins=[-1, 20, 50, 100], labels=['Low', 'Medium', 'High'])
                interaction_table = pd.crosstab(
                    [self.df['JobSatisfaction'], self.df['TrainingHours_Bin']],
                    self.df['Attrition'].str.upper(),
                    normalize='index'
                ) * 100
                print(interaction_table.round(2).to_string())
                self.df = self.df.drop(columns=['TrainingHours_Bin'])
            except Exception:
                pass
    
    def correlation_analysis(self):
        """Analyze correlations between numeric variables."""
        if len(self.numeric_cols) < 2:
            print("\nInsufficient numeric columns for correlation analysis!")
            return
            
        print("\n" + "="*60)
        print("CORRELATION ANALYSIS")
        print("="*60)
        
        corr_matrix = self.df[self.numeric_cols].corr()
        
        # Find strong correlations
        strong_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.5:
                    strong_corr.append({
                        'Variable_1': corr_matrix.columns[i],
                        'Variable_2': corr_matrix.columns[j],
                        'Correlation': round(corr_val, 2)
                    })
        
        if strong_corr:
            print("Strong correlations (|r| > 0.5):")
            strong_corr_df = pd.DataFrame(strong_corr).sort_values('Correlation', key=abs, ascending=False)
            print(strong_corr_df.to_string(index=False))
        else:
            print("No strong correlations found (|r| > 0.5)")
        
        # Full correlation matrix
        print("\nFull Correlation Matrix:")
        print(corr_matrix.round(2).to_string())
    
    def univariate_analysis(self):
        """Performs univariate analysis for all columns."""
        print("\n" + "="*60)
        print("UNIVARIATE ANALYSIS")
        print("="*60)
        
        # Numeric columns
        if self.numeric_cols:
            print("\nNumeric Column Summaries:")
            print(self.df[self.numeric_cols].describe().round(2).to_string())
        
        # Categorical columns
        print("\nCategorical Column Counts:")
        for col in self.categorical_cols:
            try:
                print(f"\nValue counts for '{col}':")
                print(self.df[col].value_counts(dropna=False).to_string())
            except Exception:
                pass

    def bivariate_analysis(self):
        """Performs bivariate analysis, limited to numeric-numeric correlations."""
        print("\n" + "="*60)
        print("BIVARIATE ANALYSIS")
        print("="*60)
        
        # Numeric vs Numeric
        print("\nNumeric vs Numeric Bivariate Analysis:")
        corr_results = []
        for i in range(len(self.numeric_cols)):
            for j in range(i + 1, len(self.numeric_cols)):
                col1, col2 = self.numeric_cols[i], self.numeric_cols[j]
                try:
                    corr = self.df[[col1, col2]].corr().iloc[0, 1]
                    corr_results.append({
                        'Variable_1': col1,
                        'Variable_2': col2,
                        'Correlation': round(corr, 2)
                    })
                except Exception:
                    pass
        if corr_results:
            print(pd.DataFrame(corr_results).to_string(index=False))

    def data_quality_report(self):
        """Generate comprehensive data quality report."""
        print("\n" + "="*60)
        print("DATA QUALITY REPORT")
        print("="*60)
        
        quality_issues = []
        
        # Check for duplicates
        duplicate_count = self.df.duplicated().sum()
        if duplicate_count > 0:
            quality_issues.append(f"Found {duplicate_count} duplicate rows")
        
        # Check for columns with single value
        single_value_cols = [col for col in self.df.columns if self.df[col].nunique() == 1]
        if single_value_cols:
            quality_issues.append(f"Columns with single value: {', '.join(single_value_cols)}")
        
        # Check for high cardinality categorical columns
        high_cardinality_cols = []
        for col in self.categorical_cols:
            try:
                cardinality_ratio = self.df[col].nunique() / len(self.df)
                if cardinality_ratio > 0.9:
                    high_cardinality_cols.append(f"{col} ({cardinality_ratio:.1%})")
            except Exception:
                pass
        if high_cardinality_cols:
            quality_issues.append(f"High cardinality categorical columns: {', '.join(high_cardinality_cols)}")
        
        # Check for potential data type issues
        potential_numeric = []
        for col in self.categorical_cols:
            try:
                pd.to_numeric(self.df[col], errors='raise')
                potential_numeric.append(col)
            except:
                pass
        if potential_numeric:
            quality_issues.append(f"Categorical columns that might be numeric: {', '.join(potential_numeric)}")
        
        if quality_issues:
            print("Potential data quality issues:")
            for i, issue in enumerate(quality_issues, 1):
                print(f"{i}. {issue}")
        else:
            print("No major data quality issues detected!")
    
    def run_full_eda(self):
        """Runs the complete EDA pipeline (selective by default)."""
        print("="*60)
        print("Starting Comprehensive EDA Analysis")
        print("="*60)

        # By default keep this light; call specific methods as needed.
        self.dataset_overview()
        # call other methods below if desired:
        self.missing_data_analysis()
        self.univariate_analysis()
        self.bivariate_analysis()
        self.numeric_analysis()
        self.categorical_analysis()
        self.correlation_analysis()
        self.attrition_analysis()
        self.manager_hierarchy_analysis()
        self.group_based_analysis()
        self.interaction_analysis()
        self.data_quality_report()
        
        print("\n" + "="*60)
        print("EDA ANALYSIS COMPLETE")
        print("="*60)


def main():
    try:
        print("Loading dataset...")
        eda = AdvancedEDA(data_path=FILE_PATH)
        eda.run_full_eda()


    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()

# ========================================================================================
# End of src/analysis/data_analysis_enhanced.py 
# ========================================================================================