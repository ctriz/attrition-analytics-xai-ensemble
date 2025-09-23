import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

"""
Creates a PCA object with a specified number of components (defaulting to 5). 
PCA finds new dimensions (called principal components) that capture the most variance in the data.

The method returns three key outputs:

explained_variance_ratio_: An array showing the proportion of the dataset's variance explained by each principal component. You can use this to see how much information each new dimension holds.
loadings: A DataFrame showing the feature loadings. This indicates how much each original variable contributes to each principal component. A high loading (positive or negative) means the original variable has a strong influence on that component.
components: The transformed data, where each original data point is represented by its scores on the new principal components.

"""

class PCAAnalyzer:
    def __init__(self, df, numeric_cols):
        self.df = df
        self.numeric_cols = numeric_cols

    def run_pca(self, n_components=5):
        """Run PCA with imputation so all rows are used."""
        imputer = SimpleImputer(strategy="mean")  # or median
        X_imputed = imputer.fit_transform(self.df[self.numeric_cols])
        X_scaled = StandardScaler().fit_transform(X_imputed)
    

        pca = PCA(n_components=n_components)
        components = pca.fit_transform(X_scaled)

        explained = pca.explained_variance_ratio_
        loadings = pd.DataFrame(
            pca.components_.T,
            columns=[f"PC{i+1}" for i in range(n_components)],
            index=self.numeric_cols
        )
        return explained, loadings, components,  self.df.index  # keep full dataset
        

"""
Output Analysis:
============================================================
RUNNING PCA ANALYSIS
============================================================
Explained Variance Ratio: [0.09096766 0.07935504 0.07490564 0.05980089 0.05601334]

PCA Loadings:
                           PC1    PC2    PC3    PC4    PC5
MonthlyIncome            0.698 -0.092  0.025 -0.039 -0.038
YearsAtCompany           0.089  0.669 -0.099 -0.107 -0.064
NumProjects              0.005 -0.016 -0.040 -0.021 -0.045
TeamSize                -0.014 -0.031 -0.026  0.084  0.034
JobSatisfaction          0.061  0.196 -0.034  0.346  0.152
EnvironmentSatisfaction -0.012 -0.009 -0.002 -0.084 -0.094
WorkLifeBalance          0.057  0.017 -0.008  0.092  0.666
DistanceFromHome        -0.024 -0.039  0.021 -0.049 -0.634
NumCompaniesWorked       0.060 -0.016  0.052 -0.014  0.067
PercentSalaryHike       -0.021  0.021 -0.055 -0.029 -0.006
PerformanceRating        0.033  0.061 -0.013  0.590 -0.161
TotalYearsExperience     0.662 -0.094  0.030 -0.042 -0.018
TownhallAttendance      -0.010  0.120  0.690 -0.013  0.013
VoluntaryContributions  -0.017  0.107  0.693 -0.017  0.015
InternalJobApplications -0.003  0.013 -0.028  0.007  0.024
TrainingHoursLastYear    0.034  0.041  0.040  0.673 -0.159
StockOptionLevel        -0.010  0.027  0.069 -0.023 -0.000
CurrentManager          -0.003  0.006  0.021  0.026  0.066
PreviousManager         -0.007 -0.019  0.017 -0.031 -0.116
YearsWithCurrManager     0.068  0.477 -0.093  0.045  0.112
YearsWithPrevManager     0.048  0.482 -0.073 -0.180 -0.148
HierarchyLevel          -0.211  0.008  0.007  0.001  0.062

The PCA output provides insight into the underlying structure of your numeric data.

***

### Explained Variance Ratio

This array shows the percentage of the total variance in the dataset that's captured by each principal component.
* **PC1**: Explains ~9.1% of the total variance.
* **PC2**: Explains ~7.9% of the total variance.
* **PC3**: Explains ~7.5% of the total variance.
* **PC4**: Explains ~6.0% of the total variance.
* **PC5**: Explains ~5.6% of the total variance.

In total, the top 5 components explain roughly 36% ($9.1 + 7.9 + 7.5 + 6.0 + 5.6$) of the dataset's variance. This suggests that the information is spread across many variables and there isn't a single dominant factor.

***

### PCA Loadings

The loadings indicate how each original variable contributes to the new principal components. A high absolute value (e.g., `0.6` or `-0.6`) means the variable has a strong influence on that component.

* **PC1**: This component is heavily influenced by **MonthlyIncome** (0.698) and **TotalYearsExperience** (0.662). This could be interpreted as a **"Career Progression"** component, as higher income and more experience often go together.
* **PC2**: This component is strongly driven by **YearsAtCompany** (0.669), **YearsWithCurrManager** (0.477), and **YearsWithPrevManager** (0.482). This seems to be a **"Job Tenure/Stability"** component.
* **PC3**: The main contributors here are **VoluntaryContributions** (0.693) and **TownhallAttendance** (0.690). This could represent a **"Company Engagement"** or **"Organizational Participation"** factor.
* **PC4**: This component is defined by **TrainingHoursLastYear** (0.673), **PerformanceRating** (0.590), and **JobSatisfaction** (0.346). This appears to be a **"Professional Development and Performance"** component.
* **PC5**: This component has strong negative loadings from **DistanceFromHome** (-0.634) and a strong positive loading from **WorkLifeBalance** (0.666). This could be a **"Work-Life Logistics"** component, suggesting that employees with better work-life balance tend to live closer to work.


"""
