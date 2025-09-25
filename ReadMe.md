
## Problem Statement

Employee attrition is a persistent challenge for organizations. High turnover leads to recruitment costs, productivity losses, and cultural disruption. Predicting attrition is difficult because it depends on:

-   **Internal factors**: performance, team dynamics, career growth
    
-   **External factors**: job market conditions, employer reputation
    
This project builds a **simulated HR dataset**, applies **machine learning models (XGBoost, CatBoost, Ensemble)**, and integrates **SHAP explainability** to understand attrition drivers at both the global and individual levels.

## Step 1. Data Generation

Since no real HR data is used, created a **synthetic dataset** with ~5,000 employees and 40 variables 

-   **Demographics**: Age, Gender, Ethnicity, Marital Status
-   **Employment**: Job Role, Department, Years at Company, Years with Manager
-   **Performance**: Rating, Promotions, Internal Applications  
-   **Compensation**: Monthly Income, Salary Hike %, Stock Options
-   **Engagement**: Work-Life Balance, Environment Satisfaction, Townhall Attendance
-   **Experience**: Total Years, Previous Industry, Training Hour 

This gives a **baseline HR dataset** but is still too “flat” to explain attrition in practice.

## Step 2A. PCA & Early Attempts

Initially tried **PCA for dimensionality reduction** followed by XGBoost and CatBoost.

**Why it failed**:
-   PCA compressed categorical signals → lost HR interpretability
-   Attrition patterns looked random 
-   Models underfit, with low predictive power
 
**Observed metrics**:
-   AUC: 0.50 (Random Guessing!) 
-   AUC-PR: 0.20–0.25
-   Precision: <0.3 | Recall: <0.1
  
One might consider that in attrition analytics, **recall is critical** (catch as many leavers as possible), but too high recall → low precision (false alarms). PCA made this trade-off even harder. 

**Learning** 
- PCA captured variance (income, tenure, work-life balance) but not class separation.
- Attrition needs supervised dimension reduction, not PCA.
<img width="1000" height="800" alt="Figure_4" src="https://github.com/user-attachments/assets/f2be3f30-e39c-43d5-8be9-c28b127cb691" />



## Step 2B. All Features + Encoding + SMOTE  

 **Approach**
	- Combined numeric + categorical features.
	- One-Hot Encoding for low-cardinality, Target Encoding for high-cardinality.
	- Balanced dataset with SMOTE.
	- Trained XGBoost with and without `scale_pos_weight`.

**Results**
	- ROC AUC ≈ 0.50.
	- Recall ≈ 1–7%.

**Learning**
	- Model defaulted to predicting “No Attrition”.
	- Encoding + SMOTE introduced noise.
	- Attrition drivers are interaction-heavy, not linear.

## Step 2C. Basic Feature Engineering

**Added Features**
  `RelativeIncome` = Monthly income vs department average.
  `OvertimeStress` = OverTime × (5 – JobSatisfaction).
  `CommuteBurden` = DistanceFromHome × (6 – WorkLifeBalance).
  `ManagerStability` = YearsWithCurrManager ÷ YearsAtCompany.
  `InternalMobility` = Internal job applications ÷ YearsAtCompany.

**Results (XGBoost + SMOTE)**
	- ROC AUC ≈ 0.48–0.51.
	- Recall ≈ 1–2%.

**Learning**
		Features made sense, but XGBoost + OHE couldn’t leverage them effectively.

## Step 2D. Interactive Feature Engineering
**Added Features**

 `IncomeDissatisfaction` = (5 – JobSatisfaction) × MonthlyIncome.
 `TrainingPerformanceGap` = TrainingHoursLastYear × (5 – PerformanceRating).
 `ExternalExperienceRatio` = (TotalYearsExperience – YearsAtCompany) ÷ TotalYearsExperience.

**Results (XGBoost + SMOTE)**
* Still weak, ROC AUC ≈ 0.48.
* Recall ≈ 1%.

**Learning**

* The limitation was the lack of meaningful "attrition" hooks in the data and the feature. The model is okay!
* PCA failed as it is deterministic and captures variances
* Sparse encoding and SMOTE didn't solve imbalance (20% attrition)
* Feature Engineering gave some leading indicators that stress, mobility, and tenure ratios are key predictors  
* CatBoost performed better than XGBoost -  recall improved to 38% at 0.5 threshold.
* CatBoost's ability to **natively handle categorical features** and the creation of **interactive features** (`OvertimeStress`, `InternalMobility`) helped the model uncover hidden patterns.
*  ***Threshold tuning*** is critical** – business can choose between “safe mode” (***fewer false positives***) and “radar mode” ***(catch everyone at risk)***

## A few words on Evaluation Metrics
Focus on the key evaluation metrics for this problem statement


*  **Accuracy:** This is the ratio of correct predictions to total predictions. It can be misleading in imbalanced datasets, as a model can achieve high accuracy by simply predicting the majority class.

*  **Precision:** This is the ratio of true positive predictions to all positive predictions. It answers: "Of all the employees the model said would leave, how many actually left?"

*  **Recall:** This is the ratio of true positive predictions to all actual positive cases. It answers: "Of all the employees who actually left, how many did the model correctly identify?"

*  **F1-Score:** The harmonic mean of precision and recall. It provides a balanced measure of a model's performance.

*  **AUC-ROC:** This measures the model's ability to distinguish between classes. A score of 1.0 is perfect, and 0.5 is random.

*  **PR AUC (Average Precision):** This is a better metric for imbalanced data as it focuses on the performance on the minority class.

In this context, the cost of a **false negative** (failing to identify an employee who will leave) is higher than the cost of a **false positive** (incorrectly flagging an employee who will stay).

## Why Low Threshold is Better in this context?

*  **Cost of False Negative:** The organization loses a valuable employee, potentially incurring high costs for recruitment and training a replacement.

*  **Cost of False Positive:** The organization invests time and resources in a retention strategy for an employee who wasn't at risk.

Therefore, a **lower threshold cutoff** is better as it increases the model's **recall**, ensuring that more at-risk employees are identified. This provides the business with a comprehensive "early warning system," allowing HR to intervene, even if it means approaching some employees who were not actually planning to leave.

## Step 3. Why Additional Data Was Required

The baseline dataset lacked **external & organizational context**. In reality, employees don’t leave just because of tenure or salary — they leave because of **market pull + organizational push**.

So **simulated new features**:

-   **GlassdoorRating** → simulates employer reputation (uniform 3.5-4.0), acting as an external "pull" factor. Low ratings (<3) increase attrition probability by 0.10 in the risk model, reflecting trends where poor ratings correlate with resentment and "revenge quitting."
-    **JobMarketIndex** → simulates external demand with sinusoidal variation (85-105 range), adding 0.08 probability bump for hot markets (>7).
-   **TeamCohesion** → social/team bonding with uniform scores (40-100) proxy social bonding, with low values (<0.4) adding 0.10 risk. This addresses internal "push" factors, aligning with trends of souring worker sentiment
-   **NineBoxScore** → career growth potential, simulated career grid categories (Box_1 to Box_9) and scores (1-10), with low scores (<=3) bumping risk by 0.12. This feature targets stagnation, a key driver where employees feel career-stuck.
    
These features act as **risk multipliers** and made attrition predictions realistic. By simulating realistic external "pull" factors (e.g., job market heat) and internal "push" factors (e.g., team bonding and career alignment), this boosts predictive power while introducing controlled randomness to mimic real-world variability, where attrition isn't fully deterministic but influenced by multipliers like poor employer reputation or stagnant growth potential.

## Step 4A. Feature Engineering

Beyond synthetic data, created **interaction features** that mimic HR dynamics:

-   `LowCohesion × LowNineBox` → disengaged & no career path
-   `Glassdoor × JobMarketIndex` → external pull stronger if employer brand is weak 
-   `Tenure × JobMarketIndex` → new joiners more likely to leave in hot job markets
-   `LowSalary × LowNineBox` → underpaid + stuck = flight risk

<img width="1868" height="604" alt="image" src="https://github.com/user-attachments/assets/b8306fcb-67ba-4e53-a3b3-6d99cf316ce5" />

    
Result: Dataset became **business-realistic & interpretable**.

## Step 4B. Probabilistic Randomness

***Attrition isn't fully deterministic;*** and so introduced randomness which serves a critical role in creating realistic, probabilistic outcomes by accounting for factors that increase or decrease probability. 

Employee turnover isn't fully predictable; even with strong risk factors (e.g., low NineBoxScore <=3 signaling career stagnation), external shocks (e.g., family relocation) or internal resilience can alter outcomes. ***Randomness simulates this "noise,"*** ensuring the generated Attrition column reflects probabilistic risks rather than rigid rules. 




## Step 5. Retraining with XGBoost & CatBoost

With engineered features, models performed better:
  
  **XGBoost**:  

 - ROC AUC: ~0.58–0.60
 - PR AUC: ~0.35
 - Precision/Recall balanced around 0.4

        
 **CatBoost**:
 
   -   ROC AUC: ~0.63–0.65
   -   PR AUC: ~0.40  
    -   Better at capturing interaction/categorical signals

The original models (e.g., XGBoost with SMOTE at 0.515 AUC and 1% recall) struggled with diffuse signals and imbalance. Engineered features—via interactions like Tenure_x_JobMarket (capturing market pull on short-tenured employees) and probabilistic randomness (ensuring ~10-20% attrition)—provide stronger, combinatorial predictors.
        

#### Why CatBoost Outperformed XGBoost and the Case for Ensemble

-   **CatBoost's Strengths**: Native handling of categoricals/interactions (e.g., without explicit one-hot) and ordered boosting make it "sensitive" to subtle signals like TeamCohesion drops amid hybrid work challenges, yielding ~5% AUC gain.
-   **XGBoost's Conservatism**: Faster but less nuanced on categoricals, favoring precision for cost-efficient HR (e.g., fewer false positives in retention budgeting).
-   **Ensemble Rationale**: Averaging (e.g., via stacking) combines conservatism (XGBoost's precision) with sensitivity (CatBoost's recall), potentially lifting AUC to >0.70 and F1 to ~0.45. In 2025 HR, ensembles mitigate bias in probabilistic sims, enabling robust forecasts for interventions like manager training to curb resentment-driven churn.
    
Which is why **Ensemble** was introduced.

## Step 6. Ensemble Model & SHAP Explainability

The **Ensemble (weighted average)**  builds on prior strengths—XGBoost's tree efficiency for conservative splits, CatBoost's leaf-wise growth for sensitive interactions.

**Ensemble Strategies and Evaluation **:

-   **Simple Average**: Averages probs (e.g., (XGBoost + CatBoost)/2), smoothing differences for balanced output.
-   **Weighted Average**: Uses performance-based weights (e.g., CatBoost 51.5% for its higher base AUC, XGBoost 48.5%), emphasizing sensitivity while retaining conservatism.
-   **Meta-Learner**: Stacks base probs as features for logistic regression, learning optimal combination (e.g., higher weight to CatBoost on interactions).

**Balanced predictions:**

For high-risk inputs (e.g., low GlassdoorRating + stuck NineBoxScore), returns 43.3% (medium risk: "Address resentment via feedback"), vs. 33.7% for low-risk (low-medium: "Sustain engagement"). This business-aligned output, with 9.6% gaps, enables tiered interventions, balancing precision/recall for 2025's nuanced turnover.
    
✅ Differentiation (9.6% gap)  
✅ Balanced precision vs recall  
✅ Business-aligned risk thresholds

**SHAP Analysis** was added to:

-   Global drivers like NineBoxScore (top importance ~20%: stagnation in stuck careers), JobMarketIndex (~15%: external pull in hot markets), and GlassdoorRating (~12%: resentment from poor reputation) highlight systemic risks
-   Explain individual employee predictions (“This employee's risk boosted 15% by low cohesion + hot market")


<img width="2331" height="2814" alt="ensemble_shap_summary" src="https://github.com/user-attachments/assets/0c7a5417-173b-420a-9abe-865a92553552" />

## Deployment

Each model is exposed via **Flask APIs**:

 **Port 5001 → XGBoost** (conservative)
`python src/api/serve_model_xgb.py
curl -X POST -H "Content-Type: application/json" -d @data_high_risk.json http://127.0.0.1:5001/predict`
   
**Port 5002 → CatBoost** (sensitive)
`python src/api/serve_model_cat.py
curl -X POST -H "Content-Type: application/json" -d @data_low_risk.json http://127.0.0.1:5002/predict`   

**Port 5003 → Ensemble ** (balanced, pragmatic)
`python src/api/serve_model_ensemble.py
curl -X POST -H "Content-Type: application/json" -d @data_high_risk.json http://127.0.0.1:5003/predict`

<img width="1855" height="966" alt="image" src="https://github.com/user-attachments/assets/77464e4d-62b2-4b9e-8d33-2137db2b8ca9" />

## Step 7. Why Graph Neural Networks (GNNs)?
Attrition is not just about individual employees — it is fundamentally relational. Traditional models like XGBoost and CatBoost treat employees as independent rows in a dataset, but in reality, attrition is influenced by:

- Manager–subordinate hierarchies (poor managers account for ~50% of turnover).
- Team dynamics (low cohesion often triggers cluster exits).
- Departmental context (attrition spreads like contagion within groups).

**Graph Construction**
Nodes = 5,000 employees 
Edges = Multiple weighted types for rich organizational relationships: 
	- Hierarchy (manager–subordinate, skip-level, peer managers; weights: 1.0–0.6). 
	- Dense similarity (cosine similarity on top 20 features like CurrentManager, MonthlyIncome; threshold: 0.5). 
	- Department networks (intra- and cross-role; weights: 0.85–0.6). 
	- Performance collaboration (high/medium performers; weights: 0.9–0.5). 
	- Project history, company events, skill overlap, mentorship, cross-department events, weak ties (weights: 0.9–0.2).

This creates a dense graph with 304,999 edges (density: 2.4405%, avg. degree: 61.00), memory optimized using intelligent edge filtering (max 150,000 edges).

<img width="3660" height="3178" alt="gnn_gcn_employee_graph" src="https://github.com/user-attachments/assets/7d42f053-f277-4fc3-903b-fad84af97e36" />

**Node Features**

Each employee node carries features such as:

	-Numerical: MonthlyIncome, JobSatisfaction, TrainingHours.
	-Categorical (encoded): Department, MaritalStatus.
	-Enriched Signals:

		-EngagementIndex (eNPS/pulse surveys).
		-BurnoutIndex (overtime + absenteeism).
		-PromotionStagnation (time since last promotion).

**Model Architectures**

- Dense GCN: Primary model with 4 layers, 384 hidden channels, optimized for dense graphs.  Baseline for relational propagation.
- GraphSAGE: Supported via sampling for scalability to large organizations. Efficient neighbor sampling for large graphs.
- GAT: Supported with attention to prioritize key relationships (e.g., influential managers). Learns which edges matter more (e.g., toxic manager vs average peers).

The HybridGNNClassifier combines GCN with a tabular branch (dense layers) and an attention mechanism to capture both relational (e.g., team dynamics) and individual (e.g., promotion stagnation) attrition drivers. 


**Training implemented in PyTorch Geometric** with:
- 4 layers, 384 hidden channels, 898,438 parameters. 
- Cross-entropy loss with "balanced_focal" class weights ([0.6916, 5.4152]) for 27.7% attrition imbalance.
- AdamW optimizer (lr=0.001, weight_decay=5e-5), gradient clipping (0.5), ReduceLROnPlateau scheduler.
- 400 epochs, early stopping (patience=60), stratified splits (Train: 3,000; Val: 1,000; Test: 1,000)
	
<img width="4471" height="2955" alt="dense_gnn_gcn_training_curves" src="https://github.com/user-attachments/assets/c2fcf1ff-e4c4-46d3-afdb-2f166c248bf5" />

**Evaluation**

- Metrics: Best Val AUC 0.5822 (epoch 230), Test AUC 0.5271, below benchmarks (XGBoost: 0.650, CatBoost: 0.638).
- Threshold tuning optimizes F1, balanced accuracy, and recall for imbalanced HR data.
- Reports: ROC/PR curves, metrics (`dense_gcn_metrics.json`), training history (`dense_gcn_training_history.csv`).

**Expected Outcomes**

	-Performance: Higher recall than CatBoost/XGBoost (70% vs 38%).
	-Impact: Identify team-level attrition clusters before they cascade.
	-Business Value: Preventing 100 regrettable exits saves ~$2–5M in rehiring + lost productivity.
<img width="4470" height="3543" alt="dense_gnn_gcn_threshold_analysis" src="https://github.com/user-attachments/assets/32f3aca1-5d4f-4d59-88af-d29934fb775d" />

**Trade-Offs**

- Requires GPU for faster training (CPU used in output, 23-minute runtime).
- Data prep effort for synthetic organizational data (projects, events, skills).
- Ethical need to anonymize and audit manager-level signals to avoid bias.

## Repository Structure
<img width="663" height="284" alt="image" src="https://github.com/user-attachments/assets/cdc8095b-5cdf-4f58-be73-530d654c6fd1" />


## Key Takeaway

-   PCA + basic data → failed (low AUC, poor precision/recall trade-offs)
-   Adding **synthetic org + market data** + engineered features → realistic predictions 
-   Ensemble (XGBoost + CatBoost) → **best balance of sensitivity & conservatism**
-   SHAP explainability → transparent, business-friendly insights
The project shows how **data quality + engineered features matter more than algorithms alone** in HR attrition analytics. 

## Copyright and Licensing
Copyright © 2025 Tridib C[@ctriz]. This project is licensed under the MIT License.
 




