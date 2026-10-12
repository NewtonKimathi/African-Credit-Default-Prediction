# African Credit Default Prediction Project
## Problem Statement
Credit institutions and micro-lenders across emerging African markets face significant credit risk due to asymmetric financial information and limited
formal credit history. Inaccurate risk assessment leads to elevated non-performing loan (NPL) rates, while overly conservative underwriting excludes 
creditworthy borrowers. Developing an automated, data-driven scoring system is essential to evaluate borrower risk accurately and reduce loan defaults.
## Objectives
**1.Predict Default Risk: Build and tune an end-to-end binary classification model to forecast the likelihood of customer default.**
**2.Handle Class Imbalance: Optimize models specifically for default detection by targeting evaluation metrics like F1-Score rather than raw accuracy.**
**3.Robust Generalization: Implement cross-validation strategies to minimize overfitting and ensure consistent performance on out-of-sample data.**
**4.Model Comparison: Benchmark multiple classification algorithms to determine the optimal trade-off between interpretability and predictive power.**
## Results
1. Results & Benchmarks : Competition Standing: Secured 4th place out of 14 teams (Top 30%) on the final leaderboard.
2. Metric Optimization : Successfully balanced precision and recall to maximize the cross-validated F1-Score across minority positive classes.
3. Model Performance : Tree-based ensemble methods (XGBoost and Random Forest) demonstrated superior separation on non-linear interactions compared to baseline
   Logistic Regression models
## Tech Stacks & Tools
Language: Python
Machine Learning & Modeling: Scikit-learn (Logistic Regression, Random Forest, Cross-Validation, Evaluation Metrics), XGBoost
Data Processing & Analysis: Pandas, NumPy
Data Visualization: Matplotlib, Seaborn
Environment: Jupyter Notebook
<img src="https://cdn.simpleicons.org/pandas/150458" 
     alt="Pandas" width="40" height="40" />
