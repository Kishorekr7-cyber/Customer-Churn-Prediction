# Customer Churn Prediction Model

## Objective
Build a machine learning model in Python to identify customers who are
likely to stop using a service ("churn"), using demographic data, account
information, and service usage patterns.

## Problem Statement
Customer churn (customers leaving a business) is expensive to replace with
new acquisitions. If a company can predict *which* customers are at risk
of leaving *before* they do, it can proactively intervene (discounts,
support outreach, plan upgrades) to retain them. This is framed as a
**binary classification** problem: predict `Churn` = Yes/No.

## Dataset
`data/customer_churn.csv` — 2,000+ records, 21 columns, modeled on the
structure of the well-known public "Telco Customer Churn" dataset.

| Column | Description |
|---|---|
| customerID | Unique customer identifier (dropped before modeling) |
| gender, SeniorCitizen, Partner, Dependents | Demographics |
| tenure | Months the customer has stayed with the company |
| PhoneService, MultipleLines | Phone service details |
| InternetService | DSL / Fiber optic / No |
| OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies | Add-on services |
| Contract | Month-to-month / One year / Two year |
| PaperlessBilling, PaymentMethod | Billing details |
| MonthlyCharges, TotalCharges | Billing amounts |
| **Churn** | **Target** — Yes/No |

> **Note on the dataset:** No dataset file was supplied with the task and
> this environment has no internet access to sources like Kaggle. The
> dataset was therefore generated programmatically
> (`src/generate_dataset.py`) to match the structure of the well-known
> public "Telco Customer Churn" dataset, with genuine relationships
> between features and churn risk (short tenure, month-to-month contracts,
> high monthly charges, and missing add-on services all raise churn
> probability) plus realistic noise, missing values, and duplicate rows.
> The generator even reproduces a well-known real-world quirk: a few
> zero-tenure customers have `TotalCharges` stored as a blank string
> instead of a number, which the cleaning step handles explicitly.
> **To use a real dataset instead**, replace `data/customer_churn.csv`
> with your own file using similar column names.

## Technologies Used
- Python 3
- pandas, NumPy — data loading & preprocessing
- matplotlib, seaborn — visualization
- scikit-learn — Logistic Regression, Random Forest, evaluation metrics
- XGBoost — gradient-boosted tree classifier

## Features
- Full EDA: shape, dtypes, missing values, churn rate, churn by segment
- Data cleaning: duplicate removal, `TotalCharges` type-fix, median/mode imputation
- One-hot encoding of categorical features
- Stratified train/test split (churn is an imbalanced class)
- **Three models trained and compared**: Logistic Regression, Random Forest, XGBoost
- Evaluation: Accuracy, Precision, Recall, F1-score, ROC-AUC, Confusion Matrix
- ROC curve comparison across all models
- Feature importance (Random Forest) and coefficient interpretation (Logistic Regression)
- Churn-factor visualizations (contract type, tenure, charges, internet service)
- 5-fold cross-validation on the best model
- Predictions saved to `predictions.csv`

## Methodology
1. Load the raw CSV.
2. Explore the data (shape, types, missing values, churn rate).
3. Clean the data: drop `customerID`, remove duplicates, convert
   `TotalCharges` to numeric (coercing blank strings to NaN), impute
   missing numerical values with the **median** and categorical values
   with the **mode**.
4. Identify numerical vs categorical columns.
5. Visualize churn factors (contract type, tenure, monthly charges,
   internet service) and a correlation heatmap.
6. One-hot encode categorical features (`drop_first=True`).
7. Stratified 80/20 train-test split (`random_state=42`) to preserve the
   churn ratio in both sets.
8. Scale features (for Logistic Regression) using `StandardScaler`.
9. Train Logistic Regression, Random Forest, and XGBoost.
10. Evaluate each model with Accuracy, Precision, Recall, F1, ROC-AUC,
    and a Confusion Matrix.
11. Plot ROC curves for all models on one chart.
12. Extract feature importances (Random Forest) and coefficients
    (Logistic Regression) to explain *why* the model predicts churn.
13. Cross-validate the best model for a robustness check.
14. Save test-set predictions to `predictions.csv`.

## Data Preprocessing
- **`customerID`** → dropped (unique identifier, not predictive).
- **`TotalCharges`** → converted from text to numeric; blank-string
  entries (present for a handful of zero-tenure customers) become
  missing values, then are imputed.
- **Missing numerical values** → filled with the column **median**.
- **Missing categorical values** → filled with the column **mode**.
- **Duplicates** → removed entirely.
- **Encoding** → `pandas.get_dummies(..., drop_first=True)` for all
  categorical columns to avoid the dummy-variable trap.
- **Class imbalance** → handled via a *stratified* train/test split, and
  `class_weight="balanced"` for the Random Forest so it doesn't just
  predict the majority class.

## Model
Three classifiers are trained and compared:
- `sklearn.linear_model.LogisticRegression` (on scaled features)
- `sklearn.ensemble.RandomForestClassifier` (`class_weight="balanced"`)
- `xgboost.XGBClassifier`

`train_test_split(test_size=0.2, random_state=42, stratify=y)` keeps
results reproducible and class-balanced across train/test.

## Evaluation Metrics
- **Accuracy** — overall proportion of correct predictions
- **Precision** — of customers predicted to churn, how many actually did
- **Recall** — of customers who actually churned, how many did we catch
- **F1-Score** — harmonic mean of precision and recall
- **ROC-AUC** — model's ability to rank churners above non-churners across all thresholds
- **Confusion Matrix** — full breakdown of correct/incorrect predictions per class

## Results
Exact numbers vary slightly per random seed/regeneration; see
`outputs/model_comparison.csv` and the console output of
`train_model.py` for the numbers on your run. In the reference run:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | ~0.80 | ~0.61 | ~0.41 | ~0.49 | **~0.83** |
| Random Forest | ~0.75 | ~0.47 | **~0.70** | ~0.57 | ~0.81 |
| XGBoost | ~0.79 | ~0.55 | ~0.41 | ~0.47 | ~0.78 |

Logistic Regression has the best ROC-AUC, but Random Forest has much
higher **Recall** on the churn class — in a real retention program, that
recall/precision tradeoff matters more than which model has the highest
overall accuracy, since missing an at-risk customer (false negative) is
usually costlier than a false alarm.

## Visualizations
All saved to `outputs/`:
- `churn_factors.png` — churn by contract type, tenure, charges, internet service
- `correlation_heatmap.png` — numerical features vs churn
- `confusion_matrix_logistic_regression.png`, `confusion_matrix_random_forest.png`, `confusion_matrix_xgboost.png`
- `roc_curve_comparison.png` — ROC curves for all three models on one chart
- `feature_importance.png` — top 15 features (Random Forest)

## Model Interpretation
`outputs/logreg_coefficients.csv` lists every feature's Logistic
Regression coefficient with its direction of effect (increases/decreases
churn likelihood). `outputs/feature_importance.csv` gives the Random
Forest's feature-importance ranking. Both agree that **tenure, contract
type, and monthly/total charges** are the strongest churn signals.

## Conclusion
Tree-based (Random Forest, XGBoost) and linear (Logistic Regression)
models all identify the same core churn drivers: short tenure,
month-to-month contracts, high monthly charges, and lack of add-on
services like tech support or online security. Logistic Regression edges
out on ROC-AUC and is the most interpretable; Random Forest catches more
actual churners (higher recall) at the cost of more false alarms — the
right choice depends on the business's tolerance for false positives
versus false negatives.

## Future Improvements
- Handle class imbalance more explicitly with SMOTE or threshold tuning.
- Hyperparameter-tune each model (GridSearchCV/RandomizedSearchCV).
- Add customer-support interaction data or usage trend features.
- Build a simple Streamlit/Flask dashboard for real-time churn scoring.
- Track model performance over time and retrain periodically (concept drift).

## How to Run the Project

```bash
# 1. (Optional) create a virtual environment
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) regenerate the dataset
python src/generate_dataset.py

# 4. Run the full pipeline
python src/train_model.py

# 5. (Optional) explore interactively
jupyter notebook notebooks/churn_prediction.ipynb
```

All plots will be saved to `outputs/`, and predictions will be saved to
`predictions.csv` in the project root.
