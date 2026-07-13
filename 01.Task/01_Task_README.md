### If ile does not render then click here:
[![View Notebook](https://img.shields.io/badge/View-Task%201-blue?logo=jupyter)](https://nbviewer.org/github/aDiL-Devlab/devhub-tasks2/blob/main/01.Task/01_Task.ipynb)

# Task 1 (Advanced): Term Deposit Subscription Prediction

## Objective
Predict whether a bank customer will subscribe to a term deposit as a result of a marketing campaign, and explain individual model predictions using SHAP.

## Approach

**Data Loading and Inspection**
- Loaded Bank Marketing dataset (`bank-full.csv`, UCI) with 45,211 rows and 17 columns
- No missing values or duplicate rows found
- "unknown" values in columns like `job`, `education`, and `poutcome` act as hidden missing data
- `pdays = -1` and `poutcome = unknown` both indicate customers never contacted in a previous campaign
- Target variable `y` (renamed `subscribe`) is highly imbalanced: only 11.7% subscribed

**EDA (Univariate, Bivariate, Multivariate)**
- Univariate: age concentrated 30-45, balance heavily right-skewed with outliers up to 100K+, campaign/duration have extreme outliers
- Bivariate: identified a base-rate bias risk when comparing raw counts across imbalanced groups (e.g. `default`) — resolved using percentage-based comparison (`pd.crosstab(normalize='index')`) instead of raw counts
- Verified `pdays` percentile behavior using `groupby().describe()` rather than reading estimates off a chart
- Multivariate: correlation heatmap showed `duration` most correlated with target (0.39) but flagged for leakage; `pdays` and `previous` correlated with each other (0.45); scatter plots (age vs balance, pdays vs previous) showed no clear multivariate separation between subscribers and non-subscribers

**Feature Selection**
- Numerical features assessed via correlation with target
- Categorical features assessed via subscribe-rate spread across categories (`groupby().mean()`)
- Only `duration` dropped, due to data leakage (call length is unknown before a call takes place)
- All other features retained — none showed a completely flat/zero relationship with the target

**Feature Engineering**
- Created `was_contacted_before` binary flag from `pdays` and `previous`, capturing whether a customer had any prior campaign contact
- Applied StandardScaler to `age`, `balance`, `day`, `campaign`
- Label-encoded ordinal/binary features: `education`, `month`, `housing`, `loan`, `default`
- One-hot encoded nominal features: `marital`, `contact`, `poutcome`
- Grouped rare high-performing `job` categories (`student`, `retired`) separately from the rest to avoid excessive dummy columns

**Modeling**
- Trained Logistic Regression, Decision Tree, and Random Forest classifiers
- Evaluated using Confusion Matrix, F1-Score, and ROC Curve / AUC
- Key metric: Recall (class 1), since missing an actual subscriber (false negative) is costlier than an unnecessary call (false positive)

**Model Explainability (SHAP)**
- Used `shap.TreeExplainer` on the Random Forest model
- Generated a global summary plot showing `contact_unknown`, `housing`, and `poutcome_success` as the strongest overall drivers of predictions
- Generated waterfall plots explaining 5 individual customer predictions, showing how each feature pushed that customer's probability up or down from the baseline

## Results and Insights

| Model | Accuracy | Precision | Recall | F1-Score | AUC |
|---|---|---|---|---|---|
| Logistic Regression | 69.1% | 22.9% | 66.0% | 0.340 | 0.723 |
| Decision Tree | 73.0% | 24.1% | 57.9% | 0.341 | 0.723 |
| Random Forest | 76.8% | 28.2% | 59.9% | 0.413 | **0.774** |

- Logistic Regression had the highest raw recall at the default threshold, but Random Forest had the best AUC and F1-Score, indicating stronger overall discriminative power across all thresholds
- **Random Forest was selected as the final model**, since its higher AUC suggests better potential recall once the classification threshold is tuned in a future improvement stage
- Threshold tuning (0.5 → 0.2) on Logistic Regression showed the classic precision-recall trade-off: recall rose to 96.8% but accuracy collapsed to 23.8%, with false positives exploding — confirming that maximizing recall alone is not practical for this business case
- SHAP analysis confirmed `contact_unknown` as a recurring dominant factor pulling predictions toward "no" across multiple individual customers, along with `housing` and `poutcome_success` as consistently influential features globally

## Next Steps (Improvement Stage — Pending)
- Threshold tuning on Random Forest to balance recall and false positive rate
- Potential use of SMOTE to address class imbalance further
