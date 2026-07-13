## If File does'nt render than click Here:
[![View Notebook](https://img.shields.io/badge/View-Task%203-blue?logo=jupyter)](https://nbviewer.org/github/aDiL-Devlab/devhub-tasks2/blob/main/03.%20Task/03_Task.ipynb)

# Task 3: Loan Default Risk with Business Cost Optimization

## Objective
Predict the likelihood of a loan default and optimize the decision threshold based on cost-benefit analysis, rather than relying on default accuracy-based classification.

## Dataset
Due to GitHub's file size limits, the dataset is not included in this repository.
Download it from: https://www.kaggle.com/c/home-credit-default-risk/data
- file name: `application_train.csv`

## Approach

**Data Loading and Inspection**
- Loaded Home Credit Default Risk dataset (`application_train.csv`) with 307,511 rows and 122 columns
- No duplicate rows found
- 67 columns contained missing values, ranging from under 1% to nearly 70%
- Target variable is highly imbalanced (~8% defaulters)

**Missing Value Cleanup**
- Removed 33 redundant building-related columns (`_MODE`/`_MEDI` variants), keeping only the `_AVG` version since all three measured the same underlying information
- Dropped 13 additional building `_AVG` columns after correlation with TARGET showed negligible predictive value
- Reserved 9 weak-but-potentially-useful columns (phone/email flags, region-mismatch flags) in a separate dataframe (`df_reserved`) instead of dropping them outright, to reuse in feature engineering
- Filled remaining missing values using column-appropriate strategies: median for skewed numeric columns (e.g. `EXT_SOURCE_1/2/3`, `OWN_CAR_AGE`), "Unknown"/mode for categorical columns, and 0 for count-type columns (bureau inquiries, social circle counts) where missing meant "no record"

**Exploratory Data Analysis**
- Univariate analysis on numeric and categorical columns to understand distributions
- Bivariate analysis against TARGET revealed `EXT_SOURCE_2`, `EXT_SOURCE_3`, and `DAYS_BIRTH` (age) as the clearest signals
- Discovered raw category counts were misleading (e.g. "House/apartment" appeared riskiest by count) — switching to default-rate-by-group revealed the true risk pattern (e.g. "Unemployed" and "Rented apartment" applicants are actually riskiest)
- Multivariate correlation heatmap identified several multicollinear groups: loan amount columns (`AMT_CREDIT`, `AMT_ANNUITY`, `AMT_GOODS_PRICE`), region rating columns, city-mismatch flags, social circle columns, and bureau inquiry columns

**Feature Engineering**
- Fixed a known data anomaly in `DAYS_EMPLOYED` (placeholder value 365243 treated as missing, then imputed)
- Converted `DAYS_BIRTH`/`DAYS_EMPLOYED` into readable `AGE_YEARS` and `EMPLOYED_YEARS`
- Replaced redundant loan-amount columns with ratio features: `CREDIT_INCOME_RATIO`, `ANNUITY_INCOME_RATIO`, `CREDIT_GOODS_RATIO`
- Combined 3 city-mismatch flags into `TOTAL_CITY_MISMATCH`
- Combined 6 bureau inquiry columns into `TOTAL_BUREAU_INQUIRIES`
- Combined reserved contact-info flags into `TOTAL_CONTACT_INFO`
- Combined 20 document-submission flags into `TOTAL_DOCUMENTS_SUBMITTED`
- Removed duplicate columns identified via correlation (e.g. `REGION_RATING_CLIENT`, one of the social circle pairs)

**Preprocessing**
- Applied Label Encoding on binary columns (contract type, gender, car/realty ownership)
- Applied One-Hot Encoding on nominal categorical columns (income type, family status, housing type, etc.)
- Applied Ordinal Encoding on `NAME_EDUCATION_TYPE` using its natural order
- Applied Frequency Encoding on high-cardinality columns (`OCCUPATION_TYPE`, `ORGANIZATION_TYPE`)
- Scaled only genuinely continuous columns (income, credit, days, counts), leaving binary and already-normalized columns (e.g. `EXT_SOURCE_*`) untouched
- Performed scaling after the train-test split to avoid data leakage

**Modeling**
- Trained Logistic Regression and CatBoost, both with class weighting to address the ~8% default rate imbalance
- Evaluated using precision, recall, F1-score, and ROC-AUC, with particular attention to recall on the default class
- Defined business costs: False Negative (missed default) weighted 10x higher than False Positive (wrongly flagged safe customer), since a missed default costs the bank its principal, while a false alarm only costs potential profit
- Swept decision thresholds from 0.05 to 0.95 and computed total business cost at each point to find the cost-minimizing threshold, instead of using the default 0.5 cutoff blindly

## Results and Insights

- `EXT_SOURCE_1/2/3` (external credit scores) are the strongest individual predictors of default, consistent across univariate, bivariate, and multivariate analysis
- Younger applicants default more often than older applicants
- Employment type, education level, and housing type show meaningful default-rate differences once raw counts are corrected for group size
- The bureau-inquiry and social-circle features had weak linear correlation with TARGET but were retained since tree-based models can capture non-linear patterns that correlation misses

| Model | Threshold | FN | FP | Total Cost | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.50 (optimal) | 1,621 | 17,570 | 33,780 | 0.746 |
| **CatBoost** | 0.50 (optimal) | 1,815 | 14,457 | **32,607** | **0.758** |

- Both models produced a clear U-shaped cost curve across thresholds, confirming that neither too-low nor too-high a threshold minimizes business cost
- Interestingly, the cost-optimal threshold for both models coincided with the standard 0.5 cutoff for this dataset and cost ratio
- CatBoost outperformed Logistic Regression on every metric that matters for this task — lower total business cost, higher ROC-AUC, and better precision at a comparable recall — making it the recommended model
- This shows that framing the problem around business cost (rather than accuracy alone) directly changes how model quality should be judged, and that the cheapest model in dollar terms is not necessarily the one with the highest raw accuracy
