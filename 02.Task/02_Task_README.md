# Task 2: Customer Segmentation Using Unsupervised Learning

## Objective
Cluster mall customers based on their spending habits and propose tailored marketing strategies for each identified segment.

## Approach

**Data Loading and Inspection**
- Loaded Mall Customers dataset with 200 rows and 5 columns
- No missing values or duplicate rows found
- Dropped CustomerID (unique identifier, no predictive value)

**Exploratory Data Analysis**
- Age distribution was fairly spread with no strong skew; Annual Income was roughly symmetric with a few high-value outliers
- Age showed a negative correlation with Spending Score (-0.33): younger customers tend to spend more
- Annual Income showed no linear correlation with Spending Score, which turned out to be misleading — the relationship exists but is non-linear, later revealed through clustering

**Preprocessing**
- Applied StandardScaler on Age, Annual Income, and Spending Score before clustering, since K-Means relies on distance and unscaled features (especially Income, with the widest range) would dominate the results
- Gender was excluded from clustering since the objective was spending-behavior segmentation, not gender-based segmentation

**Finding Optimal Clusters**
- Compared three methods to select the number of clusters (K): Elbow Method, Silhouette Score, and Davies-Bouldin Index
- Elbow suggested K=4, Silhouette suggested K=6, Davies-Bouldin suggested K=7 (K=5 as close second)
- Since the three metrics disagreed, K=5 was chosen as a business-friendly balance that performed reasonably well across all three

**Modeling**
- Trained K-Means (init='k-means++') with K=5 on the scaled features
- Used groupby to calculate mean Age, Income, and Spending Score per cluster, then labeled each cluster based on its profile

**Visualization**
- Applied PCA to reduce 3 features to 2 components for visualization, capturing 77.6% of total variance (PC1: 44.3%, PC2: 33.3%)
- Scatter plot confirmed clusters were reasonably well separated, with Premium Customers forming the most distinct group

## Results and Insights

| Cluster | Segment | Avg Age | Avg Income | Avg Spending Score |
|---|---|---|---|---|
| 0 | Budget Customers | - | Low | Low |
| 1 | Young Spenders | ~25 | Low-Mid | High (2nd highest) |
| 2 | Premium Customers | ~32 | High | High |
| 3 | Cautious Savers | - | High | Low |
| 4 | Average Customers | ~55 | Mid | Mid |

- Premium Customers (high income, high spending) need no discounting — focus on VIP loyalty programs instead
- Cautious Savers (high income, low spending) are the highest-potential untapped segment — best targeted with curated recommendations and trust-building offers
- Young Spenders (lower income, high spending, young age) respond well to discounts and trend-driven promotions
- Budget Customers (low income, low spending but willing to spend) respond well to deep discounts and value bundles
- Average Customers get standard seasonal promotions

## Key Takeaway
Correlation alone (Annual Income vs Spending Score ≈ 0) suggested no relationship, but clustering revealed clear behavioral groups hidden in the data. This shows that unsupervised learning can surface structure that simple linear metrics miss, and that combining statistical validation (Elbow, Silhouette, Davies-Bouldin) with business judgment produces more actionable segments than relying on a single metric.
