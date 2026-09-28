# Bank Customer Churn Prediction: Gradient Boosting vs Neural Networks

Predicts which retail-bank customers are likely to leave and compares five models: logistic regression, gradient boosting, and three tuned neural networks (MLP, MLP with BatchNorm, Wide & Deep). SQL cleans and segments the data, statistical tests check what drives churn, and models are selected on a validation set and scored once on an untouched test set. **Gradient boosting came out on top**, and the neural networks were competitive but did not beat it.

> **Data note:** the data is **simulated** (seed 42, 10,000 customers, with 100 duplicate rows and about 3% missing age and balance values added on purpose). Churn is drawn from a known process, so the results show the method working, not facts about a real bank. To use real data, replace the `customers_raw` table with your own table using the same columns.

## Business questions
1. Which customer segments churn most?
2. Are complaints and tenure statistically associated with churn?
3. Can a model rank customers by risk well enough to target retention offers, and does a neural network add anything over simpler models?

## Approach
1. **SQL (SQLite):** de-duplicate with a window function, build a feature view, and compute churn rate by segment.
2. **Hypothesis tests:** chi-square (complaints vs churn) and Mann-Whitney U (tenure, churned vs retained).
3. **Split:** 60/20/20 stratified train, validation, and test (6,000 / 2,000 / 2,000 customers).
4. **Preprocessing (fit on train only):** median imputation with missing-value indicators, standard scaling, one-hot region, log balance.
5. **Models:** logistic regression and HistGradientBoosting tuned with 5-fold CV on the training set; MLP, MLP + BatchNorm, and Wide & Deep networks tuned with a grid over units, dropout, and learning rate, using early stopping on validation AUC.
6. **Selection:** best model chosen on **validation AUC**, then all models scored on the **test set**.

## Findings from SQL segmentation
| Segment | Churn rate |
|---|---|
| Customers with a complaint vs none | 21.4% vs 9.3% |
| 3-4 products held vs 2 products | 19.0-20.2% vs 8.8% |
| Low vs high digital engagement | 15.4% vs 9.6% |
| Region | 11.9-12.6% (no meaningful difference) |

Overall churn rate: 12.4%. Customers who complained have 2.63 times the odds of churning (chi-square = 251.3, p < 0.001). Churned customers have a shorter median tenure (52 vs 62 months, Mann-Whitney p < 0.001). These are associations in simulated data, not causal claims.

## Model results

| Model | Val AUC | Test AUC | Test avg. precision | Test Brier | Test top-decile lift |
|---|---|---|---|---|---|
| **HistGradientBoosting** | **0.696** | **0.688** | **0.296** | **0.101** | **2.78x** |
| Wide & Deep | 0.693 | 0.673 | 0.277 | 0.102 | 2.66x |
| MLP + BatchNorm | 0.692 | 0.668 | 0.253 | 0.104 | 2.54x |
| MLP | 0.688 | 0.677 | 0.283 | 0.102 | 2.50x |
| Logistic regression | 0.687 | 0.671 | 0.289 | 0.102 | 2.58x |

Top-decile lift is the churn rate among the 10% of customers scored highest, divided by the overall churn rate (12.4%).

Best settings: gradient boosting used learning rate 0.1, max depth 2, 100 iterations, min 50 samples per leaf. All three networks preferred 32 units, dropout 0.3, and learning rate 0.003. Logistic regression used C = 0.01.

<img width="1691" height="440" alt="model_comparison" src="https://github.com/user-attachments/assets/beda94ab-68da-479b-845c-c896bfdb39d1" />
Model Comparison

**What the best model delivers (test set):** the riskiest 10% of customers churned at 34.5%, and contacting the riskiest 30% would reach 52% of all churners. Permutation importance ranks complaints first, followed by number of products, tenure, balance, and age.

## Reading the results
The gap between models is small. With about 250 churners in the test set, an AUC estimate carries roughly +/-0.02 of uncertainty, so the ranking among the top models could change with a different split or seed. The safe conclusion is that gradient boosting is at least as good as the neural networks on this tabular data, and simpler to train, tune, and explain.

## Limitations
- Simulated data with a clean structure; real churn data will be noisier.
- One split and one random seed; no repeated cross-validation for the final comparison.
- No cost-benefit analysis of retention offers, which is needed to choose a score cutoff. The risk bands in `score_customers()` are illustrative.

## Run it
```bash
pip install -r requirements.txt
jupyter notebook bank_churn_prediction.ipynb
```
Set `QUICK=1` in the environment (or `QUICK = True` in the config cell) for a fast test run. Trained models and a config file are saved to `saved_models/`, and `score_customers()` scores new customers with the best model.

## Repository contents
- `bank_churn_prediction.ipynb`: full pipeline (data, SQL, tests, split, tuning, evaluation, saving, scoring)
- `reports/model_comparison.png`: ROC curves, network learning curves, and feature importance
- `reports/model_results.csv`: model metrics
- `reports/segment_churn.csv`: churn rate by segment

**Tools:** Python, SQL (SQLite), pandas, SciPy, scikit-learn, TensorFlow/Keras, matplotlib
