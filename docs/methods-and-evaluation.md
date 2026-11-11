# Methods and evaluation

[Back to overview](../README.md)

## Sources

This summary uses the saved final regression/classification reports, their final code notebooks, and the competition descriptions supplied during the portfolio review. The Kaggle pages and live leaderboard could not be read directly.

Both datasets concern Airbnb listings in Chicago, Jersey City, and Washington, DC. Regression uses price as the response and MAE as the metric; classification uses Superhost status and ROC-AUC. Public and private leaderboards score different portions of held-out test labels.

## Regression implementation

The final code cleans price, rates, binary fields, dates, amenities, and bathroom text. It creates amenity counts/indicators, text length/word counts, ratios, missingness indicators, and a hotel-like routing flag.

Raw descriptions, biographies, amenities strings, listing names, neighborhood text, and host-verification text are removed after feature extraction. Remaining categorical columns are one-hot encoded using concatenated training/test predictor frames to align categories. This uses test feature categories, not test labels, and should be disclosed when interpreting the preprocessing design.

The final fitted components are:

| Component | Estimators | Learning rate | Max depth |
| --- | ---: | ---: | ---: |
| Non-hotel price regressor | 2,500 | 0.02 | 5 |
| Ordinary hotel price regressor | 1,000 | 0.03 | 3 |
| High-price hotel classifier | 800 | 0.03 | 3 |
| $40,000/$50,000 classifier | 500 | 0.03 | 2 |

The code defines the high-price hotel training subset using price ≥ $30,000. A probability threshold of 0.5 selects the high-price branch at prediction time. Final predictions are clipped to be nonnegative.

This routing approach depends on the competition's observed price regimes. It should not be presented as a general production pricing system or evidence of economic willingness to pay.

## Classification implementation

The code maps Superhost status to the target, removes it from predictors, and engineers the same broad categories of listing features. CatBoost receives object columns as categorical features, with missing categories converted to “Missing.”

Unlike regression, the final classification code does not drop all raw text columns after extracting summary features; remaining object fields are passed as categorical values. The report's simplified description should be read alongside the actual saved implementation.

The final classifier specifies Logloss training loss, AUC evaluation, 3,000 iterations, learning rate 0.03, depth 6, L2 regularization 5, and seed 12. The exported score is `predict_proba(... )[:, 1]`, not a hard label.

## Evaluation evidence

The reports describe prior model comparisons and tuning. The final notebooks follow the submission requirement to fit already-selected models on the complete training set; they do not contain cross-validation or holdout evaluation code.

Therefore:

- No numerical validation score is inferred from the final fitting notebook.
- No specific fold count, exhaustive search count, or improvement over a baseline is asserted without more evidence.
- A ranking is not equivalent to a population-level performance guarantee.
- Repeated public-leaderboard feedback can influence model selection.
- Listings from the same host may create dependence; a random row split would not establish generalization to unseen hosts.
- Date features use a fixed 2026 reference and may not represent a prospective deployment time.
- Superhost-related predictors need timing and target-definition checks before stronger real-world forecasting claims.

These are interpretation notes, not changes to the project. No training, tuning, data repair, or score reproduction was performed for this repo.

## What the result supports

The 9th-of-156 finish is a user-confirmed course competition result. Exact public/private leaderboard status and the achieved numeric score were not independently recovered. Neither the MAE grading thresholds nor the ROC-AUC grading thresholds should be substituted for an achieved score.
