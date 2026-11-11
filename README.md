# Airbnb Prediction Challenges
### STAT 303-3 · Data Science with Python III · Northwestern University

**Placed 9th/156**

This project explored two prediction problems using Airbnb listings in **Chicago, Jersey City, and Washington, DC**: predicting nightly price and identifying whether a listing's host is a Superhost.

I worked with messy listing attributes, missing values, categorical fields, dates, amenities, and text-derived features. The final approaches were an **XGBoost routing system for price prediction** and a **CatBoost classifier for Superhost probabilities**.

[Regression competition](https://www.kaggle.com/competitions/prediction-problem-2026-regression) · [Classification competition](https://www.kaggle.com/competitions/prediction-problem-2026-classification)

## The tasks

| Track | Response | Evaluation | Submission |
| --- | --- | --- | --- |
| Regression | Nightly listing price | Mean absolute error (MAE), in dollars | Listing ID and predicted price |
| Classification | Superhost status | ROC-AUC | Listing ID and predicted probability |

Each task supplied 57 columns in training data and 56 in test data, with the response withheld from test data. The coursework required reproducible Python submissions and allowed models covered in the STAT 303 sequence.

## Feature preparation

The saved code includes:

- Cleaning currency and percentage strings and mapping binary fields to numbers.
- Turning dates into host tenure and review-age features, using 2026 as the reference year.
- Counting amenities and identifying 18 amenity categories.
- Extracting bathroom counts and shared/private bathroom indicators.
- Summarizing text through length and word counts.
- Constructing capacity and review-activity ratios.
- Adding missingness indicators and handling infinite values.
- Encoding categorical columns for XGBoost and supplying categorical features directly to CatBoost.

The final implementations differ in how they handle text and categories; see the [methods notes](docs/methods-and-evaluation.md).

## Regression: modeling different price regimes

The final regression report described unusually high hotel-like listing prices around $40,000 and $50,000. Rather than fitting one regressor to every listing, the saved implementation routes predictions:

1. Identify hotel-like listings using property type and listing-name terms.
2. Predict non-hotel prices with an XGBoost regressor.
3. For hotels, classify whether the price falls into the high-price regime.
4. Regress ordinary hotel prices, or classify high-price listings into $40,000/$50,000 outputs.

The final code uses **two XGBoost regressors and two XGBoost classifiers**. Its price-derived high-price flag is removed from predictor inputs; the high-price labels are used as supervised training targets.

This is a competition-specific approach. The report's “placeholder” explanation for extreme prices is a modeling interpretation, not independently verified listing semantics.

## Classification: selecting CatBoost

The final classification report describes comparing XGBoost, CatBoost, and weighted combinations. It reports choosing CatBoost alone over the attempted ensemble.

The saved final classifier uses **3,000 iterations, learning rate 0.03, depth 6, and L2 regularization 5**, then exports positive-class probabilities. This matches the task's probability-ranking objective.

The final submission notebook trains on all supplied labeled rows. Tuning and validation experiments are described in the report rather than included in that final fitting notebook.

## Result and evidence

I placed **9th out of 156**. This ranking is my confirmed competition result; the live leaderboard was not accessible during this portfolio review.

No exact leaderboard MAE or ROC-AUC is quoted because a verified score was not recovered from the reviewed final files. The assignment's grading thresholds are not my achieved scores.

## Explore the repository

| File | Purpose |
| --- | --- |
| [Price regression notebook](regression/final-code.ipynb) | Preserved final code cells, with saved outputs removed |
| [Superhost classification notebook](classification/final-code.ipynb) | Preserved final code cells, with saved outputs removed |
| [Methods and evaluation](docs/methods-and-evaluation.md) | Model details, validation limits, and interpretation |
| [Source and contribution notes](docs/source-and-contributions.md) | File provenance and what remains to confirm |
| [requirements.txt](requirements.txt) | Dependency list inferred from the code |

## Running the code

Obtain the authorized data for each competition. Place that track's `train.csv` and `test.csv` beside its notebook, install the dependencies in a suitable Python environment, and open the notebook with Jupyter. Each writes `submission.csv` in its own directory.

The data are not bundled. The dependency list is not a recovered version lockfile. Code cells were checked for Python syntax, but no model was rerun and end-to-end reproduction has not been verified in this repository.

## Portfolio scope

This repository documents completed coursework. The original modeling code has not been redesigned. Notebook outputs, execution counts, and incidental metadata were cleared; instructional template prose was removed while code cells were preserved.

The retrospective documentation was prepared with AI assistance. Original assistance and collaboration boundaries still need a fuller contribution review before making stronger claims of independent implementation or adapting this project for a resume.
