# Sonar Rock vs Mine Classification

A machine learning project that classifies sonar signals as either a **Rock (R)** or a **Mine (M)** using Logistic Regression, with model tuning, proper evaluation, and explainable AI (SHAP) to interpret predictions.

## Overview

Sonar signals bounced off an object on the sea floor return 60 energy measurements across different frequency bands. This project trains a classifier to distinguish whether the object reflecting the signal is a rock or a metal mine — a real-world binary classification problem historically studied in the [UCI Sonar dataset](https://archive.ics.uci.edu/dataset/151/connectionist+bench+sonar+mines+vs+rocks).

## Dataset

- **208 samples**, 60 numerical features per sample (energy in a frequency band, normalized 0–1), 1 label column (`R` = Rock, `M` = Mine)
- Source: UCI Machine Learning Repository — Connectionist Bench (Sonar, Mines vs. Rocks)

## Approach

1. **Data exploration** — class balance check, per-class mean energy across frequency bands
2. **Preprocessing** — `StandardScaler` to standardize all 60 features before training
3. **Model selection & tuning** — Logistic Regression tuned via `GridSearchCV` with 5-fold cross-validation over the regularization strength `C`
4. **Evaluation** — cross-validated accuracy, held-out test accuracy, precision/recall/F1 per class, confusion matrix, ROC curve and AUC
5. **Explainability** — [SHAP](https://github.com/shap/shap) (`LinearExplainer`) to show which frequency bands drive each prediction, both globally and for individual samples

## Results

| Metric | Score |
|---|---|
| Best cross-validated training accuracy (5-fold CV) | ~81.3% |
| Test accuracy | ~71–82% (varies with split, given only 208 samples) |
| Best hyperparameters | `C=0.01`, `solver='liblinear'` |

Cross-validated accuracy is reported as the primary metric since the dataset is small (208 samples) and a single train/test split has high variance. See the confusion matrix and ROC-AUC in the notebook for a fuller picture beyond a single accuracy number.

## Explainable AI

Logistic Regression is inherently interpretable, but with 60 correlated features it isn't obvious at a glance which frequency bands matter. This project uses **SHAP** to show:
- **Global importance** — which frequency bands most influence predictions across the whole test set
- **Local explanations** — a waterfall plot showing exactly why one specific sonar reading was classified as Rock or Mine

## Tech Stack

- Python, NumPy, pandas
- scikit-learn (Logistic Regression, GridSearchCV, StandardScaler, metrics)
- Matplotlib
- SHAP

## How to Run

1. Clone this repo and open `ML_Project.ipynb` in Google Colab or Jupyter
2. Upload `sonar data.csv` to the same environment (or update the file path in the notebook)
3. Run all cells in order — later cells depend on variables created earlier (scaling, tuning, model)
4. Install SHAP if running locally: `pip install shap`

## Project Structure

```
├── ML_Project.ipynb      # Main notebook: data prep, training, evaluation, SHAP
├── sonar data.csv         # Dataset (60 features + label)
└── README.md
```

## Possible Future Improvements

- Compare against other models (Random Forest, SVM, XGBoost) to benchmark Logistic Regression
- Expand the `C` hyperparameter grid to confirm the optimum isn't at the search boundary
- Add feature selection to see if a smaller subset of frequency bands retains most of the accuracy
- Wrap the model in a simple web app (Flask/Streamlit) for interactive predictions

## License

This project is open source and available for educational use.
