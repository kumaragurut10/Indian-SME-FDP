# SME Financial Distress Prediction (SME_FDP)

A research notebook that builds and validates one-year-ahead financial distress prediction models for small and medium enterprises (SMEs), using accounting-ratio predictors, an Altman Z″-based distress label, and a comparison of Logistic Regression, Decision Tree, and LightGBM classifiers — including out-of-time validation, calibration, bootstrap significance testing, and SHAP-based explainability.

## Overview

Starting from a panel of SME financial-ratio data (6,061 firm-year observations, 29 columns), the notebook:

1. Constructs an **Altman Z″ score** and a three-zone distress classification (Distress / Grey Zone / Safe) from the raw accounting data.
2. Builds a **one-year-ahead target**: predictors from year *t* are mapped to the distress label observed in year *t+1*.
3. Performs a **multicollinearity audit** (VIF + correlation) to derive three nested predictor sets **FULL** (21 features), **REDUCED** (17 features), and **STRICT** (13 features).
4. **Winsorizes** predictors at the 1st/99th percentile using training-data-only limits to control outliers without leakage.
5. Splits the data **temporally** into train (2016–2020), validation (2021–2022), test (2023–2024), and out-of-time/OOT (2025) periods.
6. Trains and tunes **Logistic Regression, Decision Tree, and LightGBM** models on each of the three feature sets (nine model/feature combinations).
7. Selects a final **operational decision threshold** and evaluates all models on the independent test set and the OOT set.
8. Runs **paired firm-level bootstrap tests**, **probability calibration** (intercept/slope), and benchmarks against a **naïve persistence** baseline.
9. Performs a **proxy-exclusion sensitivity analysis**, removing accounting ratios that act as near-direct proxies for the Altman label (e.g., Current ratio, TOL/TNW) and retrains a proxy-free STRICT LightGBM model, comparing it against a lagged Altman Z″ benchmark.
10. Explains model behavior with **SHAP** (global importance, beeswarm, dependence plots, rank-change analysis) for both the original and proxy-free models.

## Data

- **Input file:** `Financial SME Data.xlsx` (not included in this repository — see [Data](#data-access) below)
- **Shape:** 6,061 firm-year rows × 29 columns (`Company Name`, `year`, and 27 financial ratios/figures such as Return on Equity, Net Profit Margin, DSCR, Debt to Equity, Current/Quick Ratio, CFO-based ratios, Net Worth, Working Capital, TOL/TNW, etc.)
- **Target:** `Target_Distress` a binary flag derived from the Altman Z″ zone one year ahead of the predictor year, built from a leak-safe `Predictor_Year → Target_Year = Predictor_Year + 1` merge.
- **Temporal splits:**

  | Split | Years | Observations | Distress rate |
  |---|---|---|---|
  | Train | 2016–2020 | — | — |
  | Validation | 2021–2022 | — | — |
  | Test | 2023–2024 | 1,102 | 14.16% |
  | Out-of-time (OOT) | 2025 | 551 | 13.97% |

### Data access

The underlying SME financial dataset is proprietary/private and is **not** included in this repository. To run the notebook end-to-end you will need to supply your own data file with the same schema and update the `FILE_PATH` variable in the first cell.

## Methodology

### Feature sets
Three nested predictor sets were derived from a training-data-only VIF/correlation audit:

| Set | # Features | Notes |
|---|---|---|
| FULL | 21 | All candidate accounting-ratio predictors |
| REDUCED | 17 | Drops 4 redundant predictors (e.g., PBT %, PAT %, Quick ratio, Current asset/Total assets) |
| STRICT | 13 | Drops 4 further redundant predictors (e.g., Debt-to-equity, CFO/Total assets, PBDITA %, Net worth/Total assets) |

### Models
- **Logistic Regression**
- **Decision Tree**
- **LightGBM** (gradient-boosted trees)

Each model is trained on each of the three feature sets (9 model × feature-set combinations), with separate preprocessing pipelines for raw and winsorized data.

### Evaluation
Metrics reported throughout: **ROC-AUC, PR-AUC, Brier score, Log loss, Accuracy, Sensitivity, Specificity, Precision, F1-score**, plus confusion-matrix counts (TN/FP/FN/TP).

### Robustness checks
- Paired firm-level **bootstrap** resampling for statistical comparison of models
- **Calibration** diagnostics (intercept and slope) on OOT probabilities
- Comparison against a **naïve persistence** baseline (assume last known status repeats)
- **Transition-specific** performance (e.g., new distress discrimination) analysis
- **Proxy-exclusion sensitivity**: removing accounting ratios that mechanically echo the Altman label, retraining, and comparing against a lagged Altman Z″ benchmark and an "Altman + proxy-excluded" incremental model

### Explainability
**SHAP** is used throughout for global feature importance, beeswarm plots, dependence plots, and rank-change comparisons between the original STRICT model and the proxy-free STRICT model.

## Key results

Independent test set (1,102 observations, 14.16% distress rate):

| Feature set | ROC-AUC | PR-AUC | Accuracy | Sensitivity | Specificity | F1 |
|---|---|---|---|---|---|---|
| FULL | 0.9447 | 0.7301 | 0.8956 | 0.7692 | 0.9165 | 0.6761 |
| REDUCED | 0.9434 | 0.7310 | 0.8984 | 0.7692 | 0.9197 | 0.6818 |
| STRICT | 0.9430 | 0.7465 | 0.8938 | 0.7500 | 0.9175 | 0.6667 |

Out-of-time (OOT) set (551 observations, 2025, 13.97% distress rate):

| Feature set | ROC-AUC | PR-AUC | Accuracy | Sensitivity | Specificity | F1 |
|---|---|---|---|---|---|---|
| FULL | 0.9357 | 0.7733 | 0.8966 | 0.7922 | 0.9135 | 0.6816 |
| REDUCED | 0.9359 | 0.7664 | 0.8929 | 0.7922 | 0.9093 | 0.6740 |
| STRICT | 0.9370 | 0.7748 | 0.8966 | 0.7792 | 0.9156 | 0.6780 |

Proxy-free sensitivity (11 predictors, after removing Current ratio and TOL/TNW — a 15.4% reduction from the 13-predictor STRICT set):

| Model | ROC-AUC | PR-AUC |
|---|---|---|
| Proxy-excluded STRICT (LightGBM) | 0.8971 | 0.6339 |
| Altman + Proxy-excluded (incremental) | 0.9424 | 0.7906 |

All models substantially outperform the lagged Altman Z″ score and the naïve persistence baseline on OOT data, and the STRICT feature set performs competitively with FULL/REDUCED despite using fewer predictors.

## Repository contents

```
.
├── SME_FDP.ipynb     # Main analysis notebook (data prep, modeling, validation, SHAP, figures)
└── README.md
```

Running the notebook also produces (not checked into version control by default):

- `Financial_Distress_Research_Tables.xlsx` — consolidated results tables
- `Research_Figures/` — 14 figure files (PNG + TIFF), including:
  - Annual distress prevalence
  - Temporal data partition
  - Distress transition distribution
  - Original vs. proxy-excluded SHAP comparison
  - Proxy-excluded SHAP importance
  - SHAP rank changes
  - Incremental predictive performance

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
statsmodels
lightgbm
shap
openpyxl
joblib
```

Install with:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels lightgbm shap openpyxl joblib
```

(The notebook's first `%pip install` cell installs the core subset automatically; `statsmodels` and `lightgbm` are required additionally for the VIF audit and gradient boosting steps.)

## Usage

1. Clone this repository and install the requirements above.
2. Place your SME financial dataset (matching the expected 29-column schema) at a path of your choice.
3. Open `SME_FDP.ipynb` and update the `FILE_PATH` variable in the first cell to point to your data file.
4. Run all cells in order — the notebook is organized sequentially from data import through final figure generation, and later sections depend on variables created earlier (feature sets, splits, trained models).
5. Generated tables and figures will be written to the output directories referenced in the final cells (update `OUTPUT_DIR` / `OUTPUT_FILE` as needed for your environment).

## Notes

- The notebook was developed and run locally (Windows paths such as `D:\...` and `C:\Users\...` appear in the source) — update paths for your own environment before running.
- All winsorization limits, feature-selection decisions, and preprocessing statistics are computed on the **training split only** to avoid data leakage into validation/test/OOT periods.
- The proxy-exclusion analysis exists specifically to test whether model performance is being driven by predictors that are near-mechanical restatements of the Altman Z″ formula itself, rather than genuine incremental signal.

## License

Add a license of your choice (e.g., MIT) if you intend to share this repository publicly.
