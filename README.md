# FertilityIQ

## AI-Powered IVF Research Platform

FertilityIQ is an end-to-end machine-learning research prototype for
exploring IVF-related clinical and embryological data. It combines data
analysis, model training, model evaluation, SHAP explainability, a
FastAPI backend, and an interactive React dashboard.

> **Research-use disclaimer:** FertilityIQ is an experimental research
> and portfolio project. It is not a medical device, diagnostic tool,
> clinical decision-support system, or treatment recommendation system.
> Its outputs must not be used to make patient-care decisions.

## Table of contents

-   [Project overview](#project-overview)
-   [Objectives](#objectives)
-   [Features](#features)
-   [Technology stack](#technology-stack)
-   [Architecture](#architecture)
-   [Repository structure](#repository-structure)
-   [Dataset](#dataset)
-   [Data preparation](#data-preparation)
-   [Machine-learning workflow](#machine-learning-workflow)
-   [Model results](#model-results)
-   [Explainability](#explainability)
-   [Backend API](#backend-api)
-   [Frontend](#frontend)
-   [Installation](#installation)
-   [Run locally](#run-locally)
-   [Troubleshooting](#troubleshooting)
-   [Limitations and responsible use](#limitations-and-responsible-use)
-   [Future work](#future-work)
-   [Licence and attribution](#licence-and-attribution)

## Project overview

FertilityIQ demonstrates a complete ML workflow, from dataset inspection
to a web application. It allows a user to inspect recorded model
performance, view global SHAP feature importance, and submit numeric
feature values to a trained model through an API.

The project is intended to demonstrate applied machine learning, data
science, explainable AI, backend API development, and frontend
integration. It does not establish clinical validity or improved IVF
outcomes.

## Objectives

1.  Explore and validate an IVF-related dataset.
2.  Perform exploratory data analysis.
3.  Prepare features and reproducible data splits.
4.  Compare Logistic Regression, XGBoost, and LightGBM.
5.  Evaluate a selected model on a held-out test split.
6.  Inspect model behaviour with SHAP.
7.  Serve inference and analytics through FastAPI.
8.  Present results in a React dashboard.

## Features

  -----------------------------------------------------------------------
  Dashboard section                   Description
  ----------------------------------- -----------------------------------
  Overview                            Dataset summary, validation model
                                      comparison, research pipeline, SHAP
                                      highlights

  Prediction                          Sends 15 numeric features to the
                                      prediction API and displays the
                                      encoded class and probabilities

  Model Analytics                     Displays recorded held-out test
                                      metrics for XGBoost

  Explainability                      Displays global mean absolute SHAP
                                      feature importance

  Research                            Documents dataset profile and
                                      methodological limitations
  -----------------------------------------------------------------------

The prediction form's initial values are demonstration inputs, not
recommended patient values.

## Technology stack

**Machine learning:** Python, Pandas, NumPy, scikit-learn, XGBoost,
LightGBM, SHAP, Matplotlib, Seaborn, Joblib.

**Backend:** FastAPI, Uvicorn, Pydantic, Pandas, Joblib.

**Frontend:** React, Vite, JavaScript, Recharts, Lucide React, CSS.

**Tools:** Visual Studio Code, PowerShell, Git, GitHub.

Check `requirements.txt` and `frontend/package.json` for the exact
dependency versions in the current checkout.

## Architecture

``` text
IVF research dataset
        |
        v
Data validation and exploratory analysis
        |
        v
Preprocessing and stratified data split
        |
        +----------------+----------------+
        |                |                |
        v                v                v
 Logistic Regression  XGBoost          LightGBM
        |                |                |
        +----------------+----------------+
                         |
                         v
                Validation comparison
                         |
                         v
                 Selected XGBoost model
                    /          \
                   v            v
            Test evaluation  SHAP analysis
                   \            /
                    v          v
                 Saved reports and model
                         |
                         v
                    FastAPI backend
                         |
              +----------+----------+
              |                     |
              v                     v
        Prediction API        Analytics APIs
              |                     |
              +----------+----------+
                         |
                         v
                 React + Vite UI
```

The inference service loads `ml/models/best_model.joblib`. The saved
estimator is a pipeline containing its own imputation step. Do not apply
the separate `preprocessor.joblib` to the input again before passing it
to this pipeline.

## Repository structure

The project is organized around the following areas. Exact filenames may
vary by revision.

``` text
FertilityIQ/
├── backend/
│   └── app/
│       ├── api/routes/
│       │   ├── health.py
│       │   ├── prediction.py
│       │   └── analytics.py
│       ├── core/
│       ├── schemas/
│       ├── services/
│       │   └── prediction_service.py
│       └── main.py
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── App.css
│   │   └── index.css
│   └── package.json
├── ml/
│   ├── data/
│   │   ├── raw/
│   │   ├── interim/
│   │   └── processed/
│   ├── preprocessing/
│   ├── training/
│   ├── evaluation/
│   ├── explainability/
│   └── models/
├── reports/
│   ├── eda/
│   ├── statistical/
│   ├── training/
│   ├── evaluation/
│   └── explainability/
├── research/
├── requirements.txt
├── .gitignore
└── README.md
```

Do not commit raw or processed datasets, local virtual environments,
`node_modules`, credentials, or identifiable patient information.

## Dataset

The working source was a Mendeley Data dataset titled **"Comprehensive
Dataset Profile of IVF Clinical and Embryological Data."**

The Excel file used in the reported workflow contained 5,002 rows and 21
columns. Two rows had missing target values, leaving 5,000 records with
a valid target.

Target column:

`Output(failed implantation or successful implantation)`

Recorded class counts:

  Encoded target     Records
  ---------------- ---------
  0                    2,508
  1                    2,492
  Total                5,000

**The semantic mapping of 0 and 1 has not yet been verified against an
authoritative codebook.** This README therefore uses "class 0" and
"class 1" rather than claiming which represents success or failure.

A separate CSV copy had different row counts, duplicate/missing-value
behaviour, and target formatting. The Excel file was used for the
experiments documented here; different copies should not be assumed
equivalent.

### Dataset licence and access

Download the dataset from its original Mendeley Data record and verify
the licence attached to the exact version you use. Different versions
may have different terms. The raw dataset is intentionally excluded from
this repository. Do not redistribute it unless its licence explicitly
permits redistribution.

## Data preparation

The model used these 15 numeric features:

1.  Female Age
2.  AMH
3.  Male Age
4.  Sperm count
5.  Sperm motility
6.  Sperm morphology
7.  Oocyte Maturity MII
8.  Oocyte Maturity MI
9.  Oocyte Maturity others
10. Number of fertilized eggs
11. Embryo Grade 1
12. Embryo Grade 2
13. Embryo Grade 3
14. Embryo Grade 4
15. Number of embryos

Fields excluded in this iteration included `β-HCG` (potential
post-outcome leakage), `Medicines and treatments` (numeric codes without
a verified codebook), `frezzing` (unclear definition/timing),
`Egg retrieval date`, and `Number of attempts`.

Rows with missing target values were excluded. Numeric feature missing
values were handled with median imputation in the model pipelines.

### Data split

The row-level stratified split was:

  Split          Records
  ------------ ---------
  Training         3,500
  Validation         750
  Test               750
  Total            5,000

A patient identifier and patient-level grouping were not established.
This is not evidence of patient-independent generalization.

Some embryo/fertilization features may only become available after parts
of the IVF process. The intended prediction timepoint has not been
established, so the model must not be described as a pre-treatment or
pre-transfer predictor.

## Machine-learning workflow

Three classifiers were trained:

-   Logistic Regression
-   XGBoost
-   LightGBM

The pipelines used median imputation. Logistic Regression also used
standard scaling. The tree-based models were trained without that
scaling step.

Model selection was based on validation ROC-AUC. XGBoost was selected in
the recorded experiment.

## Model results

### Validation comparison

  --------------------------------------------------------------------------------------
  Model          Accuracy   Precision    Recall   F1-score   ROC-AUC    PR-AUC     Brier
                                                                                   score
  ------------ ---------- ----------- --------- ---------- --------- --------- ---------
  Logistic         0.7493      0.7514    0.7433     0.7473    0.8416    0.8101    0.1634
  Regression                                                                   

  XGBoost          0.7827      0.7360    0.8797     0.8015    0.8675    0.8431    0.1427

  LightGBM         0.7813      0.7323    0.8850     0.8015    0.8629    0.8364    0.1441
  --------------------------------------------------------------------------------------

These are validation metrics used for model comparison, not final test
results.

### Held-out test evaluation: XGBoost

  Metric                  Result
  --------------------- --------
  Accuracy                0.7720
  Precision (class 1)     0.7231
  Recall (class 1)        0.8797
  F1-score (class 1)      0.7937
  ROC-AUC                 0.8601
  PR-AUC                  0.8391
  Brier score             0.1474

Confusion matrix (actual rows, predicted columns; class order `[0, 1]`):

               Predicted 0   Predicted 1
  ---------- ------------- -------------
  Actual 0             249           127
  Actual 1              45           329

These results describe one held-out row-level split. They do not
establish external validity, clinical utility, or performance on another
population or institution. Keep the test set separate from model
selection and threshold tuning.

## Explainability

SHAP was used to calculate global mean absolute SHAP feature importance
for the selected XGBoost model.

  Feature                       Mean absolute SHAP
  --------------------------- --------------------
  Number of embryos                       1.754262
  Embryo Grade 1                          0.315717
  Number of fertilized eggs               0.287708
  Sperm morphology                        0.183991
  Male Age                                0.144590
  Sperm motility                          0.134987
  Female Age                              0.124247
  AMH                                     0.096107
  Oocyte Maturity MII                     0.086044
  Oocyte Maturity MI                      0.075821

SHAP describes how the fitted model uses features. It does not establish
causation, biological mechanism, or clinical importance. The prominence
of embryo-related features must be interpreted alongside the unresolved
prediction-time and leakage concerns.

## Backend API

The backend uses FastAPI.

### Endpoints

  --------------------------------------------------------------------------------------
  Method                  Endpoint                               Purpose
  ----------------------- -------------------------------------- -----------------------
  GET                     `/`                                    API root and
                                                                 application status

  GET                     `/api/health`                          Health check, if
                                                                 registered by the
                                                                 health router

  POST                    `/api/v1/predict`                      Prediction from the 15
                                                                 numeric model features

  GET                     `/api/v1/analytics/model-comparison`   Validation metrics

  GET                     `/api/v1/analytics/shap`               SHAP feature importance
  --------------------------------------------------------------------------------------

Inspect the live OpenAPI schema for the exact request body and response
fields.

### Start backend

From the project root:

``` powershell
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8001 --reload
```

Open: - API root: `http://127.0.0.1:8001/` - Swagger:
`http://127.0.0.1:8001/docs` - OpenAPI schema:
`http://127.0.0.1:8001/openapi.json`

Port 8001 is used because another local application was found occupying
port 8000. If you use another port, update the frontend API base URL.

The analytics routes read these generated files:

``` text
reports/training/validation_metrics.json
reports/explainability/shap_feature_importance.csv
```

The inference service loads:

``` text
ml/models/best_model.joblib
```

## Frontend

The frontend uses React, Vite, Recharts, and Lucide React.

From the project root:

``` powershell
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`.

The frontend API base URL is configured in `frontend/src/App.jsx` and
should point to `http://127.0.0.1:8001` for the local setup above.
FastAPI CORS settings must allow the Vite origin, normally
`http://localhost:5173`.

## Installation

### Prerequisites

-   Python 3.12 recommended for the current development environment
-   Node.js and npm
-   Git
-   Visual Studio Code or another editor

### 1. Clone the repository

``` powershell
git clone https://github.com/YOUR_USERNAME/FertilityIQ.git
cd FertilityIQ
```

Replace `YOUR_USERNAME` with the repository owner's GitHub username or
organization.

### 2. Create and activate a virtual environment

``` powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use an appropriate execution policy for
your account or call the environment's Python executable directly.

### 3. Install Python dependencies

``` powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Install frontend dependencies

``` powershell
cd frontend
npm install
cd ..
```

### 5. Obtain the dataset and model artifacts

Obtain an authorized dataset copy from its original source and place it
in the raw-data location expected by the project's scripts. Raw data is
excluded from Git.

Confirm that `ml/models/best_model.joblib` exists. If it is not included
in the repository, run the ML training workflow after obtaining the
dataset and generating the required reports.

## Run locally

Use two terminals.

**Terminal 1 --- Backend**, from the repository root:

``` powershell
.\.venv\Scripts\Activate.ps1
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8001 --reload
```

**Terminal 2 --- Frontend**, from the repository root:

``` powershell
cd frontend
npm run dev
```

Open `http://localhost:5173`. Keep both terminals running.

## Reproducing the ML workflow

The intended workflow is:

1.  Inspect and validate the source dataset.
2.  Run exploratory data analysis.
3.  Prepare the target and selected features.
4.  Generate stratified train, validation, and test splits.
5.  Train Logistic Regression, XGBoost, and LightGBM.
6.  Compare validation metrics and select a model.
7.  Save the selected model pipeline.
8.  Evaluate it on the held-out test set.
9.  Generate SHAP feature importance.
10. Start the backend and frontend.

The exact script names and command-line options should be checked in the
current repository before running them. Do not tune models or thresholds
using the held-out test set.

## Troubleshooting

**Port 8000 is occupied:** Run FertilityIQ on port 8001 and ensure
`API_BASE` in `App.jsx` uses the same port.

**Frontend cannot reach backend:** Confirm the backend is running, open
`/docs`, check the API base URL, and verify that CORS allows the
frontend origin.

**API returns 404:** Inspect `/openapi.json`. If the expected route is
absent, check imports and `app.include_router(...)` calls in
`backend/app/main.py`. Confirm that the frontend is calling the correct
host and port.

**Model file not found:** Confirm `ml/models/best_model.joblib` exists
and that the service's resolved path points to it.

**Analytics report not found:** Confirm
`reports/training/validation_metrics.json` and
`reports/explainability/shap_feature_importance.csv` exist. Generate the
corresponding reports if needed.

## Limitations and responsible use

1.  The semantic mapping of target classes 0 and 1 is unverified.
2.  Feature units, definitions, and coding conventions need
    confirmation.
3.  The intended prediction timepoint is not defined.
4.  Embryo and fertilization variables may create prediction-time
    leakage.
5.  Patient-level independence has not been established.
6.  External validation has not been performed.
7.  Calibration, threshold selection, and subgroup reliability analysis
    require further work.
8.  The dataset may not represent other clinics, populations, protocols,
    or time periods.
9.  Clinical utility has not been assessed.
10. The local prototype is not designed or approved for identifiable
    patient information.

The recorded metrics are experimental results from one row-level split.
They are not clinical performance guarantees. Do not use the model for
diagnosis, treatment selection, embryo selection, or patient-care
decisions.

## Future work

-   Verify target encoding and the source dataset codebook.
-   Document feature definitions, units, and availability at the
    prediction timepoint.
-   Establish patient-level identifiers and use group-aware evaluation.
-   Conduct external validation with appropriately governed independent
    data.
-   Add calibration curves, threshold analysis, and uncertainty
    estimates.
-   Evaluate subgroup performance where sample sizes permit.
-   Add automated tests for data validation, API routes, and inference.
-   Add experiment tracking and model versioning.
-   Improve deployment security, access control, audit logging, and
    privacy safeguards before any real-world data handling.

## Licence and attribution

A software licence for this repository has not yet been specified. Add a
`LICENSE` file only after selecting a licence and confirming rights to
distribute all included code and model artifacts.

The dataset is separately governed by the licence attached to the exact
Mendeley Data version used. A software licence does not override dataset
terms. Cite the original dataset record and version in academic or
public presentations. Do not redistribute the dataset unless its licence
permits it.

------------------------------------------------------------------------

## Acknowledgement

FertilityIQ was developed as an educational and portfolio project to
demonstrate an end-to-end machine-learning workflow, including data
exploration, model comparison, held-out evaluation, explainability, API
development, and frontend integration.

**FertilityIQ is a research prototype, not a clinical decision system.**
