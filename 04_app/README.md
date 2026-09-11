# FleetFit — TFM FINAL 2

This independent application package belongs to TFM_FINAL_2. See OPEN_FLEETFIT_APP.txt for the deployment status. The original FleetFit deployment belongs to the separate TFM_FINAL release.

FleetFit forecasts December 2025 from the November 2025 forecast origin. It compares aircraft candidates for a directional route using recent operating evidence and predicted monthly load factor. It does not execute notebooks or train a model during use.

## Local installation

Use Python 3.12. If reproducing the notebooks as well, extract the independent application ZIP into a separate directory so notebook outputs cannot replace its frozen artifacts. From the package root containing both `04_app/` and `03_outputs/`, run:

```bash
python3.12 -m venv .venv-fleetfit
source .venv-fleetfit/bin/activate
python -m pip install -r 04_app/requirements.lock
python -m pip check
python 04_app/validate_install.py
python -m unittest discover -s 04_app/tests -v
python -m streamlit run 04_app/app.py
```

On Windows, create the environment with `py -3.12 -m venv .venv-fleetfit` and activate it in PowerShell with `.venv-fleetfit\Scripts\Activate.ps1`. The remaining commands are the same. The recorded local installation check used macOS; Windows has not been independently tested.

Open the local URL printed by Streamlit. Keep the terminal running while using the app.

`requirements.txt` pins the direct application dependencies; `requirements.lock` records the complete tested local environment. XGBoost 3.4.1 and scikit-learn 1.6.1 match the distributed fitted pipeline. These are the requirements for this application package. See [INSTALLATION_CHECK.md](INSTALLATION_CHECK.md) for the scope of verification.

## Inference artifacts and evaluation

The app reads `../03_outputs/model/` relative to this directory. It requires only `FINAL_LOAD_FACTOR_PIPELINE.joblib`, `FINAL_INFERENCE_CONTEXT.duckdb`, `FINAL_FEATURE_SCHEMA.csv` and `FINAL_PREDICTION_CONTRACT.json`. Additional filenames recorded in the notebook export contract describe optional export outputs and are not loaded by this app. The installation verifier checks pipeline and context SHA-256 values against `FINAL_PREDICTION_CONTRACT.json`, confirms the 40-predictor schema and fitted model version, and evaluates three reference forecasts. `TFM_MODEL_ARTIFACT_DIRECTORY` optionally selects another artifact directory for isolated testing; the application, verifier and tests honour it.

The model is `XGB_N300_LR0P05_D6`. N02 Section 19 fits and evaluates this model; Sections 22–23 reuse and export the same fitted preprocessor and estimator without another fit. Reloading the exported pipeline reproduced all 32,067 holdout predictions with zero difference in the recorded run. The research improvement of 16.3% over the three-month mean is measured on 17,862 common-history observations; it is not a guaranteed improvement for each application forecast.

The application retains its calendar normalisation: month, quarter and pandemic indicator are represented as integer strings, matching the fitted encoder categories. The notebook-exported runtime can stringify floating calendar values as `12.0`; the app uses `12`. Consequently, identity of the fitted model does not imply identical candidate forecasts from those two input-preparation implementations. Application reference forecasts are checked separately by `validate_install.py`.

## Rankings and operational support

The aircraft-range check determines candidate eligibility. The operated ranking sorts by recent departures, then observed recent months and predicted LF. The projected ranking sorts by observed recent months, support tier, predicted LF and recent departures, in that order. Both display the first five candidates. Exploratory forecasts are visible and labelled accordingly.

Support labels are calculated by `_reliability_from_recent_support()` in `load_factor_recommender.py`, using the twelve months ending at the forecast origin. The first matching rule applies:

| Runtime label | Visible label | Observed months | Departures |
|---|---|---:|---:|
| Standard | Strong support | 12 | At least 12 |
| Moderate | Established support | At least 6 | At least 12 |
| Low | Emerging support | At least 3 | At least 3 |
| Exploratory | Exploratory | Otherwise | Otherwise |

The notebook-generated `FINAL_RELIABILITY_RULES.csv` (omitted from this minimal runtime package) documents error evidence by **total observed history** of each route–aircraft combination. It is not the configuration executed by the app to assign support labels. These labels are operational support categories, not calibrated accuracy probabilities or prediction intervals. The final internal tier is `Exploratory — abstain from standard recommendation`.

The app also provides route context, historical operators and aircraft profiles. Local aircraft photographs and airline logos are included; carriers without a bundled logo use a readable code badge. Logo source URLs are recorded in `assets/airlines/sources.json`. Absence from the observed-direct-flights panel means no recorded direct service in the displayed historical window, not that connecting travel is unavailable.

## Scope

The range screen is not a certification of operational feasibility. The analyst remains responsible for runway performance, regulatory constraints, aircraft and crew availability, costs and commercial suitability. Forecasts describe route–aircraft combinations aggregated across carriers, rather than a carrier-specific fleet assignment.
