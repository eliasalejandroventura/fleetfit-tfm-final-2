# FleetFit TFM FINAL 2 — execution check

This record applies only to the independent TFM_FINAL_2 package.

- The application was tested with the previously verified Python 3.12 environment and all eight pinned direct dependencies.
- The installation verifier checks the new pipeline and context hashes against the N02 export contract, the 40-predictor schema, XGBoost version and three reference forecasts.
- All 15 existing acceptance tests passed with the new artifacts.
- Ten additional routes were sampled with seed 42 from observed routes whose endpoints are available in the application. Candidate scoring, LF bounds, projected support ordering and positive route distance passed for all ten.
- All 83 aircraft profiles match the notebook-exported profile implementation.
- A new clean environment installation was not repeated during this review. Windows and the new cloud deployment have not yet been independently verified.

The pipeline SHA-256 is `4a4d50a7a2180617964ca7bc52224f60994523cb3b9ab9bf26ed29280a3068da`. The context hash is recorded in `FINAL_PREDICTION_CONTRACT.json`; the verifier checks both files.

Calendar inputs retain the application's integer-string normalisation to match the trained categories. The input-preparation distinction from the notebook-exported runtime is explained in README.md. No model fitting occurs in the app.
