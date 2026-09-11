# FleetFit TFM FINAL 2 — execution check

This record applies only to the independent TFM_FINAL_2 package.

- The application was tested with the previously verified Python 3.12 environment and all eight pinned direct dependencies.
- The installation verifier checks the new pipeline and context hashes against the N02 export contract, the 40-predictor schema, XGBoost version and three reference forecasts.
- All 15 existing acceptance tests passed with the new artifacts.
- Ten additional routes were sampled with seed 42 from observed routes whose endpoints are available in the application. Candidate scoring, LF bounds, projected support ordering and positive route distance passed for all ten.
- All 83 aircraft profiles match the notebook-exported profile implementation.
- The local checks reused the previously verified environment. The independent Streamlit deployment successfully performed a fresh installation from `04_app/requirements.txt` on Python 3.12.14 on 11 September 2026. All eight direct dependencies match their pinned versions.
- Streamlit automatically replaced PyArrow 25.0.1 with 24.0.0 during installation. The hosted environment therefore differs from the complete local lock; the model dependencies remain unchanged.
- The public application was checked through its browser interface: Orlando–Miami recommendations, the linked aircraft profile and image, and About. The A321neoLR forecast displays 80.6%, consistent with the local reference. Screenshots were reviewed at desktop width, 390-pixel mobile width and 768-pixel tablet width.
- An independent download of the public GitHub repository matched all 113 application-package files byte for byte before these deployment notes were added.
- Windows has not been independently tested. The 15 acceptance tests and ten-route sample described above ran locally; they were not executed inside the managed cloud container.

The pipeline SHA-256 is `4a4d50a7a2180617964ca7bc52224f60994523cb3b9ab9bf26ed29280a3068da`. The context hash is recorded in `FINAL_PREDICTION_CONTRACT.json`; the verifier checks both files.

Calendar inputs retain the application's integer-string normalisation to match the trained categories. The input-preparation distinction from the notebook-exported runtime is explained in README.md. No model fitting occurs in the app.

Public application: https://fleetfit-tfm-final-2.streamlit.app/
Source repository: https://github.com/eliasalejandroventura/fleetfit-tfm-final-2
