"""Repeatable acceptance checks for the standalone FleetFit application."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from streamlit.testing.v1 import AppTest


APP_DIRECTORY = Path(__file__).resolve().parents[1]
ROOT = APP_DIRECTORY.parent
ARTIFACT_DIRECTORY = Path(os.environ.get("TFM_MODEL_ARTIFACT_DIRECTORY", ROOT / "03_outputs" / "model")).resolve()
sys.path[:0] = [str(APP_DIRECTORY), str(ARTIFACT_DIRECTORY)]

from load_factor_recommender import LoadFactorRecommender  # noqa: E402
from ui_logic import (  # noqa: E402
    aircraft_family,
    average_daily_departures,
    filter_airports,
    route_rankings,
)


ROUTE_MATRIX = [
    ("JFK", "MIA", "dense domestic"),
    ("MCO", "LAX", "long domestic"),
    ("LAX", "HNL", "island market"),
    ("ATL", "CHA", "short regional"),
    ("DFW", "ABI", "non-hub regional"),
    ("SEA", "ANC", "Alaska market"),
    ("MIA", "SJU", "territorial market"),
    ("JFK", "LHR", "dense long haul"),
    ("BOS", "DUB", "thin long haul"),
    ("MIA", "GRU", "South America"),
    ("LAX", "NRT", "transpacific"),
    ("JFK", "CDG", "transatlantic"),
]


class FleetFitRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = LoadFactorRecommender(ARTIFACT_DIRECTORY)
        active = cls.model.airport_options(active_only=True)
        catalog = pd.read_csv(
            APP_DIRECTORY / "assets" / "airport_catalog.csv",
            dtype={"AIRPORT_CODE": "string"},
        )
        cls.airports = active.merge(catalog, on="AIRPORT_CODE", how="left")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.model.close()

    def test_active_airport_catalog_is_complete_and_unique(self) -> None:
        self.assertGreaterEqual(len(self.airports), 900)
        self.assertFalse(self.airports["AIRPORT_CODE"].duplicated().any())
        required = [
            "AIRPORT_CODE", "AIRPORT_ID", "AIRPORT_NAME", "CITY_NAME",
            "COUNTRY_NAME",
        ]
        self.assertFalse(self.airports[required].isna().any().any())

    def test_search_requires_three_characters_and_ranks_relevant_result(self) -> None:
        self.assertTrue(filter_airports(self.airports, "ta").empty)
        tampa = filter_airports(self.airports, "tampa", limit=10)
        self.assertFalse(tampa.empty)
        self.assertEqual(tampa.iloc[0]["AIRPORT_CODE"], "TPA")
        self.assertNotIn("TAK", tampa["AIRPORT_CODE"].tolist())
        exact = filter_airports(self.airports, "MIA", limit=10)
        self.assertEqual(exact.iloc[0]["AIRPORT_CODE"], "MIA")

    def test_persisted_airport_labels_recover_the_internal_selection(self) -> None:
        from ui_logic import recover_airport_code

        self.assertEqual(
            recover_airport_code(
                self.airports,
                "Miami International Airport (MIA) · Miami, FL",
            ),
            "MIA",
        )
        self.assertEqual(
            recover_airport_code(
                self.airports,
                "Orlando International Airport (MCO) · Orlando, FL",
            ),
            "MCO",
        )
        self.assertIsNone(recover_airport_code(self.airports, "Not an airport"))

    def test_every_aircraft_candidate_has_a_profile_and_family(self) -> None:
        candidates = self.model.aircraft_options()
        self.assertEqual(len(candidates), 83)
        image_manifest = pd.read_csv(
            APP_DIRECTORY / "assets" / "aircraft_image_manifest.csv"
        )
        self.assertEqual(len(image_manifest), 83)
        self.assertFalse(image_manifest["AIRCRAFT_TYPE"].duplicated().any())
        self.assertEqual(
            set(image_manifest["AIRCRAFT_TYPE"].astype(int)),
            set(candidates["AIRCRAFT_TYPE"].astype(int)),
        )
        image_audit = candidates[["AIRCRAFT_TYPE", "AIRCRAFT_DESCRIPTION"]].merge(
            image_manifest,
            on=["AIRCRAFT_TYPE", "AIRCRAFT_DESCRIPTION"],
            how="left",
            validate="one_to_one",
        )
        image_audit["IMAGE_EXISTS"] = image_audit["ASSET_PATH"].map(
            lambda value: (APP_DIRECTORY / str(value)).is_file()
        )
        self.assertGreaterEqual(float(image_audit["IMAGE_EXISTS"].mean()), 0.90)
        self.assertTrue(image_audit["IMAGE_EXISTS"].all())
        for relative_path in image_audit["ASSET_PATH"].unique():
            with Image.open(APP_DIRECTORY / relative_path) as image:
                self.assertGreaterEqual(image.width, 640)
                self.assertGreaterEqual(image.height, 400)
        profile_audit = self.model.connection.execute(
            """
            SELECT
                AIRCRAFT_TYPE,
                QUANTILE_CONT(SEATS / NULLIF(DEPARTURES_PERFORMED, 0), 0.05)
                    AS SEATS_P05,
                QUANTILE_CONT(SEATS / NULLIF(DEPARTURES_PERFORMED, 0), 0.50)
                    AS SEATS_MEDIAN,
                QUANTILE_CONT(SEATS / NULLIF(DEPARTURES_PERFORMED, 0), 0.95)
                    AS SEATS_P95
            FROM carrier_configuration
            WHERE DEPARTURES_PERFORMED > 0
            GROUP BY AIRCRAFT_TYPE
            """
        ).fetchdf()
        audited = candidates.merge(profile_audit, on="AIRCRAFT_TYPE", how="left")
        self.assertFalse(audited[["SEATS_P05", "SEATS_MEDIAN", "SEATS_P95"]].isna().any().any())
        self.assertTrue((audited["SEATS_MEDIAN"] > 0).all())
        self.assertTrue((audited["SEATS_P95"] >= audited["SEATS_P05"]).all())
        for row in candidates.itertuples(index=False):
            family = aircraft_family(row.AIRCRAFT_DESCRIPTION)
            self.assertTrue(family.strip())
            self.assertGreater(float(row.RANGE_NM), 0)

        for aircraft_type in [416, 614, 699, 837, 882]:
            profile = self.model.aircraft_profile(aircraft_type)
            self.assertEqual(int(profile["AIRCRAFT_TYPE"]), aircraft_type)
            self.assertGreater(float(profile["SEATS_MEDIAN"]), 0)

    def test_family_mapping_covers_principal_manufacturers(self) -> None:
        expected = {
            "Airbus Industrie A321Neolr": "Airbus A321",
            "Airbus 350-1000": "Airbus A350",
            "Boeing 777-300/300ER/333ER": "Boeing 777",
            "B787-900 Dreamliner": "Boeing 787",
            "Canadair RJ-700": "Bombardier CRJ",
        }
        for description, family in expected.items():
            self.assertEqual(aircraft_family(description), family)

    def test_route_matrix_scores_and_operational_totals(self) -> None:
        for origin, destination, segment in ROUTE_MATRIX:
            with self.subTest(route=f"{origin}-{destination}", segment=segment):
                score = self.model.score(origin, destination)
                overview = self.model.route_overview(origin, destination)
                history = self.model.route_lf_history(origin, destination)
                operators = self.model.route_carrier_insights(origin, destination)
                operated, projected = route_rankings(score)

                self.assertFalse(score.empty)
                self.assertTrue(score["PREDICTED_LF"].between(0, 1).all())
                self.assertGreater(overview["distance_miles"], 0)
                self.assertTrue(np.isfinite(overview["average_flight_minutes"]))
                self.assertGreater(overview["departures_trailing_12"], 0)
                self.assertGreater(overview["observed_months"], 0)
                self.assertLessEqual(overview["observed_months"], 12)
                self.assertLessEqual(len(history), 12)
                self.assertTrue(history["LOAD_FACTOR"].between(0, 1).all())
                self.assertTrue((operators["CAPACITY_SHARE"] >= 0.01).all())
                self.assertLessEqual(float(operators["CAPACITY_SHARE"].sum()), 1.000001)
                self.assertTrue(operated["RECENT_ROUTE_DEPARTURES"].is_monotonic_decreasing)
                months = projected["RECENT_ROUTE_MONTHS"].to_numpy()
                self.assertTrue((months[:-1] >= months[1:]).all())
                daily = average_daily_departures(
                    overview["departures_trailing_12"], overview["observed_months"]
                )
                self.assertGreater(daily, 0)
                self.assertLess(daily, 100)

    def test_jfk_mia_regression_case_is_plausible(self) -> None:
        score = self.model.score("JFK", "MIA")
        overview = self.model.route_overview("JFK", "MIA")
        operated, projected = route_rankings(score)
        self.assertEqual(operated.iloc[0]["AIRCRAFT_DESCRIPTION"], "Boeing 737-800")
        self.assertEqual(projected.iloc[0]["AIRCRAFT_DESCRIPTION"], "Boeing 737-900ER")
        self.assertGreater(float(projected.iloc[0]["PREDICTED_LF"]), 0.80)
        self.assertEqual(int(projected.iloc[0]["RECENT_ROUTE_MONTHS"]), 12)
        daily = average_daily_departures(
            overview["departures_trailing_12"], overview["observed_months"]
        )
        self.assertGreater(daily, 10)
        self.assertLess(daily, 20)

    def test_mia_lima_flight_time_excludes_unreported_zero_durations(self) -> None:
        overview = self.model.route_overview("MIA", "LIM")
        self.assertEqual(
            overview["flight_time_method"], "reported ramp-to-ramp time"
        )
        self.assertGreater(overview["average_flight_minutes"], 300)
        self.assertLess(overview["average_flight_minutes"], 390)
        implied_speed_mph = overview["distance_miles"] / (
            overview["average_flight_minutes"] / 60
        )
        self.assertGreater(implied_speed_mph, 350)
        self.assertLess(implied_speed_mph, 600)

    def test_invalid_route_inputs_fail_explicitly(self) -> None:
        with self.assertRaisesRegex(ValueError, "different airports"):
            self.model.score("JFK", "JFK")
        with self.assertRaisesRegex(ValueError, "exactly one catalogue record"):
            self.model.score("ZZZ", "MIA")


class FleetFitStreamlitTests(unittest.TestCase):
    def test_home_and_fleet_initial_state_render_without_exception(self) -> None:
        app = AppTest.from_file(APP_DIRECTORY / "app.py", default_timeout=120)
        app.run(timeout=120)
        self.assertEqual(len(app.exception), 0)
        app.query_params["view"] = "fleet"
        app.run(timeout=120)
        self.assertEqual(len(app.exception), 0)
        self.assertEqual([item.label for item in app.selectbox], [
            "Origin airport", "Destination airport"
        ])

    def test_tampa_miami_happy_path_renders_without_exception(self) -> None:
        app = AppTest.from_file(APP_DIRECTORY / "app.py", default_timeout=120)
        app.query_params["view"] = "fleet"
        app.run(timeout=120)
        app.selectbox(key="origin_code").select("TPA").run(timeout=120)
        app.selectbox(key="destination_code").select("MIA").run(timeout=120)
        app.button[-1].click().run(timeout=120)
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(
            app.session_state["recent_routes"],
            [{"origin": "TPA", "destination": "MIA"}],
        )

    def test_recent_route_can_be_restored(self) -> None:
        app = AppTest.from_file(APP_DIRECTORY / "app.py", default_timeout=120)
        app.query_params["view"] = "fleet"
        app.session_state["recent_routes"] = [
            {"origin": "MIA", "destination": "MCO"}
        ]
        app.run(timeout=120)

        self.assertEqual(len(app.exception), 0)
        app.button(key="recent_route_0").click().run(timeout=120)
        self.assertEqual(app.session_state["origin_code"], "MIA")
        self.assertEqual(app.session_state["destination_code"], "MCO")
        self.assertEqual(
            app.session_state["submitted_route"],
            {"origin": "MIA", "destination": "MCO"},
        )
        self.assertEqual(len(app.error), 0)
        self.assertEqual(len(app.warning), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
