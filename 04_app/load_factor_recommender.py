"""Standalone inference runtime for the load-factor aircraft recommender."""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd


class LoadFactorRecommender:
    """Load final artifacts and score routes without executing Notebook 02."""

    def __init__(self, artifact_directory: str | Path):
        self.artifact_directory = Path(artifact_directory).expanduser().resolve()
        contract_path = self.artifact_directory / "FINAL_PREDICTION_CONTRACT.json"
        self.contract = json.loads(contract_path.read_text(encoding="utf-8"))
        self.pipeline = joblib.load(
            self.artifact_directory / self.contract["pipeline_file"]
        )
        self.schema = pd.read_csv(
            self.artifact_directory / self.contract["feature_schema_file"]
        )
        self.features = self.schema["Feature"].tolist()
        self.categorical_features = self.schema.loc[
            self.schema["Statistical type"].eq("Categorical"), "Feature"
        ].tolist()
        self.numerical_features = [
            feature for feature in self.features
            if feature not in self.categorical_features
        ]
        self.connection = duckdb.connect(
            str(self.artifact_directory / self.contract["context_file"]),
            read_only=True,
        )
        metadata = self.connection.execute(
            "SELECT key, value FROM metadata"
        ).fetchall()
        self.metadata = dict(metadata)

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "LoadFactorRecommender":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def airport_options(self, active_only: bool = True) -> pd.DataFrame:
        """Return airports with recent commercial activity by default."""
        activity_filter = ""
        if active_only:
            activity_filter = """
                WHERE a.AIRPORT_ID IN (
                    SELECT ORIGIN_AIRPORT_ID
                    FROM carrier_configuration
                    WHERE SERVICE_DATE BETWEEN CAST(? AS DATE) - INTERVAL '11 months'
                                           AND CAST(? AS DATE)
                    GROUP BY ORIGIN_AIRPORT_ID
                    HAVING SUM(DEPARTURES_PERFORMED) >= 12
                       AND COUNT(DISTINCT SERVICE_DATE) >= 3
                )
            """
        parameters = (
            [self.metadata["forecast_origin"], self.metadata["forecast_origin"]]
            if active_only else []
        )
        return self.connection.execute(
            f"""
            SELECT a.AIRPORT_CODE, a.AIRPORT_ID, a.CITY_MARKET_ID,
                   a.STATE_ABR, a.COUNTRY
            FROM airports a
            {activity_filter}
            ORDER BY AIRPORT_CODE
            """,
            parameters,
        ).fetchdf()

    def destination_options(
        self, origin_code: str, operated_only: bool = False
    ) -> pd.DataFrame:
        """Return active destinations with recent-route support information."""
        origin = self._airport_record(origin_code)
        query = """
            SELECT
                a.AIRPORT_CODE,
                a.AIRPORT_ID,
                COALESCE(s.ROUTE_OBSERVED, FALSE) AS ROUTE_OBSERVED,
                COALESCE(s.MAXIMUM_HISTORY_MONTHS, 0) AS MAXIMUM_HISTORY_MONTHS
            FROM airports a
            LEFT JOIN route_support s
              ON s.ORIGIN_AIRPORT_ID = ?
             AND s.DEST_AIRPORT_ID = a.AIRPORT_ID
            WHERE a.AIRPORT_ID <> ?
              AND a.AIRPORT_ID IN (
                  SELECT ORIGIN_AIRPORT_ID
                  FROM carrier_configuration
                  WHERE SERVICE_DATE BETWEEN CAST(? AS DATE) - INTERVAL '11 months'
                                         AND CAST(? AS DATE)
                  GROUP BY ORIGIN_AIRPORT_ID
                  HAVING SUM(DEPARTURES_PERFORMED) >= 12
                     AND COUNT(DISTINCT SERVICE_DATE) >= 3
              )
        """
        parameters = [
            int(origin["AIRPORT_ID"]), int(origin["AIRPORT_ID"]),
            self.metadata["forecast_origin"], self.metadata["forecast_origin"],
        ]
        if operated_only:
            query += """
                AND a.AIRPORT_ID IN (
                    SELECT DEST_AIRPORT_ID
                    FROM carrier_configuration
                    WHERE ORIGIN_AIRPORT_ID = ?
                      AND SERVICE_DATE BETWEEN CAST(? AS DATE) - INTERVAL '11 months'
                                             AND CAST(? AS DATE)
                    GROUP BY DEST_AIRPORT_ID
                    HAVING SUM(DEPARTURES_PERFORMED) >= 12
                       AND COUNT(DISTINCT SERVICE_DATE) >= 3
                )
            """
            parameters.extend([
                int(origin["AIRPORT_ID"]),
                self.metadata["forecast_origin"],
                self.metadata["forecast_origin"],
            ])
        query += " ORDER BY a.AIRPORT_CODE"
        return self.connection.execute(query, parameters).fetchdf()

    def route_carrier_insights(
        self, origin_code: str, destination_code: str, limit: int = 5
    ) -> pd.DataFrame:
        """Summarise operators observed during the trailing twelve months."""
        origin = self._airport_record(origin_code)
        destination = self._airport_record(destination_code)
        forecast_origin = self.metadata["forecast_origin"]
        return self.connection.execute(
            """
            WITH carrier_totals AS (
                SELECT
                    COALESCE(NULLIF(TRIM(CARRIER), ''),
                             NULLIF(TRIM(UNIQUE_CARRIER), ''), 'NA')
                        AS CARRIER_CODE,
                    COALESCE(NULLIF(TRIM(CARRIER_NAME), ''),
                             NULLIF(TRIM(UNIQUE_CARRIER_NAME), ''),
                             CARRIER,
                             UNIQUE_CARRIER) AS CARRIER_NAME,
                    COALESCE(NULLIF(TRIM(BUSINESS_MODEL), ''), 'Unclassified')
                        AS BUSINESS_MODEL,
                    SUM(SEATS) AS SEATS,
                    SUM(DEPARTURES_PERFORMED) AS DEPARTURES
                FROM carrier_configuration
                WHERE ORIGIN_AIRPORT_ID = ?
                  AND DEST_AIRPORT_ID = ?
                  AND SERVICE_DATE BETWEEN CAST(? AS DATE) - INTERVAL '11 months'
                                       AND CAST(? AS DATE)
                GROUP BY 1, 2, 3
            )
            SELECT * FROM (
            SELECT
                CARRIER_CODE,
                CARRIER_NAME,
                BUSINESS_MODEL,
                SEATS,
                DEPARTURES,
                SEATS / NULLIF(SUM(SEATS) OVER (), 0) AS CAPACITY_SHARE
            FROM carrier_totals
            WHERE CARRIER_NAME IS NOT NULL
            ) material_operators
            WHERE CAPACITY_SHARE >= 0.01
            ORDER BY SEATS DESC, CARRIER_NAME
            LIMIT ?
            """,
            [
                int(origin["AIRPORT_ID"]),
                int(destination["AIRPORT_ID"]),
                forecast_origin,
                forecast_origin,
                int(limit),
            ],
        ).fetchdf()

    def aircraft_options(self) -> pd.DataFrame:
        """Return the active commercial aircraft catalogue used for scoring."""
        return self.connection.execute(
            """
            SELECT AIRCRAFT_TYPE, AIRCRAFT_DESCRIPTION, AIRCRAFT_GROUP,
                   RANGE_NM, RANGE_STATUTE_MILES
            FROM aircraft_candidates
            ORDER BY AIRCRAFT_DESCRIPTION, AIRCRAFT_TYPE
            """
        ).fetchdf()

    def route_overview(
        self, origin_code: str, destination_code: str
    ) -> dict:
        """Return map coordinates and recent operational facts for one route."""
        origin = self._airport_record(origin_code)
        destination = self._airport_record(destination_code)
        forecast_origin = self.metadata["forecast_origin"]
        distance_miles, distance_method = self._route_distance(
            int(origin["AIRPORT_ID"]),
            int(destination["AIRPORT_ID"]),
            origin,
            destination,
        )
        recent = self.connection.execute(
            """
            WITH recent AS (
                SELECT *
                FROM carrier_configuration
                WHERE ORIGIN_AIRPORT_ID = ?
                  AND DEST_AIRPORT_ID = ?
                  AND SERVICE_DATE BETWEEN CAST(? AS DATE) - INTERVAL '11 months'
                                       AND CAST(? AS DATE)
            ), carrier_totals AS (
                SELECT
                    COALESCE(NULLIF(TRIM(CARRIER), ''),
                             NULLIF(TRIM(UNIQUE_CARRIER), '')) AS CARRIER_CODE,
                    SUM(SEATS) AS CARRIER_SEATS
                FROM recent
                GROUP BY 1
            )
            SELECT
                (SELECT SUM(DEPARTURES_PERFORMED) FROM recent) AS DEPARTURES,
                (SELECT SUM(RAMP_TO_RAMP) FROM recent
                 WHERE RAMP_TO_RAMP > 0) AS BLOCK_TIME_MINUTES,
                (SELECT SUM(DEPARTURES_PERFORMED) FROM recent
                 WHERE RAMP_TO_RAMP > 0) AS BLOCK_TIME_DEPARTURES,
                (SELECT SUM(AIR_TIME) FROM recent
                 WHERE AIR_TIME > 0) AS AIR_TIME_MINUTES,
                (SELECT SUM(DEPARTURES_PERFORMED) FROM recent
                 WHERE AIR_TIME > 0) AS AIR_TIME_DEPARTURES,
                (SELECT COUNT(*) FROM carrier_totals
                 WHERE CARRIER_CODE IS NOT NULL
                   AND CARRIER_SEATS / NULLIF(
                       (SELECT SUM(CARRIER_SEATS) FROM carrier_totals), 0
                   ) >= 0.01) AS CARRIER_COUNT,
                (SELECT COUNT(DISTINCT AIRCRAFT_TYPE) FROM recent)
                    AS AIRCRAFT_COUNT,
                (SELECT COUNT(DISTINCT SERVICE_DATE) FROM recent)
                    AS OBSERVED_MONTHS
            """,
            [
                int(origin["AIRPORT_ID"]),
                int(destination["AIRPORT_ID"]),
                forecast_origin,
                forecast_origin,
            ],
        ).fetchdf().iloc[0]
        departures = (
            float(recent["DEPARTURES"])
            if pd.notna(recent["DEPARTURES"]) else 0.0
        )
        block_time = (
            float(recent["BLOCK_TIME_MINUTES"])
            if pd.notna(recent["BLOCK_TIME_MINUTES"]) else 0.0
        )
        block_departures = (
            float(recent["BLOCK_TIME_DEPARTURES"])
            if pd.notna(recent["BLOCK_TIME_DEPARTURES"]) else 0.0
        )
        air_time = (
            float(recent["AIR_TIME_MINUTES"])
            if pd.notna(recent["AIR_TIME_MINUTES"]) else 0.0
        )
        air_time_departures = (
            float(recent["AIR_TIME_DEPARTURES"])
            if pd.notna(recent["AIR_TIME_DEPARTURES"]) else 0.0
        )
        if block_departures:
            average_flight_minutes = block_time / block_departures
            flight_time_method = "reported ramp-to-ramp time"
        elif air_time_departures:
            average_flight_minutes = air_time / air_time_departures
            flight_time_method = "reported airborne time"
        else:
            average_flight_minutes = np.nan
            flight_time_method = "not available"
        return {
            "origin_latitude": float(origin["LATITUDE"]),
            "origin_longitude": float(origin["LONGITUDE"]),
            "destination_latitude": float(destination["LATITUDE"]),
            "destination_longitude": float(destination["LONGITUDE"]),
            "distance_miles": distance_miles,
            "distance_method": distance_method,
            "average_flight_minutes": average_flight_minutes,
            "flight_time_method": flight_time_method,
            "departures_trailing_12": int(round(departures)),
            "flights_per_week": departures / 52.1775 if departures else 0.0,
            "carrier_count": int(recent["CARRIER_COUNT"])
                if pd.notna(recent["CARRIER_COUNT"]) else 0,
            "aircraft_count": int(recent["AIRCRAFT_COUNT"])
                if pd.notna(recent["AIRCRAFT_COUNT"]) else 0,
            "observed_months": int(recent["OBSERVED_MONTHS"])
                if pd.notna(recent["OBSERVED_MONTHS"]) else 0,
        }

    def route_aircraft_insights(
        self, origin_code: str, destination_code: str, limit: int = 8
    ) -> pd.DataFrame:
        """List recent carrier-aircraft combinations observed on the route."""
        origin = self._airport_record(origin_code)
        destination = self._airport_record(destination_code)
        forecast_origin = self.metadata["forecast_origin"]
        return self.connection.execute(
            """
            SELECT
                COALESCE(NULLIF(TRIM(c.CARRIER), ''),
                         NULLIF(TRIM(c.UNIQUE_CARRIER), ''), 'NA')
                    AS CARRIER_CODE,
                COALESCE(NULLIF(TRIM(c.CARRIER_NAME), ''),
                         NULLIF(TRIM(c.UNIQUE_CARRIER_NAME), ''),
                         c.CARRIER, c.UNIQUE_CARRIER) AS CARRIER_NAME,
                COALESCE(NULLIF(TRIM(c.BUSINESS_MODEL), ''), 'Unclassified')
                    AS BUSINESS_MODEL,
                COALESCE(a.AIRCRAFT_DESCRIPTION,
                         'BTS aircraft type ' || CAST(c.AIRCRAFT_TYPE AS VARCHAR))
                    AS AIRCRAFT_DESCRIPTION,
                SUM(c.DEPARTURES_PERFORMED) AS DEPARTURES,
                SUM(c.SEATS) / NULLIF(SUM(c.DEPARTURES_PERFORMED), 0)
                    AS AVERAGE_SEATS_PER_DEPARTURE
            FROM carrier_configuration c
            LEFT JOIN aircraft_candidates a
              ON a.AIRCRAFT_TYPE = c.AIRCRAFT_TYPE
            WHERE c.ORIGIN_AIRPORT_ID = ?
              AND c.DEST_AIRPORT_ID = ?
              AND c.SERVICE_DATE BETWEEN CAST(? AS DATE) - INTERVAL '11 months'
                                     AND CAST(? AS DATE)
            GROUP BY 1, 2, 3, 4
            HAVING SUM(c.DEPARTURES_PERFORMED) > 0
            ORDER BY DEPARTURES DESC, CARRIER_NAME, AIRCRAFT_DESCRIPTION
            LIMIT ?
            """,
            [
                int(origin["AIRPORT_ID"]),
                int(destination["AIRPORT_ID"]),
                forecast_origin,
                forecast_origin,
                int(limit),
            ],
        ).fetchdf()

    def route_lf_history(
        self, origin_code: str, destination_code: str, months: int = 12
    ) -> pd.DataFrame:
        """Return the latest observed route-level LF history."""
        origin = self._airport_record(origin_code)
        destination = self._airport_record(destination_code)
        forecast_origin = self.metadata["forecast_origin"]
        return self.connection.execute(
            """
            SELECT
                SERVICE_DATE,
                MARKET_PASSENGERS / NULLIF(MARKET_SEATS, 0) AS LOAD_FACTOR,
                MARKET_PASSENGERS,
                MARKET_SEATS,
                MARKET_DEPARTURES
            FROM route_month
            WHERE ORIGIN_AIRPORT_ID = ?
              AND DEST_AIRPORT_ID = ?
              AND MARKET_SEATS > 0
              AND SERVICE_DATE <= CAST(? AS DATE)
            ORDER BY SERVICE_DATE DESC
            LIMIT ?
            """,
            [
                int(origin["AIRPORT_ID"]),
                int(destination["AIRPORT_ID"]),
                forecast_origin,
                int(months),
            ],
        ).fetchdf().sort_values("SERVICE_DATE").reset_index(drop=True)

    def aircraft_profile(self, aircraft_type: int) -> dict:
        """Return auditable operational facts for one BTS aircraft model."""
        candidate = self.connection.execute(
            "SELECT * FROM aircraft_candidates WHERE AIRCRAFT_TYPE = ?",
            [int(aircraft_type)],
        ).fetchdf()
        if candidate.empty:
            raise ValueError(f"Unknown aircraft type: {aircraft_type}")
        facts = self.connection.execute(
            "SELECT * EXCLUDE (AIRCRAFT_TYPE) FROM aircraft_profile_facts "
            "WHERE AIRCRAFT_TYPE = ?",
            [int(aircraft_type)],
        ).fetchdf().iloc[0]
        operators = self.connection.execute(
            "SELECT CARRIER_NAME, DEPARTURES FROM aircraft_profile_operators "
            "WHERE AIRCRAFT_TYPE = ? ORDER BY OPERATOR_RANK",
            [int(aircraft_type)],
        ).fetchdf()
        return {
            **candidate.iloc[0].to_dict(),
            **facts.to_dict(),
            "OPERATORS": operators["CARRIER_NAME"].dropna().astype(str).tolist(),
        }

    def score(self, origin_code: str, destination_code: str) -> pd.DataFrame:
        """Rank range-compatible aircraft for one directional route."""
        origin = self._airport_record(origin_code)
        destination = self._airport_record(destination_code)
        origin_id = int(origin["AIRPORT_ID"])
        destination_id = int(destination["AIRPORT_ID"])
        if origin_id == destination_id:
            raise ValueError("Origin and destination must be different airports.")

        distance_miles, distance_method = self._route_distance(
            origin_id, destination_id, origin, destination
        )
        candidates = self.connection.execute(
            """
            SELECT * FROM aircraft_candidates
            WHERE RANGE_STATUTE_MILES >= ?
            ORDER BY AIRCRAFT_TYPE
            """,
            [distance_miles],
        ).fetchdf()
        if candidates.empty:
            raise ValueError(
                "No active candidate has sufficient documented reference range."
            )

        global_context = self._single_context(
            "SELECT * EXCLUDE (CONTEXT_ID) FROM global_context WHERE CONTEXT_ID = 1"
        )
        origin_context = self._single_context(
            "SELECT * EXCLUDE (BTS_AIRPORT_ID) FROM airport_context "
            "WHERE BTS_AIRPORT_ID = ?",
            [origin_id],
        )
        destination_context = self._single_context(
            "SELECT * EXCLUDE (BTS_AIRPORT_ID) FROM airport_context "
            "WHERE BTS_AIRPORT_ID = ?",
            [destination_id],
        )
        route_context = self._single_context(
            "SELECT * EXCLUDE (ORIGIN_AIRPORT_ID, DEST_AIRPORT_ID) "
            "FROM route_context WHERE ORIGIN_AIRPORT_ID = ? AND DEST_AIRPORT_ID = ?",
            [origin_id, destination_id],
        )
        aircraft_context = self.connection.execute(
            "SELECT * FROM aircraft_context"
        ).fetchdf().set_index("AIRCRAFT_TYPE").to_dict("index")
        exact_context = self.connection.execute(
            """
            SELECT * EXCLUDE (ORIGIN_AIRPORT_ID, DEST_AIRPORT_ID)
            FROM exact_context
            WHERE ORIGIN_AIRPORT_ID = ? AND DEST_AIRPORT_ID = ?
            """,
            [origin_id, destination_id],
        ).fetchdf()
        exact_by_aircraft = (
            exact_context.set_index("AIRCRAFT_TYPE").to_dict("index")
            if not exact_context.empty else {}
        )
        recent_support = self.connection.execute(
            """
            SELECT
                AIRCRAFT_TYPE,
                COUNT(DISTINCT SERVICE_DATE) AS RECENT_ROUTE_MONTHS,
                SUM(DEPARTURES_PERFORMED) AS RECENT_ROUTE_DEPARTURES,
                SUM(SEATS) AS RECENT_ROUTE_SEATS,
                SUM(PASSENGERS) AS RECENT_ROUTE_PASSENGERS,
                MAX(SERVICE_DATE) AS LAST_ROUTE_OPERATION
            FROM carrier_configuration
            WHERE ORIGIN_AIRPORT_ID = ?
              AND DEST_AIRPORT_ID = ?
              AND SERVICE_DATE BETWEEN CAST(? AS DATE) - INTERVAL '11 months'
                                   AND CAST(? AS DATE)
              AND DEPARTURES_PERFORMED > 0
            GROUP BY AIRCRAFT_TYPE
            """,
            [
                origin_id,
                destination_id,
                self.metadata["forecast_origin"],
                self.metadata["forecast_origin"],
            ],
        ).fetchdf()
        recent_by_aircraft = (
            recent_support.set_index("AIRCRAFT_TYPE").to_dict("index")
            if not recent_support.empty else {}
        )

        rows = []
        for aircraft in candidates.to_dict("records"):
            aircraft_type = int(aircraft["AIRCRAFT_TYPE"])
            values = {feature: np.nan for feature in self.features}
            values.update(global_context)
            values.update(aircraft_context.get(aircraft_type, {}))
            values.update({
                "ORIGIN_OUTBOUND_LF_LAG_1": origin_context.get("OUTBOUND_LF"),
                "ORIGIN_OUTBOUND_SEATS_LAG_1": origin_context.get("OUTBOUND_SEATS"),
                "DEST_INBOUND_LF_LAG_1": destination_context.get("INBOUND_LF"),
                "DEST_INBOUND_SEATS_LAG_1": destination_context.get("INBOUND_SEATS"),
                "DEST_TRAFFIC_HUB_CLASS_LAG_1": destination_context.get(
                    "TRAFFIC_HUB_CLASS"
                ),
            })
            values.update(route_context)
            exact = exact_by_aircraft.get(aircraft_type, {})
            values.update({
                key: value for key, value in exact.items()
                if key in self.features
            })
            values.update({
                "AIRCRAFT_TYPE": aircraft_type,
                "AIRCRAFT_GROUP": aircraft["AIRCRAFT_GROUP"],
                "ORIGIN_AIRPORT_ID": origin_id,
                "DEST_AIRPORT_ID": destination_id,
                "ORIGIN_CITY_MARKET_ID": origin["CITY_MARKET_ID"],
                "DEST_CITY_MARKET_ID": destination["CITY_MARKET_ID"],
                "ORIGIN_STATE_ABR": origin["STATE_ABR"],
                "DEST_STATE_ABR": destination["STATE_ABR"],
                "ORIGIN_COUNTRY": origin["COUNTRY"],
                "DEST_COUNTRY": destination["COUNTRY"],
                "ORIGIN_WAC": origin["WAC"],
                "DEST_WAC": destination["WAC"],
                "DEST_AIRPORT_CLASS": destination["AIRPORT_CLASS"],
                "ORIGIN_FAA_HUB_CLASS_REFERENCE": origin["FAA_HUB_CLASS"],
                "DEST_FAA_HUB_CLASS_REFERENCE": destination["FAA_HUB_CLASS"],
                "ORIGIN_URBAN_SIZE_REFERENCE": origin["URBAN_SIZE"],
                "DEST_URBAN_SIZE_REFERENCE": destination["URBAN_SIZE"],
                "DISTANCE_GROUP": max(1, int(np.ceil(distance_miles / 500.0))),
            })
            history_months = exact.get("HISTORY_MONTHS_TOTAL", 0)
            recent = recent_by_aircraft.get(aircraft_type, {})
            recent_months = recent.get("RECENT_ROUTE_MONTHS", 0)
            recent_departures = recent.get("RECENT_ROUTE_DEPARTURES", 0)
            recent_seats = recent.get("RECENT_ROUTE_SEATS", 0)
            recent_passengers = recent.get("RECENT_ROUTE_PASSENGERS", 0)
            recent_observed_lf = (
                float(recent_passengers) / float(recent_seats)
                if pd.notna(recent_seats) and float(recent_seats) > 0
                else np.nan
            )
            rows.append({
                **values,
                "AIRCRAFT_DESCRIPTION": aircraft["AIRCRAFT_DESCRIPTION"],
                "RANGE_NM": aircraft["RANGE_NM"],
                "ROUTE_DISTANCE_MILES": distance_miles,
                "DISTANCE_METHOD": distance_method,
                "HISTORY_MONTHS": 0 if pd.isna(history_months) else int(history_months),
                "RECENT_ROUTE_MONTHS": (
                    0 if pd.isna(recent_months) else int(recent_months)
                ),
                "RECENT_ROUTE_DEPARTURES": (
                    0 if pd.isna(recent_departures) else int(recent_departures)
                ),
                "RECENT_OBSERVED_LF": recent_observed_lf,
                "LAST_ROUTE_OPERATION": recent.get("LAST_ROUTE_OPERATION", pd.NaT),
            })

        scoring = pd.DataFrame(rows)
        prepared = self._prepare_feature_frame(scoring)
        scoring["PREDICTED_LF"] = np.clip(
            self.pipeline.predict(prepared), 0.0, 1.0
        )
        scoring["RELIABILITY"] = scoring.apply(
            lambda row: self._reliability_from_recent_support(
                row["RECENT_ROUTE_MONTHS"], row["RECENT_ROUTE_DEPARTURES"]
            ),
            axis=1,
        )
        reliability_priority = {
            "Standard": 0,
            "Moderate": 1,
            "Low": 2,
            "Exploratory — abstain from standard recommendation": 3,
        }
        scoring["EVIDENCE_PRIORITY"] = scoring["RELIABILITY"].map(
            reliability_priority
        ).fillna(4)
        scoring = scoring.sort_values(
            [
                "EVIDENCE_PRIORITY",
                "PREDICTED_LF",
                "RECENT_ROUTE_DEPARTURES",
                "HISTORY_MONTHS",
            ],
            ascending=[True, False, False, False],
        ).reset_index(drop=True)
        scoring = scoring.drop(columns="EVIDENCE_PRIORITY")
        scoring.insert(0, "RANK", np.arange(1, len(scoring) + 1))
        return scoring

    def _airport_record(self, code: str) -> pd.Series:
        rows = self.connection.execute(
            "SELECT * FROM airports WHERE UPPER(AIRPORT_CODE) = UPPER(?)",
            [code.strip()],
        ).fetchdf()
        if len(rows) != 1:
            raise ValueError(
                f"Airport code must identify exactly one catalogue record: {code}"
            )
        return rows.iloc[0]

    def _route_distance(
        self,
        origin_id: int,
        destination_id: int,
        origin: pd.Series,
        destination: pd.Series,
    ) -> tuple[float, str]:
        row = self.connection.execute(
            """
            SELECT DISTANCE_MILES FROM route_distances
            WHERE ORIGIN_AIRPORT_ID = ? AND DEST_AIRPORT_ID = ?
            """,
            [origin_id, destination_id],
        ).fetchone()
        if row is not None:
            return float(row[0]), "Historical median"
        coordinates = [
            origin["LATITUDE"], origin["LONGITUDE"],
            destination["LATITUDE"], destination["LONGITUDE"],
        ]
        if any(pd.isna(value) for value in coordinates):
            raise ValueError(
                "Route distance cannot be constructed because coordinates are missing."
            )
        return self._great_circle_miles(*coordinates), "Great-circle calculation"

    def _single_context(self, query: str, parameters=None) -> dict:
        frame = self.connection.execute(query, parameters or []).fetchdf()
        return {} if frame.empty else frame.iloc[0].to_dict()

    def _prepare_feature_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        prepared = frame.reindex(columns=self.features).copy()
        for feature in self.categorical_features:
            if feature in {"MONTH_OF_YEAR", "QUARTER_OF_YEAR", "COVID_PERIOD"}:
                # Calendar categories were fitted as integer strings in N02.
                # The mixed numeric context row can return them as floats.
                prepared[feature] = pd.to_numeric(
                    prepared[feature], errors="raise"
                ).astype("Int64")
            prepared[feature] = prepared[feature].astype("string").fillna("__MISSING__")
        for feature in self.numerical_features:
            prepared[feature] = pd.to_numeric(
                prepared[feature], errors="coerce"
            ).astype("float64")
        return prepared

    @staticmethod
    def _great_circle_miles(lat1, lon1, lat2, lon2) -> float:
        lat1r, lon1r, lat2r, lon2r = np.radians([lat1, lon1, lat2, lon2])
        a = (
            np.sin((lat2r - lat1r) / 2) ** 2
            + np.cos(lat1r) * np.cos(lat2r)
            * np.sin((lon2r - lon1r) / 2) ** 2
        )
        return float(3958.7613 * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)))

    @staticmethod
    def _reliability_from_recent_support(months: int, departures: int) -> str:
        """Classify evidence from the 12 months available at forecast origin."""
        if months >= 12 and departures >= 12:
            return "Standard"
        if months >= 6 and departures >= 12:
            return "Moderate"
        if months >= 3 and departures >= 3:
            return "Low"
        return "Exploratory — abstain from standard recommendation"
