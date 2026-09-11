"""Pure presentation logic used by FleetFit and its automated checks."""

from __future__ import annotations

import re

import pandas as pd


RELIABILITY_ORDER = {
    "Standard": 0,
    "Moderate": 1,
    "Low": 2,
    "Exploratory — abstain from standard recommendation": 3,
}


def evidence_label(reliability: str) -> tuple[str, str]:
    labels = {
        "Standard": ("Strong support", "standard"),
        "Moderate": ("Established support", "moderate"),
        "Low": ("Emerging support", "low"),
    }
    return labels.get(reliability, ("Exploratory", "exploratory"))


def filter_airports(
    airports: pd.DataFrame,
    query: str,
    *,
    limit: int = 8,
) -> pd.DataFrame:
    """Return a compact airport search result after three typed characters."""
    normalized = str(query or "").strip().casefold()
    if len(normalized) < 3:
        return airports.iloc[0:0].copy()

    searchable = (
        airports["AIRPORT_CODE"].fillna("").astype(str)
        + " "
        + airports["AIRPORT_NAME"].fillna("").astype(str)
        + " "
        + airports["CITY_NAME"].fillna("").astype(str)
        + " "
        + airports["COUNTRY_NAME"].fillna("").astype(str)
    ).str.casefold()
    matches = airports.loc[searchable.str.contains(re.escape(normalized), regex=True)].copy()
    matches["_exact_code"] = matches["AIRPORT_CODE"].astype(str).str.casefold().eq(normalized)
    matches["_starts_code"] = matches["AIRPORT_CODE"].astype(str).str.casefold().str.startswith(normalized)
    matches["_starts_name"] = matches["AIRPORT_NAME"].astype(str).str.casefold().str.startswith(normalized)
    return (
        matches.sort_values(
            ["_exact_code", "_starts_code", "_starts_name", "AIRPORT_CODE"],
            ascending=[False, False, False, True],
        )
        .drop(columns=["_exact_code", "_starts_code", "_starts_name"])
        .head(limit)
        .reset_index(drop=True)
    )


def split_destination_matches(
    matches: pd.DataFrame,
    recent_destination_codes: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    codes = matches["AIRPORT_CODE"].astype(str)
    recent_mask = codes.isin({str(code) for code in recent_destination_codes})
    return (
        matches.loc[recent_mask].reset_index(drop=True),
        matches.loc[~recent_mask].reset_index(drop=True),
    )


def recover_airport_code(airports: pd.DataFrame, query: str) -> str | None:
    """Recover a valid airport code from a persisted display label."""
    text = str(query or "").strip()
    if not text:
        return None

    code_match = re.search(r"\(([A-Z0-9]{3})\)", text.upper())
    candidate = code_match.group(1) if code_match else text.upper()
    valid_codes = set(airports["AIRPORT_CODE"].dropna().astype(str).str.upper())
    return candidate if candidate in valid_codes else None


def route_rankings(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create operational and forecast rankings without mixing their meaning."""
    operated = (
        frame.loc[frame["RECENT_ROUTE_DEPARTURES"] > 0]
        .sort_values(
            ["RECENT_ROUTE_DEPARTURES", "RECENT_ROUTE_MONTHS", "PREDICTED_LF"],
            ascending=[False, False, False],
        )
        .reset_index(drop=True)
    )

    projected = frame.copy()
    projected["_evidence_order"] = (
        projected["RELIABILITY"].map(RELIABILITY_ORDER).fillna(4).astype(int)
    )
    projected = (
        projected.sort_values(
            [
                "RECENT_ROUTE_MONTHS",
                "_evidence_order",
                "PREDICTED_LF",
                "RECENT_ROUTE_DEPARTURES",
            ],
            ascending=[False, True, False, False],
        )
        .drop(columns="_evidence_order")
        .reset_index(drop=True)
    )
    return operated, projected


def aircraft_family(description: str) -> str:
    """Map detailed BTS descriptions to stable, user-facing aircraft families."""
    text = re.sub(r"[^A-Z0-9]+", " ", str(description).upper()).strip()
    rules = [
        (r"A380", "Airbus A380"),
        (r"A350|AIRBUS 350", "Airbus A350"),
        (r"A340", "Airbus A340"),
        (r"A330", "Airbus A330"),
        (r"A321", "Airbus A321"),
        (r"A320", "Airbus A320"),
        (r"A319", "Airbus A319"),
        (r"A220|BD 500", "Airbus A220"),
        (r"B?787", "Boeing 787"),
        (r"B?777", "Boeing 777"),
        (r"B?767", "Boeing 767"),
        (r"B?757", "Boeing 757"),
        (r"B?747", "Boeing 747"),
        (r"B?737", "Boeing 737"),
        (r"717", "Boeing 717"),
        (r"CRJ|CANADAIR RJ", "Bombardier CRJ"),
        (r"ERJ|EMBRAER", "Embraer E-Jet / ERJ"),
        (r"ATR 72", "ATR 72"),
        (r"ATR 42", "ATR 42"),
        (r"DHC8|DASH 8", "De Havilland Dash 8"),
        (r"TWIN OTTER|DHC 6", "De Havilland Twin Otter"),
        (r"CESSNA", "Cessna"),
        (r"PILATUS", "Pilatus"),
        (r"BEECH", "Beechcraft"),
    ]
    for pattern, family in rules:
        if re.search(pattern, text):
            return family
    return str(description).strip() or "Aircraft model"


def average_daily_departures(departures_trailing_12: float, observed_months: int) -> float:
    """Average daily departures over the observed monthly window."""
    months = max(0, int(observed_months or 0))
    departures = max(0.0, float(departures_trailing_12 or 0.0))
    if months == 0:
        return 0.0
    return departures / (months * 365.2425 / 12.0)
