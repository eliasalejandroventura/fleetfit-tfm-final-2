"""FleetFit Streamlit interface for aircraft-model recommendation."""

from __future__ import annotations

import base64
import html
import os
import sys
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

APP_DIRECTORY = Path(__file__).resolve().parent
ASSET_DIRECTORY = APP_DIRECTORY / "assets"
DEFAULT_ARTIFACT_DIRECTORY = APP_DIRECTORY.parent / "03_outputs" / "model"
ARTIFACT_DIRECTORY = Path(
    os.environ.get("TFM_MODEL_ARTIFACT_DIRECTORY", DEFAULT_ARTIFACT_DIRECTORY)
).expanduser().resolve()
sys.path[:0] = [str(APP_DIRECTORY), str(ARTIFACT_DIRECTORY)]

from load_factor_recommender import LoadFactorRecommender  # noqa: E402
from ui_logic import (  # noqa: E402
    aircraft_family,
    average_daily_departures,
    evidence_label,
    filter_airports,
    recover_airport_code,
    route_rankings,
    split_destination_matches,
)

st.set_page_config(
    page_title="FleetFit | TFM FINAL 2",
    page_icon="✈",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def image_data_uri(path: Path) -> str:
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{payload}"


RUNWAY_IMAGE = image_data_uri(ASSET_DIRECTORY / "pexels-runway-aircraft.jpg")
FLEETFIT_LOGO = image_data_uri(ASSET_DIRECTORY / "fleetfit-logo.png")
RESULT_WATERMARK = image_data_uri(ASSET_DIRECTORY / "results-airport-watermark.png")


@st.cache_data(show_spinner=False)
def load_aircraft_image_manifest() -> pd.DataFrame:
    manifest = pd.read_csv(ASSET_DIRECTORY / "aircraft_image_manifest.csv")
    required = {
        "AIRCRAFT_TYPE", "AIRCRAFT_DESCRIPTION", "ASSET_PATH",
        "SOURCE_PAGE", "ASSET_ORIGIN", "STATUS",
    }
    missing = required - set(manifest.columns)
    if missing:
        raise RuntimeError(f"Aircraft image manifest is missing: {sorted(missing)}")
    if manifest["AIRCRAFT_TYPE"].duplicated().any():
        raise RuntimeError("Aircraft image manifest contains duplicate BTS types")
    manifest["LOCAL_PATH"] = manifest["ASSET_PATH"].map(
        lambda value: str((APP_DIRECTORY / str(value)).resolve())
    )
    manifest["FILE_EXISTS"] = manifest["LOCAL_PATH"].map(lambda value: Path(value).is_file())
    coverage = float(manifest["FILE_EXISTS"].mean())
    if coverage < 0.90:
        raise RuntimeError(f"Aircraft image coverage is {coverage:.1%}; at least 90% is required")
    return manifest

AIRLINE_WEBSITES = {
    "AA": "https://www.aa.com/", "AS": "https://www.alaskaair.com/",
    "B6": "https://www.jetblue.com/", "DL": "https://www.delta.com/",
    "F9": "https://www.flyfrontier.com/", "G4": "https://www.allegiantair.com/",
    "HA": "https://www.hawaiianairlines.com/", "NK": "https://www.spirit.com/",
    "UA": "https://www.united.com/", "WN": "https://www.southwest.com/",
    "OO": "https://www.skywest.com/",
    "AC": "https://www.aircanada.com/", "AF": "https://www.airfrance.com/",
    "AM": "https://www.aeromexico.com/", "AV": "https://www.avianca.com/",
    "BA": "https://www.britishairways.com/", "CM": "https://www.copaair.com/",
    "EK": "https://www.emirates.com/", "IB": "https://www.iberia.com/",
    "KL": "https://www.klm.com/", "LA": "https://www.latamairlines.com/",
    "LH": "https://www.lufthansa.com/", "QR": "https://www.qatarairways.com/",
    "VS": "https://www.virginatlantic.com/", "WS": "https://www.westjet.com/",
}
AIRCRAFT_IMAGE_MANIFEST = load_aircraft_image_manifest()
AIRCRAFT_IMAGE_RECORDS = AIRCRAFT_IMAGE_MANIFEST.set_index("AIRCRAFT_TYPE").to_dict("index")


@st.cache_resource(show_spinner="Preparing FleetFit...")
def load_recommender(artifact_directory: str) -> LoadFactorRecommender:
    return LoadFactorRecommender(artifact_directory)


@st.cache_data(show_spinner=False)
def load_airports(artifact_directory: str) -> pd.DataFrame:
    active = load_recommender(artifact_directory).airport_options(active_only=True)
    catalog = pd.read_csv(
        ASSET_DIRECTORY / "airport_catalog.csv", dtype={"AIRPORT_CODE": "string"}
    )
    return active.merge(catalog, on="AIRPORT_CODE", how="left").sort_values(
        "AIRPORT_CODE"
    ).reset_index(drop=True)


st.markdown(
    f"""
    <style>
    :root{{--navy:#062942;--blue:#08789a;--teal:#0aa39e;--cyan:#77d4d0;--focus:#45bfd2;--ink:#10344d;--muted:#617d8f;--line:#cbdfe0;--amber:#f0b84d;--shadow:0 14px 34px rgba(5,43,65,.12)}}
    html,body,[class*="css"]{{font-family:Inter,ui-sans-serif,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;color:var(--ink);letter-spacing:0}}
    .stApp{{background:linear-gradient(145deg,#e9f4f5 0%,#fbfcfb 48%,#edf3f7 100%)}}
    .stApp:has(.result-page-marker){{background-image:linear-gradient(rgba(240,248,249,.91),rgba(248,250,249,.94)),url("{RESULT_WATERMARK}");background-size:cover;background-position:center;background-attachment:fixed}}
    [data-testid="stHeader"],[data-testid="stToolbar"],[data-testid="stFooter"]{{display:none!important;height:0!important;min-height:0!important}}
    [data-testid="stAppViewContainer"],[data-testid="stMain"]{{padding-top:0!important}}
    .block-container{{max-width:none;min-height:100vh;padding:0 1.1rem 0!important}}
    .stMainBlockContainer>[data-testid="stVerticalBlock"]{{min-height:100vh;gap:0!important}}
    .stMainBlockContainer>[data-testid="stVerticalBlock"]:has(.home-hero)>div:has(>.st-key-route_search){{display:flex!important;flex:1 1 auto!important}}
    .fleet-nav{{box-sizing:border-box;width:100%;height:62px;display:flex;align-items:center;justify-content:space-between;padding:0 clamp(18px,3vw,54px);margin:0;background:linear-gradient(105deg,#052a45,#07556e);border-bottom:1px solid rgba(119,212,208,.35);position:fixed;top:0;left:0;right:0;z-index:999;box-shadow:0 7px 18px rgba(5,43,65,.16)}}
    .stElementContainer:has(.fleet-nav){{height:62px!important}}
    .fleet-logo{{display:flex;align-items:center;gap:10px;text-decoration:none!important;color:#fff!important}} .fleet-logo:visited,.fleet-logo .logo-name{{color:#fff!important}}
    .fleet-logo-image{{width:45px;height:45px;object-fit:contain;filter:drop-shadow(0 5px 9px rgba(0,0,0,.22))}}
    .logo-name{{font-size:1.25rem;font-weight:850}} .nav-links{{display:flex;align-items:center;gap:4px}}
    .nav-links a{{color:#dceff1;text-decoration:none;font-weight:700;font-size:.82rem;padding:8px 11px;border-radius:6px}}
    .nav-links a:hover,.nav-links a.active{{background:rgba(119,212,208,.2);color:#fff}}
    .home-hero{{width:100vw;height:245px;box-sizing:border-box;display:flex;align-items:flex-end;overflow:hidden;position:relative;border-radius:0;padding:25px clamp(24px,4vw,72px);margin-left:calc(50% - 50vw);background-image:linear-gradient(90deg,rgba(3,28,48,.94),rgba(7,58,78,.58) 60%,rgba(8,62,81,.06)),url("{RUNWAY_IMAGE}");background-size:cover;background-position:center 53%;box-shadow:var(--shadow)}}
    .stElementContainer:has(.home-hero){{height:245px!important}}
    .home-copy{{max-width:620px;position:relative;z-index:2}} .eyebrow{{font-size:.72rem;text-transform:uppercase;font-weight:850;color:#88e6e2;letter-spacing:.08em}}
    .home-copy h1{{font-size:clamp(2.35rem,4.1vw,3.35rem);line-height:1.04;margin:0 0 15px;color:white}} .home-copy p{{font-size:1rem;line-height:1.45;color:#e5f4f4;max-width:600px;margin:0}}
    .hero-action{{display:inline-block;margin-top:12px;padding:11px 16px;border-radius:7px;background:linear-gradient(135deg,var(--amber),#ffd575);color:#173850!important;text-decoration:none!important;font-weight:850;box-shadow:0 10px 22px rgba(0,0,0,.2)}}
    .route-search-marker{{height:0;overflow:hidden}}
    .st-key-route_search{{box-sizing:border-box;width:100vw;min-width:100vw;height:100%;margin-left:calc(50% - 50vw);padding:20px clamp(24px,4vw,72px) 24px;background:linear-gradient(115deg,#052a45,#083c58 58%,#075b69);color:white;box-shadow:0 12px 30px rgba(5,43,65,.16)}}
    .st-key-route_search h2{{color:#fff!important;font-size:1.35rem;margin:0 0 12px}}
    .st-key-route_search .stSelectbox label{{color:#eaf7f7!important}}
    .st-key-route_search .stSelectbox [data-baseweb="select"]>div{{background:#f7fbfb!important;border:1px solid #86c9cc!important;box-shadow:none!important}}
    .st-key-route_search .stSelectbox [data-baseweb="select"]>div:focus-within{{border-color:var(--focus)!important;box-shadow:0 0 0 3px rgba(69,191,210,.28)!important}}
    .st-key-route_search .stSelectbox [data-baseweb="select"] *{{color:#10344d!important;-webkit-text-fill-color:#10344d!important}}
    .st-key-route_search .stButton>button{{white-space:nowrap!important;font-size:.88rem!important;padding-left:.65rem!important;padding-right:.65rem!important}}
    .st-key-route_search .stButton>button:not([data-testid="stBaseButton-primary"]):hover,.st-key-route_search .stButton>button:not([data-testid="stBaseButton-primary"]):focus{{border-color:var(--focus)!important;color:#075b69!important;box-shadow:0 0 0 3px rgba(69,191,210,.2)!important}}
    .st-key-route_search .search-hint,.st-key-route_search [data-testid="stCaptionContainer"]{{color:#b9d9df!important}}
    .recent-searches{{margin-top:12px}}
    .recent-searches-title{{margin-bottom:6px;color:#b9d9df;font-size:.69rem;font-weight:850;text-transform:uppercase;letter-spacing:.06em}}
    .st-key-route_search [class*="st-key-recent_route_"] button{{min-height:38px!important;border:1px solid rgba(134,201,204,.64)!important;background:rgba(255,255,255,.08)!important;color:#edfafa!important;box-shadow:none!important}}
    .st-key-route_search [class*="st-key-recent_route_"] button:hover,.st-key-route_search [class*="st-key-recent_route_"] button:focus{{background:rgba(124,222,220,.18)!important;color:#fff!important}}
    .fleet-footer{{box-sizing:border-box;width:100vw;margin:0 0 0 calc(50% - 50vw);padding:17px 20px;text-align:center;background:#041f34;color:#b9d4dc;font-size:.68rem;letter-spacing:.02em}}
    .stElementContainer:has(.fleet-footer){{margin-top:0!important}}
    .stElementContainer:has(.fleet-footer) [data-testid="stMarkdownContainer"]{{margin-bottom:0!important}}
    .plane-page-marker{{height:0;overflow:hidden}}
    .stApp:has(.plane-page-marker) .stMainBlockContainer>[data-testid="stVerticalBlock"]{{min-height:100vh;display:flex!important;flex-direction:column!important}}
    .stApp:has(.plane-page-marker) .stElementContainer:has(.fleet-footer){{margin-top:auto!important}}
    .home-band{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-top:13px}} .home-note{{padding:17px;background:rgba(255,255,255,.8);border:1px solid var(--line);border-radius:7px}}
    .home-note strong{{display:block;color:var(--navy);font-size:.95rem;margin-bottom:4px}} .home-note span{{color:var(--muted);font-size:.82rem;line-height:1.4}}
    .result-page-marker{{height:0;overflow:hidden}}
    .page-heading{{display:flex;justify-content:space-between;align-items:end;gap:14px;margin:5px 0 10px}} .page-heading h1{{font-size:1.55rem;margin:0;color:var(--navy)}} .page-heading p{{margin:3px 0 0;color:var(--muted);font-size:.86rem}}
    .stApp:has(.result-page-marker) .page-heading{{box-sizing:border-box;width:min(1500px,calc(100% - 24px));margin:12px auto 7px;padding:12px 16px;background:rgba(249,252,252,.84);border:1px solid rgba(169,204,207,.76);border-radius:8px;box-shadow:0 8px 24px rgba(5,43,65,.08);backdrop-filter:blur(8px)}}
    .stApp:has(.result-page-marker) .page-heading h1{{font-size:clamp(1.45rem,2.1vw,1.9rem)}}
    .stApp:has(.result-page-marker) .stElementContainer:has(.page-heading){{padding-bottom:12px!important}}
    .stApp:has(.result-page-marker) [data-testid="stExpander"]{{width:min(1500px,calc(100% - 24px));margin:18px auto 8px;background:transparent!important;border:0!important;box-shadow:none!important}}
    .stApp:has(.result-page-marker) [data-testid="stExpander"] summary{{box-sizing:border-box;width:max-content;min-width:0;padding:8px 14px!important;border:1px solid #148eae!important;border-radius:7px!important;background:linear-gradient(135deg,#158daf,#38bed0)!important;box-shadow:0 6px 16px rgba(8,120,154,.22)!important;transition:transform .16s ease,box-shadow .16s ease}}
    .stApp:has(.result-page-marker) [data-testid="stExpander"] summary:hover{{transform:translateY(-1px);box-shadow:0 9px 20px rgba(8,120,154,.28)!important}}
    .stApp:has(.result-page-marker) [data-testid="stExpander"] summary,
    .stApp:has(.result-page-marker) [data-testid="stExpander"] summary *{{color:#fff!important;font-weight:800!important}}
    .forecast-tag{{font-size:.73rem;font-weight:800;color:#166c78;background:#dff2f0;padding:7px 9px;border-radius:6px;white-space:nowrap}}
    [data-testid="stVerticalBlockBorderWrapper"]{{border-color:var(--line)!important;border-radius:8px!important;background:rgba(255,255,255,.86)!important;box-shadow:0 8px 22px rgba(14,63,85,.07)!important}}
    .stTextInput label{{color:var(--ink)!important;font-weight:750!important}} .stTextInput input{{border-radius:7px!important;border-color:#a9c9cd!important;background:#fff!important;color:var(--ink)!important;-webkit-text-fill-color:var(--ink)!important}}
    [data-baseweb="select"]>div{{border-color:#a9c9cd!important;background:#fff!important;color:var(--ink)!important}}
    [data-baseweb="select"]>div:focus-within,[data-baseweb="select"]:focus-within>div{{border-color:var(--focus)!important;box-shadow:0 0 0 3px rgba(69,191,210,.24)!important}}
    [data-baseweb="select"] input,[data-baseweb="select"] span{{color:var(--ink)!important;-webkit-text-fill-color:var(--ink)!important}}
    [data-baseweb="popover"],[role="listbox"]{{background:#fff!important;color:var(--ink)!important}}
    [data-baseweb="popover"] [role="option"],[role="listbox"] [role="option"]{{background:#fff!important;color:var(--ink)!important}}
    [data-baseweb="popover"] [role="option"] *,[role="listbox"] [role="option"] *{{color:var(--ink)!important;-webkit-text-fill-color:var(--ink)!important}}
    [role="listbox"] [aria-selected="true"],[role="option"]:hover{{background:#d8f2f5!important;color:#06344d!important}}
    .stButton>button{{border-radius:7px!important;font-weight:750!important}} [data-testid="stBaseButton-primary"]{{border:0!important;color:white!important;background:linear-gradient(110deg,var(--blue),var(--teal))!important;box-shadow:0 8px 18px rgba(18,104,139,.24)!important}}
    .search-hint{{font-size:.72rem;color:var(--muted);margin-top:-5px}} .search-section{{font-size:.66rem;text-transform:uppercase;font-weight:850;color:#668393;margin:5px 0 3px}}
    .result-head{{display:flex;justify-content:space-between;align-items:center;margin:5px 0 7px}} .result-head h2{{font-size:1.05rem;margin:0;color:var(--navy)}} .route-pill{{font-size:.78rem;color:#316075;font-weight:750}}
    .st-key-decision_grid{{width:min(1500px,calc(100% - 24px));margin:0 auto}}
    .st-key-decision_grid [data-testid="stHorizontalBlock"]{{align-items:stretch!important}}
    .st-key-decision_grid [data-testid="column"]{{display:flex!important;flex-direction:column!important}}
    .route-panel{{box-sizing:border-box;height:350px;min-height:350px;border:1px solid var(--line);border-radius:8px;background:rgba(255,255,255,.94);overflow:hidden;box-shadow:var(--shadow)}}
    .st-key-decision_grid [data-testid="stVerticalBlockBorderWrapper"]{{box-sizing:border-box;height:350px!important;min-height:350px!important;overflow:hidden;background:rgba(255,255,255,.94)!important}}
    .st-key-decision_grid [data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .map-title){{box-sizing:border-box;height:350px!important;min-height:350px!important;gap:0!important;padding:0!important;overflow:hidden;border:1px solid var(--line);border-radius:8px;background:rgba(255,255,255,.94)!important;box-shadow:var(--shadow)}}
    .panel-title{{box-sizing:border-box;height:57px;min-height:57px;padding:10px 13px 8px;display:flex;align-items:flex-start;justify-content:space-between;gap:10px;border-bottom:1px solid #dce9e9;background:linear-gradient(90deg,rgba(229,245,246,.74),rgba(255,255,255,.35))}}
    .panel-title strong{{font-size:.86rem;line-height:1.25;color:var(--navy)}} .panel-title span{{font-size:.62rem;line-height:1.3;color:var(--muted);text-align:right}}
    .ranking-panel{{padding:0 12px 8px}} .ranking-panel .panel-title{{margin:0 -12px 3px;padding-left:12px;padding-right:12px}}
    .map-title{{font-size:.86rem;font-weight:800;color:var(--navy)}}
    .rank-row{{display:grid;grid-template-columns:24px minmax(0,1fr) 62px;gap:7px;align-items:center;min-height:0;padding:3px 2px;border-bottom:1px solid #dfebeb}} .rank-row:last-child{{border-bottom:0}}
    .rank-number{{width:22px;height:22px;border-radius:50%;display:grid;place-items:center;background:#dbeff0;color:#0c5264;font-weight:850;font-size:.68rem}}
    .rank-model{{font-weight:800;font-size:.76rem;color:var(--navy);line-height:1.12;text-decoration:none!important;display:block}} .rank-model:hover{{color:var(--teal)}} .rank-family{{font-size:.61rem;line-height:1.1;color:#367185;margin-top:1px}}
    .rank-meta{{font-size:.59rem;line-height:1.12;color:var(--muted);margin-top:2px}} .rank-lf{{text-align:right;font-weight:850;font-size:.88rem;color:#0a6b75}}
    .support-dot{{display:inline-block;width:6px;height:6px;border-radius:50%;margin-right:4px}} .standard{{background:#1da77d}} .moderate{{background:#42a9b8}} .low{{background:#e3a832}} .exploratory{{background:#df765f}}
    .kpi-grid{{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:7px;margin-top:9px}} .kpi{{padding:9px 10px;background:rgba(255,255,255,.84);border:1px solid var(--line);border-radius:7px;min-height:63px}}
    .stElementContainer:has(.kpi-grid){{padding-bottom:14px!important}}
    .kpi-label{{font-size:.61rem;text-transform:uppercase;font-weight:800;color:#6b8392}} .kpi-value{{font-size:.92rem;font-weight:850;color:var(--navy);margin-top:4px}}
    .section-title{{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:7px}} .section-title h3{{font-size:.9rem;margin:0;color:var(--navy)}} .section-title span{{font-size:.64rem;color:var(--muted)}}
    .history-chart{{width:100%;overflow:hidden}} .history-chart svg{{display:block;width:100%;height:auto;min-height:190px}}
    .chart-axis-label{{font-size:12px;fill:#607b8e}} .chart-axis-title{{font-size:13px;font-weight:750;fill:#31566d}}
    .direct-flight-marker{{height:0;overflow:hidden}}
    [data-testid="stVerticalBlockBorderWrapper"]:has(.direct-flight-marker){{border:1px solid #46b9dc!important;background:linear-gradient(135deg,rgba(218,246,255,.98),rgba(222,251,247,.96))!important;box-shadow:0 14px 30px rgba(0,113,214,.18)!important}}
    .direct-flight-intro{{margin:13px 0 7px;padding:10px 12px;border:1px solid #005fe0;border-radius:7px;background:linear-gradient(115deg,#006cff,#009dff);color:#fff;font-size:.78rem;font-weight:850;letter-spacing:.01em;box-shadow:0 8px 18px rgba(0,108,255,.28)}}
    .operator-row{{display:flex;align-items:center;gap:8px;padding:7px 2px;border-bottom:1px solid #cae3e4}} .operator-row:last-child{{border-bottom:0}} .operator-logo{{width:31px;height:31px;object-fit:contain;background:white;border-radius:5px;border:1px solid #badbdd;padding:3px;transition:transform .16s ease,box-shadow .16s ease}}
    .operator-logo:hover{{transform:translateY(-1px);box-shadow:0 5px 12px rgba(5,43,65,.16)}}
    .operator-info{{min-width:0;flex:1}} .operator-name{{font-size:.72rem;font-weight:780;color:var(--navy);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}} .operator-name a{{color:#056da3;text-decoration:underline;text-decoration-thickness:1px;text-underline-offset:2px}} .operator-name a::after{{content:" ↗"}} .operator-name a:hover{{color:var(--teal)}} .operator-meta{{font-size:.61rem;color:var(--muted)}} .operator-share{{font-size:.75rem;font-weight:850;color:#167c82}}
    .stApp:has(.plane-page-marker) .st-key-plane_controls,
    .stApp:has(.plane-page-marker) .profile-hero,
    .stApp:has(.plane-page-marker) .profile-facts,
    .stApp:has(.plane-page-marker) .profile-operators{{box-sizing:border-box;width:min(1320px,calc(100% - 32px));margin-left:auto;margin-right:auto}}
    .st-key-plane_controls{{margin-top:20px;margin-bottom:14px}}
    .st-key-plane_controls [data-testid="stVerticalBlock"]{{gap:.35rem!important}}
    .st-key-plane_controls h3{{margin:0!important;padding:0!important;color:var(--navy)!important;font-size:1.65rem!important;line-height:1.2!important}}
    .st-key-plane_controls [data-testid="stCaptionContainer"],.st-key-plane_controls [data-testid="stCaptionContainer"] p{{margin:0!important;color:#4e6f82!important}}
    .st-key-plane_controls [data-testid="stSelectbox"]{{margin:5px 0 0!important}}
    .st-key-plane_controls [data-testid="stHorizontalBlock"]{{display:grid!important;grid-template-columns:minmax(0,1fr) 42px!important;gap:8px!important}}
    .st-key-plane_controls [data-testid="column"]{{width:auto!important;min-width:0!important}}
    .st-key-plane_controls button[aria-label="Clear value"]{{display:none!important}}
    .st-key-plane_clear_model button{{width:100%!important;height:40px!important;padding:0!important;border:1px solid #0b8fb3!important;border-radius:7px!important;background:#0b8fb3!important;color:#fff!important;font-size:1.35rem!important;font-weight:700!important;line-height:1!important;box-shadow:0 4px 10px rgba(9,99,132,.2)!important}}
    .st-key-plane_clear_model button:hover{{background:#087a9a!important;border-color:#087a9a!important}}
    .profile-hero{{display:grid;grid-template-columns:minmax(270px,.78fr) minmax(0,1.22fr);gap:28px;align-items:center;background:linear-gradient(125deg,#082a45,#0c5878 58%,#12958f);padding:24px;border-radius:8px;color:white;box-shadow:0 14px 32px rgba(5,43,65,.18);overflow:hidden}}
    .profile-hero>div:first-child{{align-self:center;padding-left:8px;min-width:0}} .profile-hero h1{{font-size:clamp(2rem,3.4vw,3rem);line-height:1.04;margin:8px 0 12px}} .profile-hero p{{color:#d8eeee;margin:0;font-size:.92rem;overflow-wrap:anywhere}}
    .aircraft-visual{{min-width:0}}
    .aircraft-photo{{box-sizing:border-box;width:100%;height:auto;aspect-ratio:8/5;background:#eef5f6;border:1px solid rgba(255,255,255,.8);border-radius:7px;padding:0;overflow:hidden;display:grid;place-items:center}}
    .aircraft-photo img{{display:block;width:100%;height:100%;object-fit:contain;object-position:center;transform:none}} .photo-credit{{font-size:.55rem;color:#d5ebee;text-align:right;margin-top:4px}} .photo-credit a{{color:#e7f7f8}} .image-na{{color:#607b8e;font-weight:800;text-align:center}} .profile-facts{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-top:10px}}
    .profile-fact{{min-height:72px;padding:11px 13px;background:rgba(255,255,255,.94);border:1px solid var(--line);border-radius:7px;box-shadow:0 6px 16px rgba(5,43,65,.05)}} .profile-fact span{{display:block;font-size:.62rem;text-transform:uppercase;color:var(--muted);font-weight:800}} .profile-fact strong{{display:block;font-size:1rem;color:var(--navy);margin-top:5px}}
    .profile-operators{{box-sizing:border-box;margin-top:10px;margin-bottom:20px;padding:15px 18px 17px;background:rgba(255,255,255,.94);border:1px solid var(--line);border-radius:8px;box-shadow:0 8px 22px rgba(14,63,85,.07);overflow:visible}}
    .profile-operators .section-title{{gap:16px;flex-wrap:wrap;margin-bottom:10px}} .operator-chip-list{{display:flex;flex-wrap:wrap;gap:7px}} .operator-chip{{display:inline-flex;align-items:center;min-height:30px;padding:6px 10px;border:1px solid #b8dadd;border-radius:999px;background:#edf8f8;color:#174d63;font-size:.72rem;font-weight:720}}
    .about-panel{{max-width:850px;padding:25px;background:rgba(255,255,255,.84);border:1px solid var(--line);border-radius:8px;box-shadow:var(--shadow)}} .about-panel h1{{margin-top:0;color:var(--navy)}} .about-panel p{{line-height:1.6;color:#395d72}}
    @media(max-width:1100px){{.kpi-grid{{grid-template-columns:repeat(3,1fr)}}}}
    @media(max-width:760px){{.nav-links a{{padding:7px 6px;font-size:.7rem}}.home-band,.profile-facts{{grid-template-columns:1fr 1fr}}.profile-hero{{grid-template-columns:1fr;gap:16px;padding:18px}}.profile-hero>div:first-child{{padding-left:0}}.route-panel,.st-key-decision_grid [data-testid="stVerticalBlockBorderWrapper"]{{height:330px!important;min-height:330px!important}}}}
    @media(max-width:520px){{.block-container{{padding:0 .55rem 0!important}}.fleet-nav{{height:58px;padding:0 9px}}.stElementContainer:has(.fleet-nav){{height:58px!important}}.fleet-logo-image{{width:41px;height:41px}}.logo-name{{font-size:1rem}}.nav-links{{gap:0}}.nav-links a{{font-size:.57rem;padding:6px 4px}}.home-hero{{height:215px;padding:20px 16px}}.stElementContainer:has(.home-hero){{height:215px!important}}.home-copy h1{{font-size:2rem;line-height:1.05;margin-bottom:12px}}.home-copy p{{font-size:.84rem;line-height:1.4}}.st-key-route_search{{padding:17px 14px 20px}}.home-band,.profile-facts{{grid-template-columns:1fr}}.kpi-grid{{grid-template-columns:repeat(2,1fr)}}.page-heading,.result-head{{align-items:flex-start;flex-direction:column;gap:5px}}.stApp:has(.plane-page-marker) .st-key-plane_controls,.stApp:has(.plane-page-marker) .profile-hero,.stApp:has(.plane-page-marker) .profile-facts,.stApp:has(.plane-page-marker) .profile-operators{{width:100%}}.st-key-plane_controls{{margin-top:14px;margin-bottom:10px}}.profile-hero{{padding:16px}}.stApp:has(.result-page-marker) .page-heading,.st-key-decision_grid,.stApp:has(.result-page-marker) [data-testid="stExpander"]{{width:100%}}.history-chart svg{{min-height:155px}}.chart-axis-label{{font-size:11px}}}}

    /* Responsive layout: keep navigation readable and the footer at the page end. */
    [data-testid="stMain"]{{overflow-x:hidden}}
    .stMainBlockContainer>[data-testid="stVerticalBlock"]{{display:flex!important;flex-direction:column!important}}
    .stElementContainer:has(.fleet-footer){{margin-top:auto!important}}
    .about-panel{{box-sizing:border-box;width:min(900px,100%);max-width:900px;margin:20px auto;padding:clamp(20px,3vw,40px)}}
    .about-panel h1{{font-size:clamp(1.7rem,3vw,2.4rem);line-height:1.15}}
    .section-title{{flex-wrap:wrap;gap:6px}}
    .ranking-panel{{height:auto;min-height:350px;overflow:visible}}
    .operator-monogram{{display:grid;place-items:center;flex-shrink:0;width:37px;height:37px;background:#e2f1f3;border:1px solid #badbdd;border-radius:5px;color:#10344d;font-size:.75rem;font-weight:800}}
    @media(max-width:900px){{
      .st-key-decision_grid [data-testid="stHorizontalBlock"]{{flex-direction:column!important}}
      .st-key-decision_grid [data-testid="stColumn"]{{width:100%!important;flex:1 1 100%!important}}
      .ranking-panel{{height:auto!important;min-height:0!important;padding-bottom:12px}}
      .rank-row{{padding:8px 2px}} .rank-model{{font-size:.88rem}} .rank-meta{{font-size:.72rem;line-height:1.3}} .rank-family{{font-size:.72rem}}
      .panel-title{{height:auto;min-height:57px}} .panel-title span{{font-size:.72rem}}
    }}
    @media(max-width:520px){{
      .fleet-nav{{height:98px;flex-direction:column;justify-content:center;gap:4px;padding:5px 9px}}
      .stElementContainer:has(.fleet-nav){{height:98px!important}}
      .fleet-logo-image{{width:32px;height:32px}} .nav-links{{width:100%;justify-content:center;gap:4px}}
      .nav-links a{{font-size:.75rem;padding:10px 8px;min-height:38px;box-sizing:border-box}}
      .operator-name{{font-size:.85rem;white-space:normal}} .operator-meta{{font-size:.75rem}}
      .photo-credit{{font-size:.75rem}} .profile-fact span{{font-size:.75rem}}
    }}
    .history-chart-mobile{{display:none}} .chart-axis-label{{font-size:14px}}
    @media(max-width:760px){{.history-chart-desktop{{display:none}}.history-chart-mobile{{display:block}}.history-chart-mobile .chart-axis-label,.history-chart-mobile .chart-axis-title{{font-size:15px}}}}
    .st-key-route_controls .stButton>button{{white-space:nowrap!important;min-height:40px}}
    .stApp:has(.result-page-marker) .recent-searches-title{{color:#42687d}}
    @media(max-width:900px){{.st-key-route_controls [data-testid="stHorizontalBlock"]{{flex-direction:column!important}}.st-key-route_controls [data-testid="stColumn"]{{width:100%!important;flex:1 1 100%!important}}}}
    [data-testid="stMain"]{{container-type:inline-size}}
    .home-hero,.st-key-route_search,.fleet-footer{{width:100cqw;min-width:0;margin-left:calc(50% - 50cqw)}}
    .st-key-plane_controls [data-testid="stColumn"]{{width:100%!important;min-width:0!important;flex:1!important}}
    .stElementContainer:has(.profile-hero),.stElementContainer:has(.profile-facts){{padding-bottom:12px}}
    @media(max-width:360px){{.nav-links a{{font-size:.7rem;padding:10px 6px}}}}
    @media(min-width:901px){{.aircraft-photo{{height:340px;max-height:none;aspect-ratio:auto}}.aircraft-photo img{{min-height:0}}}}
    .stElementContainer:has(.about-panel){{flex:1;display:flex;align-items:center;justify-content:center}}
    [data-testid="stAlertContainer"]:has([data-testid="stAlertContentWarning"]){{background:#fff3cd!important;border:1px solid #d8b65b!important}}
    [data-testid="stAlertContentWarning"],[data-testid="stAlertContentWarning"] p{{color:#634700!important;-webkit-text-fill-color:#634700!important}}
    .st-key-route_search{{min-width:100cqw;flex-shrink:0}}
    .mapboxgl-ctrl-attrib{{background:#f8fbfc!important;color:#15364a!important}}
    .mapboxgl-ctrl-attrib a{{color:#15364a!important}}
    .mapboxgl-ctrl-group button{{background-color:#fff!important;color:#10344d!important}}
    .mapboxgl-ctrl-group .mapboxgl-ctrl-icon{{filter:none!important}}
    </style>
    """,
    unsafe_allow_html=True,
)


def current_view() -> str:
    view = str(st.query_params.get("view", "fleet")).lower()
    return view if view in {"home", "fleet", "plane", "about"} else "home"


def render_navigation(view: str) -> None:
    links = [("fleet", "Fleet Recommendation"), ("plane", "Plane Insights"), ("about", "About")]
    nav = "".join(f'<a class="{"active" if view == target else ""}" href="?view={target}" target="_self">{label}</a>' for target, label in links)
    st.markdown(
        '<nav class="fleet-nav"><a class="fleet-logo" href="?view=fleet" target="_self">'
        f'<img class="fleet-logo-image" src="{FLEETFIT_LOGO}" alt="FleetFit logo">'
        '<span class="logo-name">FleetFit <small style="display:block;font-size:.55rem;letter-spacing:.08em">TFM FINAL 2</small></span></a>'
        f'<div class="nav-links">{nav}</div></nav>',
        unsafe_allow_html=True,
    )


def airport_label(code: str) -> str:
    record = airport_lookup.get(code, {})
    name, city = record.get("AIRPORT_NAME"), record.get("CITY_NAME")
    if pd.notna(name) and str(name).strip():
        suffix = f" · {city}" if pd.notna(city) and str(city).strip() else ""
        return f"{name} ({code}){suffix}"
    return f"{city} ({code})" if pd.notna(city) and str(city).strip() else code


def airport_name(code: str) -> str:
    name = airport_lookup.get(code, {}).get("AIRPORT_NAME")
    return str(name) if pd.notna(name) and str(name).strip() else code


def render_airport_search(label: str, prefix: str, recent_codes: set[str] | None = None) -> str | None:
    codes = airports["AIRPORT_CODE"].astype(str).tolist()
    recent_codes = recent_codes or set()
    if recent_codes:
        codes.sort(key=lambda code: (code not in recent_codes, airport_label(code).casefold()))
    else:
        codes.sort(key=lambda code: airport_label(code).casefold())

    def option_label(code: str) -> str:
        marker = "Recent direct · " if code in recent_codes else ""
        return marker + airport_label(code)

    selected = st.selectbox(
        label,
        options=codes,
        index=None,
        key=f"{prefix}_code",
        format_func=option_label,
        placeholder="Start typing an airport name, city or code",
    )
    return str(selected) if selected else None


def _remember_route(origin: str, destination: str) -> None:
    route = {"origin": origin, "destination": destination}
    previous = st.session_state.get("recent_routes", [])
    st.session_state["recent_routes"] = [
        route,
        *[
            item for item in previous
            if item.get("origin") != origin or item.get("destination") != destination
        ],
    ][:3]


def _restore_recent_route(origin: str, destination: str) -> None:
    st.session_state["origin_code"] = origin
    st.session_state["destination_code"] = destination
    st.session_state["submitted_route"] = {"origin": origin, "destination": destination}


def render_recent_routes() -> None:
    routes = st.session_state.get("recent_routes", [])[:3]
    if not routes:
        return
    st.markdown(
        '<div class="recent-searches"><div class="recent-searches-title">Recent searches</div></div>',
        unsafe_allow_html=True,
    )
    columns = st.columns(3, gap="small")
    for index, route in enumerate(routes):
        origin, destination = str(route["origin"]), str(route["destination"])
        with columns[index]:
            st.button(
                f"{origin} → {destination}",
                key=f"recent_route_{index}",
                on_click=_restore_recent_route,
                args=(origin, destination),
                use_container_width=True,
            )


def ranking_html(frame: pd.DataFrame, title: str, subtitle: str, metric: str) -> str:
    rows = []
    for rank, row in enumerate(frame.head(5).itertuples(index=False), start=1):
        label, css_class = evidence_label(str(row.RELIABILITY))
        family = aircraft_family(str(row.AIRCRAFT_DESCRIPTION))
        href = f"?view=plane&aircraft={int(row.AIRCRAFT_TYPE)}"
        if metric == "departures":
            primary = f"{int(row.RECENT_ROUTE_DEPARTURES):,} dep."
            secondary = f"Observed LF {float(row.RECENT_OBSERVED_LF):.1%}" if pd.notna(row.RECENT_OBSERVED_LF) else "Observed LF N/A"
        else:
            primary = f"{float(row.PREDICTED_LF):.1%}"
            secondary = f"{int(row.RECENT_ROUTE_DEPARTURES):,} recent departures"
        rows.append('<div class="rank-row">' f'<div class="rank-number">{rank}</div><div>' f'<a class="rank-model" href="{href}" target="_self">{html.escape(str(row.AIRCRAFT_DESCRIPTION))}</a>' f'<div class="rank-family">{html.escape(family)}</div>' f'<div class="rank-meta"><span class="support-dot {css_class}"></span>{label} · {int(row.RECENT_ROUTE_MONTHS)}/12 months · {html.escape(secondary)}</div>' f'</div><div class="rank-lf">{primary}</div></div>')
    if not rows:
        message = (
            "No direct flights were recorded for this route in the last 12 observed months."
            if metric == "departures"
            else "No aircraft meets the evidence and projection requirements."
        )
        rows.append(f'<div class="rank-meta">{message}</div>')
    return '<div class="route-panel ranking-panel"><div class="panel-title">' f'<strong>{html.escape(title)}</strong><span>{html.escape(subtitle)}</span></div>' + "".join(rows) + "</div>"


def format_duration(minutes: float) -> str:
    if pd.isna(minutes) or minutes <= 0:
        return "N/A"
    hours, remaining = divmod(int(round(minutes)), 60)
    return f"{hours}h {remaining:02d}m" if hours else f"{remaining}m"


def kpi_html(overview: dict) -> str:
    daily = average_daily_departures(overview["departures_trailing_12"], overview["observed_months"])
    facts = [("Flight time", format_duration(overview["average_flight_minutes"])), ("Distance", f'{overview["distance_miles"]:,.0f} mi'), ("Avg daily departures", f"{daily:.1f}"), ("Airlines", str(overview["carrier_count"])), ("Aircraft models", str(overview["aircraft_count"])), ("Observed window", f'{overview["observed_months"]}/12 mo')]
    return '<div class="kpi-grid">' + "".join(f'<div class="kpi"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div></div>' for label, value in facts) + "</div>"


def operator_html(frame: pd.DataFrame) -> str:
    rows = []
    for row in frame.itertuples(index=False):
        share = float(row.CAPACITY_SHARE)
        if not np.isfinite(share) or share < .01:
            continue
        code = str(row.CARRIER_CODE).strip().upper()
        website = AIRLINE_WEBSITES.get(code)
        logo_path = ASSET_DIRECTORY / "airlines" / f"{code}.png"
        if logo_path.is_file():
            logo = image_data_uri(logo_path)
            image = f'<img class="operator-logo" src="{logo}" alt="{html.escape(code)} logo">'
        else:
            image = f'<span class="operator-monogram" aria-label="{html.escape(code)}">{html.escape(code)}</span>' 
        if website:
            image = f'<a href="{website}" target="_blank" rel="noopener">{image}</a>'
        carrier_name = html.escape(str(row.CARRIER_NAME))
        name = (
            f'<a href="{website}" target="_blank" rel="noopener">{carrier_name}</a>'
            if website else carrier_name
        )
        rows.append('<div class="operator-row">' + image + '<div class="operator-info">' f'<div class="operator-name">{name}</div>' f'<div class="operator-meta">{html.escape(str(row.BUSINESS_MODEL))} · {int(round(float(row.DEPARTURES))):,} departures</div>' f'</div><div class="operator-share">{share:.1%}</div></div>')
    return "".join(rows) if rows else '<div class="operator-meta">No material recent operator is available.</div>'


def route_deck(overview: dict, origin: str, destination: str) -> pdk.Deck:
    source = [float(overview["origin_longitude"]), float(overview["origin_latitude"])]
    target = [float(overview["destination_longitude"]), float(overview["destination_latitude"])]
    # Keep both endpoints in the same world copy across the date line.
    longitude_delta = (target[0] - source[0] + 180) % 360 - 180
    midpoint = [(source[0] + longitude_delta / 2 + 180) % 360 - 180, (source[1] + target[1]) / 2]
    # Place point/text markers in the visible world copy used by the map center.
    for coordinates in (source, target):
        coordinates[0] = midpoint[0] + (coordinates[0] - midpoint[0] + 180) % 360 - 180
    span = max(abs(longitude_delta), abs(source[1] - target[1]))
    zoom = float(np.clip(5.0 - np.log2(max(span, 1.0)), 1.2, 4.5))
    points = pd.DataFrame([{"airport": origin, "coordinates": source, "color": [240,184,77]}, {"airport": destination, "coordinates": target, "color": [66,202,197]}])
    # Sample the great-circle line in one world copy so it stays continuous at 180°.
    def unit_vector(point):
        longitude, latitude = np.radians(point)
        return np.array([np.cos(latitude) * np.cos(longitude), np.cos(latitude) * np.sin(longitude), np.sin(latitude)])
    start, finish = unit_vector(source), unit_vector(target)
    angle = float(np.arccos(np.clip(np.dot(start, finish), -1, 1)))
    path = []
    for fraction in np.linspace(0, 1, 65):
        if abs(np.sin(angle)) < 1e-8:
            position = (1 - fraction) * np.array(source) + fraction * np.array(target)
        else:
            vector = (np.sin((1 - fraction) * angle) * start + np.sin(fraction * angle) * finish) / np.sin(angle)
            longitude = float(np.degrees(np.arctan2(vector[1], vector[0])))
            latitude = float(np.degrees(np.arctan2(vector[2], np.hypot(vector[0], vector[1]))))
            position = [midpoint[0] + (longitude - midpoint[0] + 180) % 360 - 180, latitude]
        path.append([float(position[0]), float(position[1])])
    arcs = [{"path": path}]
    return pdk.Deck(views=[pdk.View(type="MapView", controller=True, repeat=True)], map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json", initial_view_state=pdk.ViewState(longitude=midpoint[0], latitude=midpoint[1], zoom=zoom, pitch=0), layers=[pdk.Layer("PathLayer", arcs, get_path="path", get_color=[119,212,208], get_width=4, width_min_pixels=4, width_max_pixels=4, joint_rounded=True, cap_rounded=True, wrap_longitude=False), pdk.Layer("ScatterplotLayer", points, get_position="coordinates", get_fill_color="color", get_radius=8, radius_min_pixels=7, radius_max_pixels=13, wrap_longitude=False, stroked=True, get_line_color=[255,255,255], line_width_min_pixels=2, pickable=True), pdk.Layer("TextLayer", points, get_position="coordinates", get_text="airport", get_color=[255,255,255], get_size=16, get_pixel_offset=[0,-18], wrap_longitude=False)], tooltip={"text": "{airport}"})


def render_history_chart(history: pd.DataFrame) -> None:
    if history.empty:
        st.info("No recent route load-factor history is available.")
        return
    chart = history.copy()
    chart["MONTH_LABEL"] = pd.to_datetime(chart["SERVICE_DATE"]).dt.strftime("%b %Y")
    chart["LF_PERCENT"] = chart["LOAD_FACTOR"] * 100
    lower, upper = max(0, float(chart.LF_PERCENT.min()) - 4), min(100, float(chart.LF_PERCENT.max()) + 4)
    if upper - lower < 10:
        midpoint = (upper + lower) / 2
        lower, upper = max(0, midpoint - 5), min(100, midpoint + 5)
    def draw_chart(width: int, mobile: bool) -> str:
        left, right, top, bottom = 66.0, float(width - 20), 18.0, 192.0
        span = max(upper - lower, 1.0)
        x_step = (right - left) / max(len(chart) - 1, 1)
        points = []
        circles = []
        x_labels = []
        for index, row in enumerate(chart.itertuples(index=False)):
            x = left + index * x_step
            y = bottom - ((float(row.LF_PERCENT) - lower) / span) * (bottom - top)
            points.append(f"{x:.1f},{y:.1f}")
            label = html.escape(str(row.MONTH_LABEL))
            circles.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="#149c98" '
                f'stroke="#ffffff" stroke-width="2"><title>{label}: '
                f'{float(row.LF_PERCENT):.1f}%</title></circle>'
            )
            x_labels.append(
                f'<text x="{x:.1f}" y="218" transform="rotate(-32 {x:.1f} 218)" '
                f'text-anchor="end" class="chart-axis-label">{label if not mobile or index % 3 == 0 or index == len(chart)-1 else ""}</text>'
            )
        tick_lines, tick_labels = [], []
        for value in np.linspace(lower, upper, 5):
            y = bottom - ((float(value) - lower) / span) * (bottom - top)
            tick_lines.append(
                f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" '
                'stroke="#dce9e9" stroke-width="1"/>'
            )
            tick_labels.append(
                f'<text x="56" y="{y + 4:.1f}" text-anchor="end" '
                f'class="chart-axis-label">{value:.0f}%</text>'
            )
        area_points = f"{left:.1f},{bottom:.1f} " + " ".join(points) + f" {right:.1f},{bottom:.1f}"
        svg = (
            f'<div class="history-chart history-chart-{"mobile" if mobile else "desktop"}"><svg viewBox="0 0 {width} 250" '
            'role="img" aria-label="Trailing twelve-month route load factor">'
            f'<defs><linearGradient id="lfArea{width}" x1="0" y1="0" x2="0" y2="1">'
            '<stop offset="0%" stop-color="#56c8c5" stop-opacity="0.38"/>'
            '<stop offset="100%" stop-color="#56c8c5" stop-opacity="0.04"/>'
            '</linearGradient></defs>'
            + "".join(tick_lines)
            + "".join(tick_labels)
            + f'<polygon points="{area_points}" fill="url(#lfArea{width})"/>'
            + f'<polyline points="{" ".join(points)}" fill="none" stroke="#12688b" '
              'stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/>'
            + "".join(circles)
            + "".join(x_labels)
            + '<text x="18" y="112" transform="rotate(-90 18 112)" '
              'text-anchor="middle" class="chart-axis-title">Load factor</text>'
            + '</svg></div>'
        )
        return svg
    st.markdown(draw_chart(1000, False) + draw_chart(400, True), unsafe_allow_html=True)


def render_home() -> None:
    st.query_params["view"] = "fleet"
    st.rerun()


def render_fleet() -> None:
    existing_route = st.session_state.get("submitted_route")
    if not existing_route:
        st.markdown(
            '<section class="home-hero"><div class="home-copy">'
            '<h1>Aircraft Model Recommendation</h1>'
            '<p>Search a directional route and compare recent fleet use with the model forecast.</p>'
            '</div></section>',
            unsafe_allow_html=True,
        )
    else:
        route_origin = str(existing_route["origin"])
        route_destination = str(existing_route["destination"])
        st.markdown(
            '<div class="result-page-marker"></div><div class="page-heading"><div><h1>Aircraft Model Recommendation</h1>'
            f'<p>{html.escape(airport_name(route_origin))} ({route_origin}) → {html.escape(airport_name(route_destination))} ({route_destination})</p></div>'
            f'<span class="forecast-tag">Forecast · {html.escape(model.metadata["target_month"][:7])}</span></div>',
            unsafe_allow_html=True,
        )
    controls = st.expander("Change route", expanded=False) if existing_route else st.container(key="route_search")
    with controls:
        if not existing_route:
            st.markdown('<div class="route-search-marker"></div><h2>Choose a route to begin</h2>', unsafe_allow_html=True)
        with st.container(key="route_controls"):
            origin_col, destination_col, action_col = st.columns([1,1,.45], gap="medium", vertical_alignment="bottom")
            with origin_col:
                origin = render_airport_search("Origin airport", "origin")
            recent_codes = set(model.destination_options(origin, operated_only=True)["AIRPORT_CODE"].astype(str)) if origin else set()
            with destination_col:
                destination = render_airport_search("Destination airport", "destination", recent_codes)
            with action_col:
                submitted = st.button("Recommend", type="primary", use_container_width=True)
            render_recent_routes()
    if submitted:
        if not origin or not destination:
            st.warning("Select both an origin and a destination.")
        elif origin == destination:
            st.warning("Origin and destination must be different airports.")
        else:
            _remember_route(origin, destination)
            st.session_state["submitted_route"] = {"origin": origin, "destination": destination}
            st.rerun()
    route = st.session_state.get("submitted_route")
    if not route:
        return
    origin, destination = route["origin"], route["destination"]
    try:
        ranking = model.score(origin, destination)
        overview = model.route_overview(origin, destination)
        history = model.route_lf_history(origin, destination)
        operators = model.route_carrier_insights(origin, destination)
    except Exception as error:
        st.error(f"The route could not be evaluated: {error}")
        return
    operated, projected = route_rankings(ranking)
    with st.container(key="decision_grid"):
        map_col, operated_col, projected_col = st.columns(3, gap="small")
        with map_col:
            with st.container(border=True):
                st.markdown('<div class="panel-title"><strong class="map-title">Directional route</strong><span>Origin to destination</span></div>', unsafe_allow_html=True)
                st.pydeck_chart(route_deck(overview, origin, destination), use_container_width=True, height=291)
        with operated_col:
            st.markdown(ranking_html(operated, "Most operated on direct flights", "Last 12 observed months", "departures"), unsafe_allow_html=True)
        with projected_col:
            st.markdown(ranking_html(projected, "Projected LF by recent support", "Recent months, support tier, then LF", "lf"), unsafe_allow_html=True)
    st.markdown(kpi_html(overview), unsafe_allow_html=True)
    chart_col, operator_col = st.columns([1.35,.65], gap="small")
    with chart_col:
        with st.container(border=True):
            st.markdown('<div class="section-title"><h3>Trailing route load factor</h3><span>Latest 12 observed months</span></div>', unsafe_allow_html=True)
            render_history_chart(history)
    with operator_col:
        with st.container(border=True):
            st.markdown('<div class="direct-flight-marker"></div><div class="direct-flight-intro">Observed airlines</div>', unsafe_allow_html=True)
            st.markdown(operator_html(operators), unsafe_allow_html=True)


def render_plane() -> None:
    st.markdown('<div class="plane-page-marker"></div>', unsafe_allow_html=True)
    options = model.aircraft_options()
    labels = {f'{aircraft_family(row.AIRCRAFT_DESCRIPTION)} · {row.AIRCRAFT_DESCRIPTION} · BTS {int(row.AIRCRAFT_TYPE)}':int(row.AIRCRAFT_TYPE) for row in options.itertuples(index=False)}
    requested = st.query_params.get("aircraft")
    requested_type = int(requested) if requested and str(requested).isdigit() else None
    names = list(labels)
    default = next((i for i,name in enumerate(names) if labels[name] == requested_type), None)
    requested_name = names[default] if default is not None else None
    selector_key = "plane_profile_search"
    selector_query_key = "_plane_profile_query"
    query_signature = str(requested_type) if requested_type is not None else None
    if selector_key not in st.session_state or st.session_state.get(selector_query_key) != query_signature:
        st.session_state[selector_key] = requested_name
        st.session_state[selector_query_key] = query_signature

    def sync_plane_selection() -> None:
        selected_name = st.session_state.get(selector_key)
        if selected_name is None:
            st.query_params.pop("aircraft", None)
            st.session_state[selector_query_key] = None
            return
        selected_type = labels[selected_name]
        st.query_params["aircraft"] = str(selected_type)
        st.session_state[selector_query_key] = str(selected_type)

    def clear_plane_selection() -> None:
        st.session_state[selector_key] = None
        st.query_params.pop("aircraft", None)
        st.session_state[selector_query_key] = None

    with st.container(key="plane_controls"):
        st.subheader("Plane Insights")
        st.caption("Select an aircraft model to review its operating profile, capacity and principal operators.")
        search_column, clear_column = st.columns([12, 1], gap="small", vertical_alignment="center")
        with search_column:
            selection = st.selectbox(
                "Aircraft model",
                names,
                index=None,
                placeholder="Search an aircraft model",
                key=selector_key,
                on_change=sync_plane_selection,
                label_visibility="collapsed",
            )
        with clear_column:
            st.button(
                "×",
                key="plane_clear_model",
                help="Clear aircraft model",
                on_click=clear_plane_selection,
                disabled=selection is None,
                width="stretch",
            )
    if selection is None:
        st.info("Select an aircraft model to open its profile.")
        return
    aircraft_type = labels[selection]
    profile = model.aircraft_profile(aircraft_type)
    description, family = str(profile["AIRCRAFT_DESCRIPTION"]), aircraft_family(profile["AIRCRAFT_DESCRIPTION"])
    image_record = AIRCRAFT_IMAGE_RECORDS.get(int(aircraft_type))
    if image_record and bool(image_record.get("FILE_EXISTS")):
        image = image_data_uri(Path(str(image_record["LOCAL_PATH"])))
        source = str(image_record.get("SOURCE_PAGE", ""))
        if source.startswith("http"):
            credit = f'<div class="photo-credit">Image source: <a href="{html.escape(source)}" target="_blank" rel="noopener">Wikipedia / Wikimedia Commons</a></div>'
        else:
            credit = '<div class="photo-credit">FleetFit aircraft asset</div>'
        visual = f'<div class="aircraft-visual"><div class="aircraft-photo"><img src="{image}" alt="{html.escape(description)}"></div>{credit}</div>'
    else:
        visual = f'<div class="aircraft-visual"><div class="aircraft-photo"><div class="image-na">{html.escape(family.upper())}<br>PHOTO NOT AVAILABLE</div></div></div>'
    st.markdown('<section class="profile-hero"><div><div class="eyebrow">Aircraft model profile</div>' f'<h1>{html.escape(family)}</h1><p>{html.escape(description)} · BTS type {aircraft_type}</p></div>{visual}</section>', unsafe_allow_html=True)
    range_nm = float(profile["RANGE_NM"])
    range_miles = range_nm * 1.15078
    range_km = range_nm * 1.852
    facts = [("First observed",str(profile["FIRST_OBSERVED_MONTH"])[:7]),("Reference range",f'{range_miles:,.0f} mi / {range_km:,.0f} km'),("Median seats",f'{float(profile["SEATS_MEDIAN"]):,.0f}'),("Observed seat range",f'{float(profile["SEATS_P05"]):,.0f}–{float(profile["SEATS_P95"]):,.0f}')]
    st.markdown('<div class="profile-facts">' + "".join(f'<div class="profile-fact"><span>{label}</span><strong>{value}</strong></div>' for label,value in facts) + "</div>", unsafe_allow_html=True)
    operator_chips = "".join(f'<span class="operator-chip">{html.escape(str(operator))}</span>' for operator in profile["OPERATORS"])
    st.markdown('<div class="profile-operators"><div class="section-title"><h3>Principal observed operators</h3><span>Ranked by performed departures</span></div>' f'<div class="operator-chip-list">{operator_chips}</div></div>', unsafe_allow_html=True)


def render_about() -> None:
    st.markdown('<section class="about-panel"><div class="eyebrow" style="color:#16878a">About FleetFit</div><h1>Decision support for aircraft model selection</h1><p>FleetFit estimates one-month-ahead load factor at directional route, aircraft-model and month level. It separates observed fleet use from the evidence-supported model forecast.</p><p>The prototype does not certify runway performance, regulatory permission, crew availability or fleet ownership. Exploratory combinations remain visible but are clearly distinguished from standard recommendations.</p><p>Support labels describe the route-aircraft observations in the twelve months available at the forecast origin. Strong support requires 12 observed months and at least 12 departures; established support requires at least 6 months and 12 departures; emerging support requires at least 3 months and 3 departures. Other combinations are exploratory. These labels are not calibrated prediction intervals.</p><p>Projected candidates are ordered by observed recent months, support tier, predicted LF and recent departures. Lifetime history is additional context.</p><p>This frozen demonstration forecasts December 2025 from the November 2025 origin. All predictions use the exported analytical pipeline. The application does not retrain the model.</p></section>', unsafe_allow_html=True)


def render_footer() -> None:
    st.markdown(
        '<div class="fleet-footer">TFM FINAL 2 · Elias Ventura - UCM Master IA Data Science Big Data - 2026</div>',
        unsafe_allow_html=True,
    )


model = load_recommender(str(ARTIFACT_DIRECTORY))
airports = load_airports(str(ARTIFACT_DIRECTORY))
airport_lookup = airports.set_index("AIRPORT_CODE").to_dict("index")
view = current_view()
render_navigation(view)
if view == "home": render_home()
elif view == "fleet": render_fleet()
elif view == "plane": render_plane()
else: render_about()
render_footer()
