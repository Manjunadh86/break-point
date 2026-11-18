from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
DB_PATH = DATA_DIR / "research.db"

SACKMANN_MIRROR = "https://raw.githubusercontent.com/Aneeshers/tennis-sackmann-archive/main"
SACKMANN_BASE = f"{SACKMANN_MIRROR}/atp"
SACKMANN_WTA_BASE = f"{SACKMANN_MIRROR}/wta"
# Jeff Sackmann's tennis_pointbypoint repo is no longer published upstream; this fork
# carries the full 2011-Sept 2017 release (CC BY-NC-SA 4.0, research use only).
PBP_BASE = "https://raw.githubusercontent.com/vinhonrubia/tennis_pointbypoint/master"
PBP_FILES = (
    "atp_main_archive", "atp_main_current", "atp_qual_archive", "atp_qual_current",
    "ch_main_archive", "ch_main_current", "ch_qual_archive", "ch_qual_current",
    "wta_main_archive", "wta_main_current", "wta_qual_archive", "wta_qual_current",
    "itf_main_archive", "itf_main_current",
)
TENNIS_DATA_XLSX = "http://www.tennis-data.co.uk/{year}/{year}.xlsx"

DEFAULT_YEARS = (2022, 2023, 2024, 2025, 2026)
# Years needed so Elo exists for the point-by-point era (2011-2017) and the live desk.
PBP_ERA_YEARS = tuple(range(2009, 2027))

ODDS_API_KEY = os.getenv("ODDS_API_KEY", "").strip()
ODDS_SPORT_KEYS = [
    k.strip()
    for k in os.getenv(
        "ODDS_SPORT_KEYS",
        "tennis_atp_us_open,tennis_atp_french_open,tennis_atp_wimbledon,tennis_atp_australian_open",
    ).split(",")
    if k.strip()
]
ODDS_REGIONS = os.getenv("ODDS_REGIONS", "uk,eu,au")

ELO_START = 1500.0
ELO_K = 32.0
ELO_RETIREMENT_K = 16.0
