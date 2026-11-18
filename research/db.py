from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path

from research.config import DATA_DIR, DB_PATH

SCHEMA = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS matches (
    id INTEGER PRIMARY KEY,
    tourney_id TEXT NOT NULL,
    tourney_name TEXT,
    surface TEXT,
    draw_size INTEGER,
    tourney_level TEXT,
    tourney_date TEXT,
    match_num INTEGER,
    round TEXT,
    best_of INTEGER,
    score TEXT,
    minutes INTEGER,
    winner_id INTEGER,
    winner_name TEXT,
    winner_hand TEXT,
    winner_ht INTEGER,
    winner_ioc TEXT,
    winner_age REAL,
    winner_seed TEXT,
    winner_entry TEXT,
    winner_rank INTEGER,
    winner_rank_points INTEGER,
    loser_id INTEGER,
    loser_name TEXT,
    loser_hand TEXT,
    loser_ht INTEGER,
    loser_ioc TEXT,
    loser_age REAL,
    loser_seed TEXT,
    loser_entry TEXT,
    loser_rank INTEGER,
    loser_rank_points INTEGER,
    w_ace INTEGER, w_df INTEGER, w_svpt INTEGER, w_1stIn INTEGER,
    w_1stWon INTEGER, w_2ndWon INTEGER, w_SvGms INTEGER, w_bpSaved INTEGER, w_bpFaced INTEGER,
    l_ace INTEGER, l_df INTEGER, l_svpt INTEGER, l_1stIn INTEGER,
    l_1stWon INTEGER, l_2ndWon INTEGER, l_SvGms INTEGER, l_bpSaved INTEGER, l_bpFaced INTEGER,
    completion TEXT,
    source TEXT,
    winner_elo REAL,
    loser_elo REAL,
    winner_surf_elo REAL,
    loser_surf_elo REAL,
    UNIQUE (tourney_id, match_num, winner_id, loser_id)
);

CREATE INDEX IF NOT EXISTS idx_matches_date ON matches(tourney_date);
CREATE INDEX IF NOT EXISTS idx_matches_level ON matches(tourney_level);
CREATE INDEX IF NOT EXISTS idx_matches_winner ON matches(winner_id);
CREATE INDEX IF NOT EXISTS idx_matches_loser ON matches(loser_id);
CREATE INDEX IF NOT EXISTS idx_matches_wname ON matches(winner_name);
CREATE INDEX IF NOT EXISTS idx_matches_lname ON matches(loser_name);

CREATE TABLE IF NOT EXISTS players (
    player_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    hand TEXT,
    ioc TEXT,
    ht INTEGER
);

CREATE TABLE IF NOT EXISTS ratings (
    player_id INTEGER NOT NULL,
    surface TEXT NOT NULL,
    elo REAL NOT NULL,
    matches INTEGER NOT NULL,
    PRIMARY KEY (player_id, surface)
);

CREATE TABLE IF NOT EXISTS closing_odds (
    id INTEGER PRIMARY KEY,
    match_date TEXT,
    tournament TEXT,
    series TEXT,
    surface TEXT,
    round TEXT,
    winner_name TEXT,
    loser_name TEXT,
    winner_rank INTEGER,
    loser_rank INTEGER,
    comment TEXT,
    b365_w REAL, b365_l REAL,
    ps_w REAL, ps_l REAL,
    max_w REAL, max_l REAL,
    avg_w REAL, avg_l REAL,
    match_id INTEGER,
    UNIQUE (match_date, tournament, winner_name, loser_name, round)
);

CREATE INDEX IF NOT EXISTS idx_odds_date ON closing_odds(match_date);

CREATE TABLE IF NOT EXISTS odds_snapshots (
    id INTEGER PRIMARY KEY,
    snapped_at TEXT NOT NULL,
    sport_key TEXT,
    event_id TEXT,
    commence_time TEXT,
    tournament TEXT,
    player_a TEXT,
    player_b TEXT,
    bookmaker TEXT,
    market TEXT,
    player_name TEXT,
    price REAL,
    implied REAL
);

CREATE INDEX IF NOT EXISTS idx_snap_event ON odds_snapshots(event_id, snapped_at);

CREATE TABLE IF NOT EXISTS watchlist (
    id INTEGER PRIMARY KEY,
    created_at TEXT NOT NULL,
    sport TEXT NOT NULL,
    player_a TEXT NOT NULL,
    player_b TEXT NOT NULL,
    surface TEXT,
    tournament TEXT,
    start_time TEXT,
    notes TEXT,
    line_a REAL,
    line_b REAL
);

CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT
);

-- One row per match with a point-by-point string (Sackmann tennis_pointbypoint).
CREATE TABLE IF NOT EXISTS pbp_matches (
    id INTEGER PRIMARY KEY,
    date TEXT,
    tny_name TEXT,
    tour TEXT,              -- ATP, CH, WTA, ITF
    draw TEXT,              -- Main, Qual
    p1 TEXT NOT NULL,       -- served first
    p2 TEXT NOT NULL,
    winner INTEGER,         -- 1 or 2
    pbp TEXT NOT NULL,
    score TEXT,
    adf_flag INTEGER,
    best_of INTEGER,
    n_points INTEGER,
    valid INTEGER,
    parse_error TEXT,
    source_file TEXT,
    match_id INTEGER,       -- matches.id when linked by names + date
    p1_id INTEGER,
    p2_id INTEGER,
    p1_elo REAL,            -- pre-match Elo from the linked Sackmann row
    p2_elo REAL,
    p1_surf_elo REAL,
    p2_surf_elo REAL,
    surface TEXT,
    tourney_level TEXT,
    UNIQUE (date, p1, p2, pbp)
);

CREATE INDEX IF NOT EXISTS idx_pbp_tour_date ON pbp_matches(tour, date);
CREATE INDEX IF NOT EXISTS idx_pbp_match ON pbp_matches(match_id);

-- Every emitted in-play probability bundle, for paper trading and retraining.
CREATE TABLE IF NOT EXISTS paper_log (
    id INTEGER PRIMARY KEY,
    logged_at TEXT NOT NULL,
    event_id TEXT,
    p1 TEXT, p2 TEXT, tour TEXT, surface TEXT,
    state TEXT,             -- JSON score state
    p_prior REAL, p_updated REAL, p_model REAL, p_settled REAL,
    market_p REAL, price REAL, edge REAL, stake_frac REAL,
    note TEXT
);
"""


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "raw").mkdir(parents=True, exist_ok=True)


def connect(path: Path | None = None) -> sqlite3.Connection:
    ensure_dirs()
    db = path or DB_PATH
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(conn: sqlite3.Connection | None = None) -> sqlite3.Connection:
    own = conn is None
    conn = conn or connect()
    conn.executescript(SCHEMA)
    conn.commit()
    if own:
        return conn
    return conn


@contextmanager
def session():
    conn = init_db()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
