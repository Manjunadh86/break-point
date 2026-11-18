"""Retirement hazard overlay.

Retirements are ~3% of Challenger matches but a large share of "the 1.0X favourite
lost" tickets. They are modelled separately from the point engine:

  hazard per game = base(sex, level, surface) x player multiplier x MTO spike

and integrated over the expected remaining games to give P(player i retires before
the match ends). `settle()` then converts the on-court win probability into the
probability of being *paid* under the book's retirement rule.
"""

from __future__ import annotations

import json
import math
import sqlite3
from datetime import date as _date, timedelta

from research.config import DATA_DIR
from research.live import parse_sets

MODEL_DIR = DATA_DIR / "models"
RATES_PATH = MODEL_DIR / "hazard_rates.json"

MEN_SOURCES = ("atp", "qual_chall")

# Medical-timeout spike: multiplier on the per-game hazard, decaying with games since the MTO.
# Prior values (not fitted: public pbp data has no MTO events). Cutmore & Knottenbelt model
# retirement as injury arrivals with exponentially distributed severity and per-point damping.
MTO_SPIKE = 8.0
MTO_DECAY_GAMES = 3.0


def level_bucket(level: str | None, group: str) -> str:
    lv = (level or "").upper()
    if group == "men":
        if lv == "C":
            return "CH"
        if lv in {"G", "M", "A", "F", "D", "O"}:
            return "ATP"
        return "ITF"
    if lv.isdigit() or (lv.startswith("W") and lv[1:].isdigit()):
        return "ITF"
    return "WTA"


def _games(score: str | None) -> int:
    info = parse_sets(score)
    if not info:
        return 0
    g = sum(a + b for a, b in info["completed"])
    if info.get("unfinished"):
        g += sum(info["unfinished"])
    return g


def build_rates(conn: sqlite3.Connection) -> dict:
    """Per-match and per-game retirement rates by (group, level bucket, surface)."""
    has_tiers = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='event_tiers'").fetchone()
    rows = conn.execute(
        f"""
        SELECT m.source, m.tourney_level, m.surface, m.completion, m.score,
               {"et.segment" if has_tiers else "NULL"} AS segment
        FROM matches m {"LEFT JOIN event_tiers et ON et.tourney_id = m.tourney_id" if has_tiers else ""}
        WHERE m.completion IN ('Completed', 'Retired') AND m.score IS NOT NULL
        """
    ).fetchall()
    agg: dict[str, dict] = {}
    for r in rows:
        group = "men" if r["source"] in MEN_SOURCES else "women"
        bucket = level_bucket(r["tourney_level"], group)
        surf = (r["surface"] or "Unknown").title()
        keys = [f"{group}|{bucket}|{surf}", f"{group}|{bucket}|*", f"{group}|*|*"]
        if r["segment"]:
            keys = [f"{group}|{r['segment']}|{surf}", f"{group}|{r['segment']}|*"] + keys
        for key in keys:
            a = agg.setdefault(key, {"matches": 0, "retired": 0, "games": 0})
            a["matches"] += 1
            a["games"] += _games(r["score"])
            a["retired"] += 1 if r["completion"] == "Retired" else 0
    out = {}
    for key, a in agg.items():
        out[key] = {
            "matches": a["matches"],
            "retired": a["retired"],
            "games": a["games"],
            "per_match": round(a["retired"] / a["matches"], 5) if a["matches"] else None,
            "per_game": round(a["retired"] / a["games"], 6) if a["games"] else None,
        }
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RATES_PATH.write_text(json.dumps(out, indent=2))
    return out


def load_rates() -> dict:
    if RATES_PATH.exists():
        return json.loads(RATES_PATH.read_text())
    return {}


def base_hazard_per_game(rates: dict, group: str, bucket: str, surface: str | None) -> float:
    """Retirements per game played; falls back up the hierarchy. Default ~1.5 per 1000 games."""
    surf = (surface or "Unknown").title()
    tour_bucket = {"CH_BIG": "CH", "CH_MID": "CH", "CH_SMALL": "CH", "ITF_BIG": "ITF", "ITF_SMALL": "ITF"}.get(bucket, bucket)
    for key in (f"{group}|{bucket}|{surf}", f"{group}|{bucket}|*",
                f"{group}|{tour_bucket}|{surf}", f"{group}|{tour_bucket}|*", f"{group}|*|*"):
        rec = rates.get(key)
        if rec and rec.get("per_game") and rec["matches"] >= 500:
            return rec["per_game"]
    return 0.0015


def player_multiplier(conn: sqlite3.Connection, player_id: int | None, as_of: str | None = None) -> dict:
    """Retirements and walkovers by this player in the previous 12 months, as a hazard multiplier."""
    if not player_id:
        return {"multiplier": 1.0, "retirements_12m": 0, "matches_12m": 0}
    d0 = _date.fromisoformat(as_of[:10]) if as_of else _date.today()
    since = (d0 - timedelta(days=365)).isoformat()
    row = conn.execute(
        """
        SELECT COUNT(*) AS n,
               SUM(CASE WHEN completion IN ('Retired','Walkover','Default') AND loser_id = ? THEN 1 ELSE 0 END) AS ret
        FROM matches
        WHERE (winner_id = ? OR loser_id = ?) AND tourney_date >= ? AND tourney_date < ?
        """,
        (player_id, player_id, player_id, since, d0.isoformat()),
    ).fetchone()
    n, ret = int(row["n"] or 0), int(row["ret"] or 0)
    # 0 retirements -> 0.9, 1 -> 1.6, 2 -> 2.3 ..., capped at 3.5
    mult = min(0.9 + 0.7 * ret, 3.5) if n else 1.0
    return {"multiplier": mult, "retirements_12m": ret, "matches_12m": n}


def expected_remaining_games(sets: tuple[int, int], games: tuple[int, int], p_set: float = 0.5,
                             avg_set_games: float = 10.0) -> tuple[float, float]:
    """(games left in this set, games expected after this set) for a best-of-three."""
    g1, g2 = games
    left_this_set = max(1.0, avg_set_games - (g1 + g2))
    sets_played = sets[0] + sets[1]
    if sets_played >= 2:
        after = 0.0
    elif sets_played == 1:
        leader_p = p_set if sets[0] > sets[1] else 1.0 - p_set
        after = (1.0 - leader_p) * avg_set_games
    else:
        after = avg_set_games + 2.0 * p_set * (1.0 - p_set) * avg_set_games
    return left_this_set, after


def retire_probability(hazard_per_game: float, remaining_games: float, mto_games_ago: float | None = None) -> float:
    """P(retire before the end) given a constant per-game hazard, plus an optional MTO spike."""
    if remaining_games <= 0:
        return 0.0
    spike = 0.0
    if mto_games_ago is not None:
        spike = MTO_SPIKE * MTO_DECAY_GAMES * math.exp(-mto_games_ago / MTO_DECAY_GAMES) * (
            1.0 - math.exp(-remaining_games / MTO_DECAY_GAMES)
        )
    cum = hazard_per_game * (remaining_games + spike)
    return 1.0 - math.exp(-cum)


def settle(p_match: float, p_ret_1: float, p_ret_2: float, rule: str = "ball_served",
           share_before_set1: float = 0.0) -> dict:
    """Probability that a bet on player 1 is paid, under a retirement rule.

    rule:
      ball_served  retirements stand; the retiring player loses (most exchanges)
      one_set      void if the retirement happens before set 1 is completed
      match        void on any retirement
    share_before_set1: fraction of the remaining retirement hazard that falls inside set 1
    (0 once set 1 is complete).
    """
    done = max(0.0, 1.0 - p_ret_1 - p_ret_2)
    win_on_court = p_match * done
    if rule == "match":
        void = p_ret_1 + p_ret_2
        win = win_on_court
    elif rule == "one_set":
        void = (p_ret_1 + p_ret_2) * share_before_set1
        win = win_on_court + p_ret_2 * (1.0 - share_before_set1)
    else:
        void = 0.0
        win = win_on_court + p_ret_2
    lose = max(0.0, 1.0 - win - void)
    return {
        "rule": rule,
        "p_win": win,
        "p_void": void,
        "p_lose": lose,
        "p_win_given_action": win / (1.0 - void) if void < 1 else 0.0,
    }


def overlay(conn: sqlite3.Connection, group: str, bucket: str, surface: str | None,
            sets: tuple[int, int], games: tuple[int, int], p_set: float, p_match: float,
            p1_id: int | None, p2_id: int | None, mto_1_games_ago: float | None = None,
            mto_2_games_ago: float | None = None, rule: str = "ball_served", as_of: str | None = None) -> dict:
    rates = load_rates()
    if not rates:
        rates = build_rates(conn)
    base = base_hazard_per_game(rates, group, bucket, surface)
    m1 = player_multiplier(conn, p1_id, as_of)
    m2 = player_multiplier(conn, p2_id, as_of)
    left, after = expected_remaining_games(sets, games, p_set)
    remaining = left + after
    r1 = retire_probability(base * m1["multiplier"], remaining, mto_1_games_ago)
    r2 = retire_probability(base * m2["multiplier"], remaining, mto_2_games_ago)
    share1 = (left / remaining) if (sets == (0, 0) and remaining > 0) else 0.0
    s = settle(p_match, r1, r2, rule, share1)
    return {
        "base_hazard_per_game": base,
        "player_1": {**m1, "p_retire": r1},
        "player_2": {**m2, "p_retire": r2},
        "expected_remaining_games": remaining,
        "settlement": s,
        "p_settled": s["p_win_given_action"],
    }
