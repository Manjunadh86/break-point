from __future__ import annotations

import math
import re
import sqlite3
from collections import defaultdict
from functools import lru_cache
from typing import Any

from research.analyze import _player_matches, _rating, devig_two_way, elo_win_prob, resolve_player
from research.db import DB_PATH

SET_RE = re.compile(r"(\d+)\s*[-–]\s*(\d+)(?:\s*\(\s*\d+\s*\))?")

PHASES = [
    {
        "id": "open",
        "label": "First 10–15 minutes",
        "clock": "0–15 min",
        "state": "0-0",
        "why": "Still a pre-match bet. Walkovers are already known. Recorded Challenger retirements in this tape almost never land in the first 20 minutes.",
    },
    {
        "id": "set1",
        "label": "During set 1",
        "clock": "until the first set ends",
        "state": "0-0",
        "why": "Breaks move live prices. This desk has no point log, so this window is noise unless you are watching the hold pattern yourself.",
    },
    {
        "id": "after_s1",
        "label": "The set-1 print",
        "clock": "median minutes / set",
        "state": "1-0 / 0-1",
        "why": "Largest clean information update in a best-of-3. This is the moment the dataset can re-price.",
    },
    {
        "id": "set2",
        "label": "Set 2 with a lead",
        "clock": "after set 1 → straight-sets finish",
        "state": "1-0",
        "why": "Close-out versus break-back. A 7-6 lead is not a 6-0 lead.",
    },
    {
        "id": "decider",
        "label": "Deciding set",
        "clock": "after 1–1",
        "state": "1-1",
        "why": "Almost a new match. A real Elo gap still shows up, but it is small and juice is still in both sides.",
    },
]


def parse_sets(score: str | None) -> dict | None:
    if not score:
        return None
    raw = score.replace("–", "-")
    retired = "RET" in raw.upper() or "DEF" in raw.upper()
    pairs = [(int(a), int(b)) for a, b in SET_RE.findall(raw) if int(a) <= 20 and int(b) <= 20]
    if not pairs:
        return None

    def finished(w: int, l: int) -> bool:
        hi, lo = max(w, l), min(w, l)
        return (hi >= 6 and hi - lo >= 2) or (hi == 7 and lo in {5, 6})

    completed = []
    unfinished = None
    for w, l in pairs:
        if finished(w, l):
            completed.append((w, l))
        else:
            unfinished = (w, l)
    if not completed and not unfinished:
        return None
    first = completed[0] if completed else None
    winner_sets = sum(1 for w, l in completed if w > l)
    loser_sets = sum(1 for w, l in completed if l > w)
    return {
        "completed": completed,
        "unfinished": unfinished,
        "winner_sets": winner_sets,
        "loser_sets": loser_sets,
        "first": first,
        "retired": retired,
        "n_completed": len(completed),
        "straight": winner_sets >= 2 and loser_sets == 0 and not retired,
        "set1_winner_took_it": bool(first and first[0] > first[1]),
        "set1_tight": bool(first and abs(first[0] - first[1]) <= 2),
        "set1_tb": bool(first and sorted(first) == [6, 7]),
        "set1_bagel": bool(first and min(first) <= 1 and max(first) >= 6),
    }


def elo_gap_bucket(gap: float | None) -> str:
    if gap is None:
        return "unknown"
    ag = abs(gap)
    if ag < 40:
        return "even (<40 Elo)"
    if ag < 100:
        return "slight (40–99)"
    if ag < 180:
        return "clear (100–179)"
    return "heavy (180+)"


def set_win_prob_from_match(p_match: float) -> float:
    p = min(max(p_match, 0.02), 0.98)
    lo, hi = 0.02, 0.98
    for _ in range(40):
        mid = (lo + hi) / 2
        est = mid * mid * (3 - 2 * mid)
        if est < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _first_to(q: float, a_need: int, b_need: int) -> float:
    if a_need <= 0:
        return 1.0
    if b_need <= 0:
        return 0.0
    total = 0.0
    for k in range(b_need):
        total += math.comb(a_need - 1 + k, k) * (q ** a_need) * ((1 - q) ** k)
    return min(max(total, 0.0), 1.0)


def iid_state_prob(p_match: float, sets_a: int, sets_b: int, best_of: int = 3) -> float:
    q = set_win_prob_from_match(p_match)
    need = best_of // 2 + 1
    if sets_a >= need:
        return 1.0
    if sets_b >= need:
        return 0.0
    return _first_to(q, need - sets_a, need - sets_b)


def _rate(wins: int, n: int) -> dict:
    if n <= 0:
        return {"n": 0, "rate": None, "lo": None, "hi": None}
    p = wins / n
    se = 1.96 * math.sqrt(p * (1 - p) / n)
    return {"n": n, "rate": p, "lo": max(0.0, p - se), "hi": min(1.0, p + se)}


def _med(xs: list[float]) -> float | None:
    if not xs:
        return None
    xs = sorted(xs)
    return xs[len(xs) // 2]


def _favorite_won(row) -> bool | None:
    if row["winner_elo"] is not None and row["loser_elo"] is not None:
        return row["winner_elo"] >= row["loser_elo"]
    if row["winner_rank"] is not None and row["loser_rank"] is not None:
        return row["winner_rank"] <= row["loser_rank"]
    return None


def _iter_matches(conn: sqlite3.Connection, level: str, surface: str | None):
    sql = """
        SELECT score, minutes, surface, tourney_level, best_of, completion,
               winner_rank, loser_rank, winner_elo, loser_elo,
               winner_id, loser_id
        FROM matches
        WHERE completion IN ('Completed', 'Retired')
    """
    params: list[Any] = []
    if level == "C":
        sql += " AND tourney_level = 'C'"
    elif level == "ATP":
        sql += " AND source = 'atp' AND tourney_level IN ('A','M','G','F')"
    if surface:
        sql += " AND surface = ?"
        params.append(surface)
    return conn.execute(sql, params)


def _fingerprint(conn: sqlite3.Connection) -> tuple:
    row = conn.execute("SELECT COUNT(*), MAX(id) FROM matches").fetchone()
    return (str(DB_PATH), row[0], row[1])


def live_study(conn: sqlite3.Connection, level: str = "C", surface: str | None = None) -> dict:
    return _live_study_cached(_fingerprint(conn), level, surface or "")


@lru_cache(maxsize=16)
def _live_study_cached(fp: tuple, level: str, surface: str) -> dict:
    conn = sqlite3.connect(fp[0])
    conn.row_factory = sqlite3.Row
    try:
        return _compute_study(conn, level, surface or None)
    finally:
        conn.close()


def _compute_study(conn: sqlite3.Connection, level: str, surface: str | None) -> dict:
    close = {"all": [0, 0], "tight": [0, 0], "tb": [0, 0], "bagel": [0, 0]}
    fav_s1 = [0, 0]
    dog_s1 = [0, 0]
    fav_lost_s1 = [0, 0]
    decider_fav = [0, 0]
    elo_close: dict[str, list] = defaultdict(lambda: [0, 0])
    surf_close: dict[str, list] = defaultdict(lambda: [0, 0])
    minutes_2set: list[int] = []
    minutes_3set: list[int] = []
    per_set: list[float] = []
    ret_clock: dict[str, int] = defaultdict(int)
    ret_n = parsed = completed = straight = deciders = 0

    for row in _iter_matches(conn, level, surface):
        info = parse_sets(row["score"])
        if not info or not info["first"]:
            continue
        parsed += 1
        fav_won = _favorite_won(row)

        if row["completion"] == "Retired":
            ret_n += 1
            m = row["minutes"]
            if m is None:
                ret_clock["unknown"] += 1
            elif m < 20:
                ret_clock["0–19 min"] += 1
            elif m < 40:
                ret_clock["20–39 min"] += 1
            elif m < 70:
                ret_clock["40–69 min"] += 1
            else:
                ret_clock["70+ min"] += 1
            continue

        completed += 1
        if info["straight"]:
            straight += 1
        if info["winner_sets"] >= 2 and info["loser_sets"] == 1:
            deciders += 1
            if fav_won is not None:
                decider_fav[1] += 1
                decider_fav[0] += int(fav_won)

        if row["minutes"] and info["n_completed"] == 2:
            minutes_2set.append(row["minutes"])
        if row["minutes"] and info["n_completed"] >= 3:
            minutes_3set.append(row["minutes"])
        if row["minutes"] and info["n_completed"]:
            per_set.append(row["minutes"] / info["n_completed"])

        s1_to_match_winner = info["set1_winner_took_it"]
        close["all"][1] += 1
        close["all"][0] += int(s1_to_match_winner)
        if info["set1_tight"]:
            close["tight"][1] += 1
            close["tight"][0] += int(s1_to_match_winner)
        if info["set1_tb"]:
            close["tb"][1] += 1
            close["tb"][0] += int(s1_to_match_winner)
        if info["set1_bagel"]:
            close["bagel"][1] += 1
            close["bagel"][0] += int(s1_to_match_winner)

        if fav_won is True and s1_to_match_winner:
            # Favorite took set 1 and closed.
            fav_s1[0] += 1
            fav_s1[1] += 1
        elif fav_won is False and s1_to_match_winner:
            # Underdog took set 1 and closed — favorite dropped set 1 and lost.
            dog_s1[0] += 1
            dog_s1[1] += 1
            fav_lost_s1[1] += 1
        elif fav_won is True and not s1_to_match_winner:
            # Favorite dropped set 1, then came back.
            fav_lost_s1[0] += 1
            fav_lost_s1[1] += 1
            dog_s1[1] += 1
        elif fav_won is False and not s1_to_match_winner:
            # Favorite took set 1, then lost.
            fav_s1[1] += 1

        if row["winner_elo"] is not None and row["loser_elo"] is not None:
            b = elo_gap_bucket(abs(row["winner_elo"] - row["loser_elo"]))
            elo_close[b][1] += 1
            elo_close[b][0] += int(s1_to_match_winner)
        surf = row["surface"] or "Unknown"
        surf_close[surf][1] += 1
        surf_close[surf][0] += int(s1_to_match_winner)

    set1_min, two, three = _med(per_set), _med(minutes_2set), _med(minutes_3set)
    phases = []
    for phase in PHASES:
        item = dict(phase)
        if phase["id"] == "after_s1" and set1_min:
            item["clock"] = f"~{int(round(set1_min))} min (median minutes / set)"
        if phase["id"] == "set2" and two:
            item["clock"] = f"until ~{int(round(two))} min if it stays 2–0"
        if phase["id"] == "decider" and two and three:
            item["clock"] = f"1–1 near ~{int(round(two))} min · done ~{int(round(three))} min"
        phases.append(item)

    come_n = close["all"][1] - close["all"][0]
    return {
        "level": "Challenger" if level == "C" else "ATP tour",
        "surface": surface or "all",
        "n_parsed": parsed,
        "n_completed": completed,
        "n_retired": ret_n,
        "clock": {
            "median_min_per_set": set1_min,
            "median_straight_sets_min": two,
            "median_three_set_min": three,
            "read": (
                f"A set here lasts about {int(round(set1_min)) if set1_min else '?'} minutes. "
                f"Straight sets finish near {int(round(two)) if two else '?'} min. "
                f"A third set pushes the finish to ~{int(round(three)) if three else '?'} min. "
                "The clock tells you which *state* you are in. It is not a minute that prints money."
            ),
        },
        "phases": phases,
        "set1_closeout": {
            "all": _rate(close["all"][0], close["all"][1]),
            "tight_first_set": _rate(close["tight"][0], close["tight"][1]),
            "tiebreak_first_set": _rate(close["tb"][0], close["tb"][1]),
            "bagel_or_6_1": _rate(close["bagel"][0], close["bagel"][1]),
        },
        "comeback_after_losing_set1": _rate(come_n, close["all"][1]),
        "straight_sets_share": _rate(straight, completed),
        "reaches_decider": _rate(deciders, completed),
        "decider_favorite_wins": _rate(decider_fav[0], decider_fav[1]),
        "after_set1": {
            "favorite_won_set1_and_match": _rate(fav_s1[0], fav_s1[1]),
            "underdog_won_set1_and_match": _rate(dog_s1[0], dog_s1[1]),
            "favorite_lost_set1_but_won_match": _rate(fav_lost_s1[0], fav_lost_s1[1]),
            "set1_winner_then_lost": _rate(come_n, close["all"][1]),
        },
        "closeout_by_elo_gap": {k: _rate(v[0], v[1]) for k, v in sorted(elo_close.items())},
        "closeout_by_surface": {k: _rate(v[0], v[1]) for k, v in sorted(surf_close.items())},
        "retirement_clock": {
            "n": ret_n,
            "buckets": dict(ret_clock),
            "read": (
                "In this Challenger tape, most retirements stop the match after 40 minutes — "
                "after a set has already been played. That is a late body-risk, not an opening-minute scam."
            ),
        },
        "how_to_use": [
            "The set-1 winner closes about four times out of five. Fading every set-1 winner is the memorable 1-in-6, not the market.",
            "A live plus-money on the set-1 loser has to be about 6.00 just to break even on that 18% base rate. 3.50–5.00 still loses.",
            "When the pre-match favorite drops set 1 they come back about 23% of the time, not half. Do not fade the set-1 leader just because they were the dog.",
            "A 7-6 first set closes out worse than 6-0/6-1. If the live short price treats them the same, the tight-set closer is the less certain side.",
            "There is no Challenger live-odds tape here. We can say when probability moved. We cannot say a clock is +EV.",
        ],
    }


def _player_inplay(conn: sqlite3.Connection, player_id: int) -> dict:
    close, come, dec = [0, 0], [0, 0], [0, 0]
    for row in _player_matches(conn, player_id, limit=250):
        info = parse_sets(row["score"])
        if not info or not info["first"] or row["completion"] != "Completed":
            continue
        won = row["winner_id"] == player_id
        took_s1 = info["set1_winner_took_it"] if won else not info["set1_winner_took_it"]
        if took_s1:
            close[1] += 1
            close[0] += int(won)
        else:
            come[1] += 1
            come[0] += int(won)
        if info["winner_sets"] >= 2 and info["loser_sets"] == 1:
            dec[1] += 1
            dec[0] += int(won)
    return {
        "closeout_after_set1": _rate(close[0], close[1]),
        "comeback_after_set1": _rate(come[0], come[1]),
        "deciding_set": _rate(dec[0], dec[1]),
    }


def _pct(stat: dict) -> str:
    if not stat or stat.get("rate") is None:
        return "n/a"
    return f"{stat['rate'] * 100:.0f}% (n={stat['n']})"


def live_quote(
    conn: sqlite3.Connection,
    name_a: str,
    name_b: str,
    sets_a: int = 0,
    sets_b: int = 0,
    surface: str | None = None,
    set1_games: str | None = None,
    line_a: float | None = None,
    line_b: float | None = None,
    level: str = "C",
) -> dict:
    pa = resolve_player(conn, name_a)
    pb = resolve_player(conn, name_b)
    if not pa or not pb:
        return {
            "error": "player_not_found",
            "missing": [n for n, p in ((name_a, pa), (name_b, pb)) if p is None],
        }

    rate_a = _rating(conn, pa["player_id"], surface)
    rate_b = _rating(conn, pb["player_id"], surface)
    elo_a = rate_a["surface_elo"] if surface and rate_a["surface_elo"] else rate_a["elo"]
    elo_b = rate_b["surface_elo"] if surface and rate_b["surface_elo"] else rate_b["elo"]
    p0 = elo_win_prob(elo_a, elo_b)
    p_iid = iid_state_prob(p0, sets_a, sets_b)
    study = live_study(conn, level=level, surface=surface)
    in_a = _player_inplay(conn, pa["player_id"])
    in_b = _player_inplay(conn, pb["player_id"])

    tight = bagel = tb = False
    if set1_games:
        parsed = parse_sets(set1_games)
        if parsed and parsed["first"]:
            tight, bagel, tb = parsed["set1_tight"], parsed["set1_bagel"], parsed["set1_tb"]

    table = study["set1_closeout"]
    empirical = None
    label = "pre-match"
    if sets_a >= 2:
        empirical, p_iid, label = 1.0, 1.0, "match over"
    elif sets_b >= 2:
        empirical, p_iid, label = 0.0, 0.0, "match over"
    elif sets_a == 0 and sets_b == 0:
        empirical, label = p0, "pre-match / during set 1"
    elif sets_a == 1 and sets_b == 0:
        if tb:
            base, label = table["tiebreak_first_set"]["rate"], "A took set 1 in a tiebreak"
        elif bagel:
            base, label = table["bagel_or_6_1"]["rate"], "A took set 1 6-0/6-1"
        elif tight:
            base, label = table["tight_first_set"]["rate"], "A took a tight set 1"
        else:
            base, label = table["all"]["rate"], "A won set 1"
        empirical = None if base is None else 0.55 * base + 0.45 * p_iid
    elif sets_a == 0 and sets_b == 1:
        base = table["all"]["rate"]
        empirical = None if base is None else 0.55 * (1 - base) + 0.45 * p_iid
        label = "B won set 1 — A's comeback prior"
    elif sets_a == 1 and sets_b == 1:
        fav_dec = study["decider_favorite_wins"]["rate"]
        if fav_dec is None:
            empirical = p_iid
        elif p0 >= 0.5:
            empirical = 0.5 * fav_dec + 0.5 * p_iid
        else:
            empirical = 0.5 * (1 - fav_dec) + 0.5 * p_iid
        label = "1–1 deciding set"

    blended = p_iid if empirical is None else empirical
    market = devig_two_way(line_a, line_b) if line_a and line_b else None
    gap = None
    if market and market["prob_a"] is not None:
        gap = blended - market["prob_a"]

    clock = study["clock"]
    if sets_a + sets_b == 0:
        when = (
            f"Still the open. First-set information arrives around "
            f"{int(round(clock['median_min_per_set'] or 35))} min."
        )
    elif sets_a + sets_b == 1:
        when = (
            "This is the set-1 print — the only strong clock in the study. "
            f"Straight-sets matches are usually done by ~{int(round(clock['median_straight_sets_min'] or 70))} min."
        )
    else:
        when = (
            f"Deciders in this cut finish around {int(round(clock['median_three_set_min'] or 110))} min. "
            "The leftover favorite edge is small; juice still sits in both sides."
        )

    return {
        "a": {**pa, "inplay": in_a},
        "b": {**pb, "inplay": in_b},
        "pre_match": {"p_a": p0, "elo_a": elo_a, "elo_b": elo_b},
        "state": {"sets_a": sets_a, "sets_b": sets_b, "set1_games": set1_games, "label": label},
        "model": {
            "iid_sets": p_iid,
            "blended": blended,
            "note": (
                "IID sets invert pre-match Elo into a set-win rate, then update the BO3 score. "
                "The blend pulls toward the Challenger close-out / tight-set / decider table. "
                "A prior to hold next to the live number — not a tip."
            ),
        },
        "market": (
            {**(market or {}), "line_a": line_a, "line_b": line_b, "gap_a": gap}
            if line_a or line_b
            else None
        ),
        "when": when,
        "player_notes": [
            f"{pa['name']} close-out after set 1: {_pct(in_a['closeout_after_set1'])}",
            f"{pb['name']} close-out after set 1: {_pct(in_b['closeout_after_set1'])}",
            f"{pa['name']} deciding sets: {_pct(in_a['deciding_set'])}",
            f"{pb['name']} deciding sets: {_pct(in_b['deciding_set'])}",
        ],
        "tour": {
            "closeout": table,
            "decider_favorite": study["decider_favorite_wins"],
            "clock": clock,
        },
    }
