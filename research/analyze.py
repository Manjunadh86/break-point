from __future__ import annotations

import sqlite3
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any

from research.config import ELO_K, ELO_RETIREMENT_K, ELO_START
from research.names import fold, same_player

LEVEL_LABEL = {
    "G": "Grand Slam",
    "M": "Masters 1000",
    "A": "ATP Tour",
    "C": "Challenger",
    "S": "ITF / Satellite",
    "F": "Tour Finals",
    "D": "Davis Cup",
}


def parse_completion(score: str | None) -> str:
    text = (score or "").upper()
    compact = text.replace(" ", "")
    if "W/O" in text or compact in {"WO", "W/O"} or compact.endswith("WO") or "WALK" in text:
        return "Walkover"
    if "RET" in text:
        return "Retired"
    if "DEF" in text:
        return "Default"
    return "Completed"


def expected_score(elo_a: float, elo_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400.0))


def elo_win_prob(elo_a: float, elo_b: float) -> float:
    return expected_score(elo_a, elo_b)


def update_elo(winner: float, loser: float, k: float) -> tuple[float, float]:
    exp_w = expected_score(winner, loser)
    return winner + k * (1 - exp_w), loser + k * (0 - (1 - exp_w))


def implied_from_decimal(price: float | None) -> float | None:
    if price is None or price <= 1:
        return None
    return 1.0 / price


def devig_two_way(price_a: float | None, price_b: float | None) -> dict[str, float | None]:
    ia = implied_from_decimal(price_a)
    ib = implied_from_decimal(price_b)
    if ia is None or ib is None:
        return {"prob_a": None, "prob_b": None, "overround": None}
    total = ia + ib
    return {
        "prob_a": ia / total,
        "prob_b": ib / total,
        "overround": total - 1.0,
    }


def hold_pct(sv_gms: int | None, bp_saved: int | None, bp_faced: int | None) -> float | None:
    if not sv_gms:
        return None
    broken = max((bp_faced or 0) - (bp_saved or 0), 0)
    return 1.0 - (broken / sv_gms)


def first_serve_won(first_in: int | None, first_won: int | None) -> float | None:
    if not first_in:
        return None
    return (first_won or 0) / first_in


def rebuild_ratings(conn: sqlite3.Connection) -> None:
    """Walk matches in time order and stamp pre-match Elo onto each row."""
    conn.execute("DELETE FROM ratings")
    overall: dict[int, float] = defaultdict(lambda: ELO_START)
    surface: dict[tuple[int, str], float] = defaultdict(lambda: ELO_START)
    counts: dict[tuple[int, str], int] = defaultdict(int)

    rows = conn.execute(
        """
        SELECT id, winner_id, loser_id, surface, completion
        FROM matches
        ORDER BY tourney_date, match_num, id
        """
    ).fetchall()

    stamped = []
    for row in rows:
        surf = row["surface"] or "Unknown"
        w_id, l_id = row["winner_id"], row["loser_id"]
        w_elo, l_elo = overall[w_id], overall[l_id]
        w_s, l_s = surface[(w_id, surf)], surface[(l_id, surf)]
        stamped.append((w_elo, l_elo, w_s, l_s, row["id"]))
        k = ELO_RETIREMENT_K if row["completion"] in {"Retired", "Walkover", "Default"} else ELO_K
        overall[w_id], overall[l_id] = update_elo(w_elo, l_elo, k)
        surface[(w_id, surf)], surface[(l_id, surf)] = update_elo(w_s, l_s, k)
        counts[(w_id, "ALL")] += 1
        counts[(l_id, "ALL")] += 1
        counts[(w_id, surf)] += 1
        counts[(l_id, surf)] += 1

    conn.executemany(
        """
        UPDATE matches
        SET winner_elo=?, loser_elo=?, winner_surf_elo=?, loser_surf_elo=?
        WHERE id=?
        """,
        stamped,
    )

    conn.executemany(
        "INSERT OR REPLACE INTO ratings(player_id, surface, elo, matches) VALUES (?,?,?,?)",
        [(pid, "ALL", elo, counts[(pid, "ALL")]) for pid, elo in overall.items()],
    )
    conn.executemany(
        "INSERT OR REPLACE INTO ratings(player_id, surface, elo, matches) VALUES (?,?,?,?)",
        [
            (pid, surf, elo, counts[(pid, surf)])
            for (pid, surf), elo in surface.items()
        ],
    )


def find_players(conn: sqlite3.Connection, query: str, limit: int = 12) -> list[dict]:
    q = f"%{query.strip()}%"
    rows = conn.execute(
        """
        SELECT player_id, name, ioc, hand,
               (SELECT elo FROM ratings r WHERE r.player_id = players.player_id AND r.surface='ALL') AS elo
        FROM players
        WHERE name LIKE ?
        ORDER BY name
        LIMIT ?
        """,
        (q, limit),
    ).fetchall()
    return [dict(r) for r in rows]


def resolve_player(conn: sqlite3.Connection, name: str) -> dict | None:
    exact = conn.execute(
        "SELECT player_id, name, ioc, hand, ht FROM players WHERE name = ? COLLATE NOCASE",
        (name.strip(),),
    ).fetchone()
    if exact:
        return dict(exact)
    hits = find_players(conn, name, limit=25)
    if not hits:
        return None
    for hit in hits:
        if same_player(hit["name"], name) or fold(hit["name"]) == fold(name):
            return {
                "player_id": hit["player_id"],
                "name": hit["name"],
                "ioc": hit["ioc"],
                "hand": hit["hand"],
                "ht": None,
            }
    return hits[0]


def _player_matches(conn: sqlite3.Connection, player_id: int, limit: int = 40) -> list[sqlite3.Row]:
    return conn.execute(
        """
        SELECT * FROM matches
        WHERE winner_id = ? OR loser_id = ?
        ORDER BY tourney_date DESC, match_num DESC
        LIMIT ?
        """,
        (player_id, player_id, limit),
    ).fetchall()


def _as_of(rows: list[sqlite3.Row], player_id: int) -> dict[str, Any]:
    wins = 0
    played = 0
    challenger_wins = 0
    challenger_played = 0
    retires = 0
    surfaces: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    hold, first = [], []
    last5 = []
    last_date = rows[0]["tourney_date"] if rows else None
    recent_14 = 0
    cutoff = (datetime.now() - timedelta(days=14)).strftime("%Y-%m-%d")

    for i, row in enumerate(rows):
        won = row["winner_id"] == player_id
        played += 1
        wins += int(won)
        if row["tourney_level"] == "C":
            challenger_played += 1
            challenger_wins += int(won)
        if (not won) and row["completion"] in {"Retired", "Walkover", "Default"}:
            retires += 1
        surf = row["surface"] or "?"
        surfaces[surf][1] += 1
        surfaces[surf][0] += int(won)
        if i < 5:
            last5.append(
                {
                    "date": row["tourney_date"],
                    "opponent": row["loser_name"] if won else row["winner_name"],
                    "won": won,
                    "score": row["score"],
                    "level": LEVEL_LABEL.get(row["tourney_level"], row["tourney_level"]),
                    "surface": row["surface"],
                    "tournament": row["tourney_name"],
                    "completion": row["completion"],
                }
            )
        if cutoff and row["tourney_date"] and row["tourney_date"] >= cutoff:
            recent_14 += 1
        ace_col = "w_" if won else "l_"
        hold.append(
            hold_pct(row[f"{ace_col}SvGms"], row[f"{ace_col}bpSaved"], row[f"{ace_col}bpFaced"])
        )
        first.append(first_serve_won(row[f"{ace_col}1stIn"], row[f"{ace_col}1stWon"]))

    def avg(values: list[float | None]) -> float | None:
        clean = [v for v in values if v is not None]
        return sum(clean) / len(clean) if clean else None

    return {
        "wins": wins,
        "played": played,
        "win_pct": wins / played if played else None,
        "challenger_win_pct": challenger_wins / challenger_played if challenger_played else None,
        "challenger_played": challenger_played,
        "retirements_against": retires,
        "surfaces": {
            s: {"wins": w, "played": p, "win_pct": w / p if p else None}
            for s, (w, p) in surfaces.items()
        },
        "last5": last5,
        "matches_last_14d": recent_14,
        "hold_pct": avg(hold[:20]),
        "first_serve_won": avg(first[:20]),
        "last_match_date": last_date,
    }


def _rating(conn: sqlite3.Connection, player_id: int, surface: str | None) -> dict:
    overall = conn.execute(
        "SELECT elo, matches FROM ratings WHERE player_id=? AND surface='ALL'",
        (player_id,),
    ).fetchone()
    surf = None
    if surface:
        surf = conn.execute(
            "SELECT elo, matches FROM ratings WHERE player_id=? AND surface=?",
            (player_id, surface),
        ).fetchone()
    return {
        "elo": overall["elo"] if overall else ELO_START,
        "matches": overall["matches"] if overall else 0,
        "surface_elo": surf["elo"] if surf else None,
        "surface_matches": surf["matches"] if surf else 0,
    }


def h2h(conn: sqlite3.Connection, a_id: int, b_id: int) -> dict:
    rows = conn.execute(
        """
        SELECT tourney_date, tourney_name, surface, score, tourney_level, winner_id, winner_name
        FROM matches
        WHERE (winner_id=? AND loser_id=?) OR (winner_id=? AND loser_id=?)
        ORDER BY tourney_date DESC
        """,
        (a_id, b_id, b_id, a_id),
    ).fetchall()
    a_wins = sum(1 for r in rows if r["winner_id"] == a_id)
    return {
        "played": len(rows),
        "a_wins": a_wins,
        "b_wins": len(rows) - a_wins,
        "recent": [
            {
                "date": r["tourney_date"],
                "tournament": r["tourney_name"],
                "surface": r["surface"],
                "score": r["score"],
                "level": LEVEL_LABEL.get(r["tourney_level"], r["tourney_level"]),
                "winner": r["winner_name"],
            }
            for r in rows[:8]
        ],
    }


def _odds_for_names(conn: sqlite3.Connection, name_a: str, name_b: str) -> list[dict]:
    rows = conn.execute(
        """
        SELECT * FROM closing_odds
        WHERE (winner_name LIKE ? OR loser_name LIKE ? OR winner_name LIKE ? OR loser_name LIKE ?)
        ORDER BY match_date DESC
        LIMIT 80
        """,
        (f"%{name_a.split()[-1]}%", f"%{name_a.split()[-1]}%", f"%{name_b.split()[-1]}%", f"%{name_b.split()[-1]}%"),
    ).fetchall()
    out = []
    for r in rows:
        if (same_player(r["winner_name"], name_a) and same_player(r["loser_name"], name_b)) or (
            same_player(r["winner_name"], name_b) and same_player(r["loser_name"], name_a)
        ):
            out.append(dict(r))
    return out[:12]


def matchup_report(
    conn: sqlite3.Connection,
    name_a: str,
    name_b: str,
    surface: str | None = None,
    line_a: float | None = None,
    line_b: float | None = None,
) -> dict:
    pa = resolve_player(conn, name_a)
    pb = resolve_player(conn, name_b)
    if not pa or not pb:
        return {
            "error": "player_not_found",
            "missing": [n for n, p in ((name_a, pa), (name_b, pb)) if p is None],
        }

    rows_a = _player_matches(conn, pa["player_id"])
    rows_b = _player_matches(conn, pb["player_id"])
    form_a = _as_of(rows_a, pa["player_id"])
    form_b = _as_of(rows_b, pb["player_id"])
    rate_a = _rating(conn, pa["player_id"], surface)
    rate_b = _rating(conn, pb["player_id"], surface)

    elo_a = rate_a["surface_elo"] if surface and rate_a["surface_elo"] else rate_a["elo"]
    elo_b = rate_b["surface_elo"] if surface and rate_b["surface_elo"] else rate_b["elo"]
    model_p_a = elo_win_prob(elo_a, elo_b)

    market = devig_two_way(line_a, line_b) if line_a and line_b else None
    disagreement = None
    if market and market["prob_a"] is not None:
        disagreement = model_p_a - market["prob_a"]

    hist_odds = _odds_for_names(conn, pa["name"], pb["name"])
    from research.live import iid_state_prob

    return {
        "a": {**pa, **rate_a, **form_a},
        "b": {**pb, **rate_b, **form_b},
        "surface": surface,
        "h2h": h2h(conn, pa["player_id"], pb["player_id"]),
        "model": {
            "p_a": model_p_a,
            "p_b": 1 - model_p_a,
            "elo_a": elo_a,
            "elo_b": elo_b,
            "note": (
                "Surface Elo when both players have matches on this surface; "
                "otherwise overall Elo. This is a research prior, not a tip."
            ),
        },
        "market": {
            "line_a": line_a,
            "line_b": line_b,
            **(market or {}),
            "disagreement_a": disagreement,
            "note": (
                "Disagreement = model probability minus de-vigged market. "
                "Gaps this small are usually noise plus juice."
            ),
        }
        if line_a or line_b
        else None,
        "historical_close": hist_odds,
        "inplay": {
            "after_a_wins_set1": iid_state_prob(model_p_a, 1, 0),
            "after_b_wins_set1": iid_state_prob(model_p_a, 0, 1),
            "at_one_all": iid_state_prob(model_p_a, 1, 1),
            "note": "Best-of-3 update if sets are independent. The In-play clock panel adds Challenger close-out tables.",
        },
        "flags": _flags(form_a, form_b, pa["name"], pb["name"]),
    }


def _flags(a: dict, b: dict, name_a: str, name_b: str) -> list[str]:
    flags = []
    if (a.get("matches_last_14d") or 0) >= 4:
        flags.append(f"{name_a} has {a['matches_last_14d']} matches in the last 14 days of sample — fatigue risk on Challenger slates.")
    if (b.get("matches_last_14d") or 0) >= 4:
        flags.append(f"{name_b} has {b['matches_last_14d']} matches in the last 14 days of sample.")
    if (a.get("retirements_against") or 0) >= 2:
        flags.append(f"{name_a} has {a['retirements_against']} RET/W/O in recent sample.")
    if (b.get("retirements_against") or 0) >= 2:
        flags.append(f"{name_b} has {b['retirements_against']} RET/W/O in recent sample.")
    if a.get("played", 0) < 8:
        flags.append(f"Thin sample for {name_a} in the loaded years.")
    if b.get("played", 0) < 8:
        flags.append(f"Thin sample for {name_b} in the loaded years.")
    return flags


def recent_challengers(conn: sqlite3.Connection, surface: str | None = None, limit: int = 40) -> list[dict]:
    sql = """
        SELECT tourney_date, tourney_name, surface, round, score, completion,
               winner_name, loser_name, winner_rank, loser_rank,
               winner_elo, loser_elo, winner_entry, loser_entry
        FROM matches
        WHERE tourney_level = 'C'
    """
    params: list[Any] = []
    if surface:
        sql += " AND surface = ?"
        params.append(surface)
    sql += " ORDER BY tourney_date DESC, match_num DESC LIMIT ?"
    params.append(limit)
    return [dict(r) for r in conn.execute(sql, params).fetchall()]


def challenger_context(conn: sqlite3.Connection) -> dict:
    row = conn.execute(
        """
        SELECT
            COUNT(*) AS n,
            AVG(CASE WHEN winner_rank IS NOT NULL AND loser_rank IS NOT NULL AND winner_rank > loser_rank THEN 1.0 ELSE 0.0 END) AS upset_by_rank,
            AVG(CASE WHEN completion='Retired' THEN 1.0 ELSE 0.0 END) AS ret_rate,
            AVG(CASE WHEN completion='Walkover' THEN 1.0 ELSE 0.0 END) AS wo_rate
        FROM matches
        WHERE tourney_level='C'
        """
    ).fetchone()
    atp = conn.execute(
        """
        SELECT
            COUNT(*) AS n,
            AVG(CASE WHEN winner_rank IS NOT NULL AND loser_rank IS NOT NULL AND winner_rank > loser_rank THEN 1.0 ELSE 0.0 END) AS upset_by_rank,
            AVG(CASE WHEN completion='Retired' THEN 1.0 ELSE 0.0 END) AS ret_rate
        FROM matches
        WHERE source='atp' AND tourney_level IN ('A','M','G','F')
        """
    ).fetchone()
    buckets = conn.execute(
        """
        SELECT bucket, COUNT(*) AS n,
               AVG(underdog_won) AS underdog_won
        FROM (
            SELECT
                CASE
                    WHEN ABS(COALESCE(winner_rank, 999) - COALESCE(loser_rank, 999)) < 20 THEN 'rank gap < 20'
                    WHEN ABS(COALESCE(winner_rank, 999) - COALESCE(loser_rank, 999)) < 50 THEN 'rank gap 20-49'
                    WHEN ABS(COALESCE(winner_rank, 999) - COALESCE(loser_rank, 999)) < 100 THEN 'rank gap 50-99'
                    ELSE 'rank gap 100+'
                END AS bucket,
                CASE WHEN winner_rank > loser_rank THEN 1.0 ELSE 0.0 END AS underdog_won
            FROM matches
            WHERE tourney_level='C' AND winner_rank IS NOT NULL AND loser_rank IS NOT NULL
        )
        GROUP BY bucket
        """
    ).fetchall()
    return {
        "challenger": dict(row) if row else {},
        "atp": dict(atp) if atp else {},
        "upset_buckets": [dict(b) for b in buckets],
        "read": (
            "Compare Challenger main draws to ATP tour matches only (qualifying is excluded). "
            "Higher retirement rates and closer rank-gap buckets are why Challenger numbers jump. "
            "A jump is a new price, not proof the old one was wrong."
        ),
    }


def market_report(conn: sqlite3.Connection) -> dict:
    rows = conn.execute(
        """
        SELECT series, surface, avg_w, avg_l, ps_w, ps_l, b365_w, b365_l, max_w, max_l, comment
        FROM closing_odds
        WHERE avg_w IS NOT NULL AND avg_l IS NOT NULL
        """
    ).fetchall()
    if not rows:
        return {
            "n": 0,
            "note": "No tennis-data closing lines loaded yet. Run: python -m research ingest --odds",
        }

    def fav_won(avg_w: float, avg_l: float) -> bool:
        return avg_w < avg_l

    samples = []
    for r in rows:
        if r["comment"] and "retired" in r["comment"].lower():
            continue
        dw = devig_two_way(r["avg_w"], r["avg_l"])
        if dw["prob_a"] is None:
            continue
        fav = fav_won(r["avg_w"], r["avg_l"])
        fav_prob = dw["prob_a"] if fav else dw["prob_b"]
        samples.append(
            {
                "fav_won": fav,
                "fav_prob": fav_prob,
                "overround": dw["overround"],
                "ps_gap": (
                    abs((r["ps_w"] or 0) - (r["avg_w"] or 0))
                    if r["ps_w"]
                    else None
                ),
                "series": r["series"],
                "surface": r["surface"],
            }
        )

    def calib(lo: float, hi: float) -> dict:
        chunk = [s for s in samples if lo <= (s["fav_prob"] or 0) < hi]
        if not chunk:
            return {"n": 0, "fav_win_rate": None, "avg_implied": None}
        return {
            "n": len(chunk),
            "fav_win_rate": sum(s["fav_won"] for s in chunk) / len(chunk),
            "avg_implied": sum(s["fav_prob"] for s in chunk) / len(chunk),
        }

    bands = [
        ("50-60%", 0.50, 0.60),
        ("60-70%", 0.60, 0.70),
        ("70-80%", 0.70, 0.80),
        ("80-90%", 0.80, 0.90),
        ("90%+", 0.90, 1.01),
    ]
    return {
        "n": len(samples),
        "avg_overround": sum(s["overround"] or 0 for s in samples) / len(samples),
        "favorite_win_rate": sum(s["fav_won"] for s in samples) / len(samples),
        "calibration": [{"band": label, **calib(lo, hi)} for label, lo, hi in bands],
        "note": (
            "These are ATP closing lines from tennis-data.co.uk (not Challenger, not Stake). "
            "If favorites win about as often as the de-vigged close implies, the close is efficient. "
            "Beating that after juice is the hard part."
        ),
    }


def snapshot_curves(conn: sqlite3.Connection, limit_events: int = 20) -> list[dict]:
    events = conn.execute(
        """
        SELECT event_id, player_a, player_b, tournament, MAX(snapped_at) AS last_seen
        FROM odds_snapshots
        GROUP BY event_id
        ORDER BY last_seen DESC
        LIMIT ?
        """,
        (limit_events,),
    ).fetchall()
    curves = []
    for ev in events:
        points = conn.execute(
            """
            SELECT snapped_at, bookmaker, player_name, price, implied
            FROM odds_snapshots
            WHERE event_id=?
            ORDER BY snapped_at
            """,
            (ev["event_id"],),
        ).fetchall()
        by_time: dict[str, dict] = {}
        for p in points:
            bucket = by_time.setdefault(
                p["snapped_at"],
                {"snapped_at": p["snapped_at"], "books": []},
            )
            bucket["books"].append(
                {
                    "bookmaker": p["bookmaker"],
                    "player": p["player_name"],
                    "price": p["price"],
                    "implied": p["implied"],
                }
            )
        curves.append(
            {
                "event_id": ev["event_id"],
                "player_a": ev["player_a"],
                "player_b": ev["player_b"],
                "tournament": ev["tournament"],
                "points": list(by_time.values()),
            }
        )
    return curves


def stats_overview(conn: sqlite3.Connection) -> dict:
    matches = conn.execute("SELECT COUNT(*) AS n FROM matches").fetchone()["n"]
    chall = conn.execute("SELECT COUNT(*) AS n FROM matches WHERE tourney_level='C'").fetchone()["n"]
    players = conn.execute("SELECT COUNT(*) AS n FROM players").fetchone()["n"]
    odds = conn.execute("SELECT COUNT(*) AS n FROM closing_odds").fetchone()["n"]
    snaps = conn.execute("SELECT COUNT(*) AS n FROM odds_snapshots").fetchone()["n"]
    last = conn.execute("SELECT MAX(tourney_date) AS d FROM matches").fetchone()["d"]
    ingested = conn.execute("SELECT value FROM meta WHERE key='last_ingest'").fetchone()
    return {
        "matches": matches,
        "challenger_matches": chall,
        "players": players,
        "closing_odds": odds,
        "snapshots": snaps,
        "latest_match_date": last,
        "last_ingest": ingested["value"] if ingested else None,
    }


def list_watchlist(conn: sqlite3.Connection) -> list[dict]:
    return [
        dict(r)
        for r in conn.execute("SELECT * FROM watchlist ORDER BY id DESC LIMIT 100").fetchall()
    ]


def add_watch(conn: sqlite3.Connection, payload: dict) -> dict:
    conn.execute(
        """
        INSERT INTO watchlist (created_at, sport, player_a, player_b, surface, tournament, start_time, notes, line_a, line_b)
        VALUES (datetime('now'), ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            payload.get("sport") or "challenger",
            payload["player_a"].strip(),
            payload["player_b"].strip(),
            payload.get("surface") or None,
            payload.get("tournament") or None,
            payload.get("start_time") or None,
            payload.get("notes") or None,
            payload.get("line_a"),
            payload.get("line_b"),
        ),
    )
    return {"ok": True}


def delete_watch(conn: sqlite3.Connection, watch_id: int) -> None:
    conn.execute("DELETE FROM watchlist WHERE id=?", (watch_id,))
