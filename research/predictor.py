from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path

import numpy as np

from research.analyze import _rating, elo_win_prob, resolve_player
from research.config import DATA_DIR
from research.live import parse_sets
from research.markov import match_from_state, next_branches

MODEL_DIR = DATA_DIR / "models"
MODEL_PATH = MODEL_DIR / "live_gbm.joblib"
METRICS_PATH = MODEL_DIR / "live_gbm_metrics.json"
TOUR_SERVE = 0.635
TOUR_RETURN = 0.365


def _num(value) -> float | None:
    if value is None:
        return None
    try:
        if isinstance(value, float) and math.isnan(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def serve_profile(conn: sqlite3.Connection, player_id: int, surface: str | None = None) -> dict:
    sql = """
        SELECT surface, winner_id,
               w_svpt, w_1stWon, w_2ndWon, w_SvGms, w_bpSaved, w_bpFaced,
               l_svpt, l_1stWon, l_2ndWon, l_SvGms, l_bpSaved, l_bpFaced
        FROM matches
        WHERE (winner_id = ? OR loser_id = ?) AND completion = 'Completed'
    """
    params: list = [player_id, player_id]
    if surface:
        sql += " AND surface = ?"
        params.append(surface)
    sql += " ORDER BY tourney_date DESC LIMIT 80"
    serve_pts = serve_won = ret_pts = ret_won = 0
    holds = games = 0
    n = 0
    for row in conn.execute(sql, params):
        n += 1
        won = row["winner_id"] == player_id
        svpt = row["w_svpt"] if won else row["l_svpt"]
        w1 = row["w_1stWon"] if won else row["l_1stWon"]
        w2 = row["w_2ndWon"] if won else row["l_2ndWon"]
        opp_svpt = row["l_svpt"] if won else row["w_svpt"]
        opp_w1 = row["l_1stWon"] if won else row["w_1stWon"]
        opp_w2 = row["l_2ndWon"] if won else row["w_2ndWon"]
        svg = row["w_SvGms"] if won else row["l_SvGms"]
        bps = row["w_bpSaved"] if won else row["l_bpSaved"]
        bpf = row["w_bpFaced"] if won else row["l_bpFaced"]
        if svpt and w1 is not None and w2 is not None:
            serve_pts += svpt
            serve_won += (w1 or 0) + (w2 or 0)
        if opp_svpt and opp_w1 is not None:
            ret_pts += opp_svpt
            ret_won += opp_svpt - ((opp_w1 or 0) + (opp_w2 or 0))
        if svg:
            games += svg
            holds += svg - max((bpf or 0) - (bps or 0), 0)
    raw_serve = serve_won / serve_pts if serve_pts >= 80 else None
    raw_return = ret_won / ret_pts if ret_pts >= 80 else None
    hold = holds / games if games >= 15 else None
    # shrink toward tour when thin
    def shrink(raw, tour, n_pts, k=400):
        if raw is None:
            return tour
        w = n_pts / (n_pts + k)
        return w * raw + (1 - w) * tour

    p_serve = shrink(raw_serve, TOUR_SERVE, serve_pts)
    p_return = shrink(raw_return, TOUR_RETURN, ret_pts)
    return {
        "n_matches": n,
        "serve_points": serve_pts,
        "raw_serve": raw_serve,
        "raw_return": raw_return,
        "hold": hold,
        "p_serve": min(max(p_serve, 0.50), 0.78),
        "p_return": min(max(p_return, 0.22), 0.50),
    }


def pair_point_probs(a: dict, b: dict) -> tuple[float, float]:
    """P(A wins a point on serve), P(A wins a point on return)."""
    p_serve = 0.5 * a["p_serve"] + 0.5 * (1 - b["p_return"])
    p_return = 0.5 * a["p_return"] + 0.5 * (1 - b["p_serve"])
    return min(max(p_serve, 0.50), 0.78), min(max(p_return, 0.22), 0.50)


FEATURE_NAMES = [
    "elo_1",
    "elo_2",
    "elo_gap",
    "rank_1",
    "rank_2",
    "p_serve",
    "p_return",
    "sets_1",
    "sets_2",
    "set1_margin",
    "set1_tb",
    "set1_bagel",
    "underdog",
    "clay",
    "hard",
    "grass",
    "markov",
]


def _feat_row(
    elo_1: float,
    elo_2: float,
    rank_1: float | None,
    rank_2: float | None,
    p_serve: float,
    p_return: float,
    sets_1: int,
    sets_2: int,
    set1_margin: float,
    set1_tb: float,
    set1_bagel: float,
    surface: str | None,
    markov: float,
) -> list[float]:
    surf = (surface or "").lower()
    return [
        elo_1,
        elo_2,
        elo_1 - elo_2,
        rank_1 if rank_1 is not None else 250,
        rank_2 if rank_2 is not None else 250,
        p_serve,
        p_return,
        float(sets_1),
        float(sets_2),
        set1_margin,
        set1_tb,
        set1_bagel,
        1.0 if elo_1 < elo_2 else 0.0,
        1.0 if surf == "clay" else 0.0,
        1.0 if surf == "hard" else 0.0,
        1.0 if surf == "grass" else 0.0,
        markov,
    ]


def _collect_rows(conn: sqlite3.Connection) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = conn.execute(
        """
        SELECT tourney_date, score, surface, best_of, completion,
               winner_id, loser_id, winner_rank, loser_rank,
               winner_elo, loser_elo
        FROM matches
        WHERE tourney_level = 'C' AND completion = 'Completed'
          AND winner_elo IS NOT NULL AND loser_elo IS NOT NULL
          AND best_of = 3
        """
    ).fetchall()
    X, y, dates = [], [], []
    for row in rows:
        info = parse_sets(row["score"])
        if not info or not info["first"] or info["n_completed"] < 1:
            continue
        first = info["first"]
        margin_w = first[0] - first[1]
        tb = 1.0 if sorted(first) == [6, 7] else 0.0
        bagel = 1.0 if min(first) <= 1 else 0.0
        # two orientations
        sides = [
            {
                "id": row["winner_id"],
                "elo": row["winner_elo"],
                "rank": row["winner_rank"],
                "sets": info["winner_sets"],
                "won": 1,
                "s1_margin": margin_w,
                "s1_won": 1 if info["set1_winner_took_it"] else 0,
            },
            {
                "id": row["loser_id"],
                "elo": row["loser_elo"],
                "rank": row["loser_rank"],
                "sets": info["loser_sets"],
                "won": 0,
                "s1_margin": -margin_w,
                "s1_won": 0 if info["set1_winner_took_it"] else 1,
            },
        ]
        # snapshot after set 1: sets are 1-0 or 0-1
        s1_a, s1_b = (1, 0) if info["set1_winner_took_it"] else (0, 1)
        snapshots = [("s1", s1_a, s1_b)]
        if info["winner_sets"] >= 2 and info["loser_sets"] == 1:
            snapshots.append(("11", 1, 1))
        for tag, sa, sb in snapshots:
            for i, j in ((0, 1), (1, 0)):
                a, b = sides[i], sides[j]
                sets_1 = sa if i == 0 else sb
                sets_2 = sb if i == 0 else sa
                if tag == "s1":
                    sets_1 = 1 if a["s1_won"] else 0
                    sets_2 = 1 - sets_1
                p_serve, p_return = 0.63, 0.37
                markov = match_from_state(
                    sets_1, sets_2, 0, 0, 0, 0, True, 0.635, 0.365
                )["match"]
                # average the other server too
                markov = 0.5 * markov + 0.5 * match_from_state(
                    sets_1, sets_2, 0, 0, 0, 0, False, 0.635, 0.365
                )["match"]
                X.append(
                    _feat_row(
                        a["elo"],
                        b["elo"],
                        _num(a["rank"]),
                        _num(b["rank"]),
                        p_serve,
                        p_return,
                        sets_1,
                        sets_2,
                        a["s1_margin"],
                        tb,
                        bagel,
                        row["surface"],
                        markov,
                    )
                )
                y.append(a["won"])
                dates.append(row["tourney_date"] or "2000-01-01")
    return np.array(X, dtype=float), np.array(y, dtype=int), np.array(dates)


def train_live_model(conn: sqlite3.Connection) -> dict:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
    import joblib

    X, y, dates = _collect_rows(conn)
    if len(y) < 2000:
        raise RuntimeError(f"Not enough labeled states: {len(y)}")
    cut = "2025-01-01"
    train = dates < cut
    test = dates >= cut
    if test.sum() < 200:
        # fallback random-ish last 20%
        n = int(len(y) * 0.8)
        train = np.zeros(len(y), dtype=bool)
        train[:n] = True
        test = ~train

    model = HistGradientBoostingClassifier(
        max_depth=4,
        learning_rate=0.08,
        max_iter=180,
        min_samples_leaf=80,
        l2_regularization=0.4,
        random_state=7,
    )
    model.fit(X[train], y[train])
    proba = model.predict_proba(X[test])[:, 1]
    metrics = {
        "n_train": int(train.sum()),
        "n_test": int(test.sum()),
        "log_loss": float(log_loss(y[test], proba)),
        "brier": float(brier_score_loss(y[test], proba)),
        "auc": float(roc_auc_score(y[test], proba)),
        "base_rate": float(y[test].mean()),
        "cut": cut,
        "features": FEATURE_NAMES,
        "note": (
            "Trained on Challenger set-boundary states (after set 1, and 1-1). "
            "Time-split: before 2025 train, 2025–2026 test. "
            "This is a calibration layer on top of the Markov engine, not a CNN."
        ),
    }
    # underdog-lost-set1 slice
    # sets_1 is feature index 7, underdog index 12
    xt, yt = X[test], y[test]
    mask = (xt[:, 7] == 0) & (xt[:, 8] == 1) & (xt[:, 12] == 1)
    if mask.sum() > 50:
        metrics["underdog_down_set1"] = {
            "n": int(mask.sum()),
            "actual": float(yt[mask].mean()),
            "pred": float(proba[mask].mean()),
            "brier": float(brier_score_loss(yt[mask], proba[mask])),
        }
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    return metrics


def _load_model():
    if not MODEL_PATH.exists():
        return None
    import joblib

    return joblib.load(MODEL_PATH)


def forecast(
    conn: sqlite3.Connection,
    p1_name: str,
    p2_name: str,
    sets_1: int = 0,
    sets_2: int = 0,
    games_1: int = 0,
    games_2: int = 0,
    points_1: str | int = 0,
    points_2: str | int = 0,
    server: str = "p1",
    surface: str | None = None,
    set1: str | None = None,
    in_tb: bool = False,
    line_1: float | None = None,
    line_2: float | None = None,
) -> dict:
    p1 = resolve_player(conn, p1_name)
    p2 = resolve_player(conn, p2_name)
    missing = [n for n, p in ((p1_name, p1), (p2_name, p2)) if p is None]
    if not p1:
        p1 = {"player_id": None, "name": p1_name, "ioc": None, "hand": None, "ht": None}
    if not p2:
        p2 = {"player_id": None, "name": p2_name, "ioc": None, "hand": None, "ht": None}

    def _elo(player):
        if not player.get("player_id"):
            return 1500.0
        rate = _rating(conn, player["player_id"], surface)
        return rate["surface_elo"] if surface and rate["surface_elo"] else rate["elo"]

    def _serve(player):
        if not player.get("player_id"):
            return {
                "n_matches": 0,
                "serve_points": 0,
                "raw_serve": None,
                "raw_return": None,
                "hold": None,
                "p_serve": TOUR_SERVE,
                "p_return": TOUR_RETURN,
            }
        return serve_profile(conn, player["player_id"], surface)

    elo_1, elo_2 = _elo(p1), _elo(p2)
    s1, s2 = _serve(p1), _serve(p2)
    p_serve, p_return = pair_point_probs(s1, s2)
    p1_serves = server.lower() in {"p1", "1", p1["name"].split()[-1].lower()}

    mk = match_from_state(
        sets_1,
        sets_2,
        games_1,
        games_2,
        points_1,
        points_2,
        p1_serves,
        p_serve,
        p_return,
        in_tb=in_tb,
    )
    branches = next_branches(sets_1, sets_2, games_1, games_2, p1_serves, p_serve, p_return)

    set1_margin = set1_tb = set1_bagel = 0.0
    if set1:
        parsed = parse_sets(set1)
        if parsed and parsed["first"]:
            a, b = parsed["first"]
            # set1 string is from P1 view if they type their games first
            set1_margin = float(a - b)
            set1_tb = 1.0 if sorted((a, b)) == [6, 7] else 0.0
            set1_bagel = 1.0 if min(a, b) <= 1 else 0.0
    elif sets_1 + sets_2 >= 1:
        set1_margin = 2.0 if sets_1 > sets_2 else -2.0

    markov_only = mk["match"]
    gbm_p = None
    model = _load_model()
    if model is not None:
        feat = np.array(
            [
                _feat_row(
                    elo_1,
                    elo_2,
                    None,
                    None,
                    p_serve,
                    p_return,
                    sets_1,
                    sets_2,
                    set1_margin,
                    set1_tb,
                    set1_bagel,
                    surface,
                    markov_only,
                )
            ]
        )
        gbm_p = float(model.predict_proba(feat)[0, 1])

    # Mid-set trust Markov more; at set boundary trust the GBM comeback layer.
    at_boundary = games_1 == 0 and games_2 == 0 and str(points_1) in {"0", "0", ""} and str(points_2) in {"0", ""}
    if gbm_p is None:
        blended = markov_only
        blend_note = "Markov only — run python -m research train to fit the Challenger state model."
    elif at_boundary and sets_1 + sets_2 > 0:
        blended = 0.45 * markov_only + 0.55 * gbm_p
        blend_note = "Set-boundary blend: GBM (Challenger comebacks) + Markov serve model."
    else:
        blended = 0.75 * markov_only + 0.25 * (gbm_p if gbm_p is not None else markov_only)
        blend_note = "In-game: Markov on serve/return owns the next point; GBM slightly calibrates the match."

    from research.analyze import devig_two_way

    market = devig_two_way(line_1, line_2) if line_1 and line_2 else None
    pre = elo_win_prob(elo_1, elo_2)
    metrics = json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else None

    underdog = elo_1 < elo_2
    story = _story(p1["name"], p2["name"], sets_1, sets_2, blended, pre, underdog, mk)

    return {
        "p1": {**p1, "elo": elo_1, "serve": s1, "underdog": underdog},
        "p2": {**p2, "elo": elo_2, "serve": s2, "underdog": not underdog},
        "state": {
            "sets": [sets_1, sets_2],
            "games": [games_1, games_2],
            "points": [points_1, points_2],
            "server": "p1" if p1_serves else "p2",
            "surface": surface,
            "set1": set1,
        },
        "pre_match": pre,
        "markov": mk,
        "gbm": gbm_p,
        "blended": blended,
        "branches": branches,
        "market": {**(market or {}), "line_1": line_1, "line_2": line_2} if line_1 or line_2 else None,
        "model": {
            "metrics": metrics,
            "blend": blend_note,
            "why_not_cnn": (
                "A CNN wants a grid (pixels, spectrograms). A tennis score is a small discrete state. "
                "The strongest model here is the scoring Markov chain plus a boosted tree on historical "
                "Challenger set-states — the same family used in serious in-play tennis models."
            ),
        },
        "story": story
        + (
            [f"No ATP/Challenger history for {', '.join(missing)} — tour-average serve is in use."]
            if missing
            else []
        ),
        "next": {
            "p1_wins_point": mk["point"],
            "p1_wins_game": mk["game"],
            "p1_wins_set": mk["set"],
            "p1_wins_match": blended,
        },
    }


def _story(n1, n2, s1, s2, p, pre, underdog, mk) -> list[str]:
    lines = []
    if s2 > s1 and underdog:
        lines.append(
            f"{n1} is the underdog and down {s1}–{s2}. Pre-match they were {pre*100:.0f}%. "
            f"From this score the engine has them at {p*100:.0f}% to still win the match."
        )
    elif s1 > s2 and underdog:
        lines.append(
            f"{n1} already stole a set. Live match prior is {p*100:.0f}% "
            f"(pre-match {pre*100:.0f}%)."
        )
    else:
        lines.append(f"From this score {n1} wins the match with prior {p*100:.0f}% (pre-match {pre*100:.0f}%).")
    lines.append(
        f"Next point {mk['point']*100:.0f}% · this game {mk['game']*100:.0f}% · this set {mk['set']*100:.0f}%."
    )
    lines.append(
        "Players do come back from bigger numbers — that is already in the Challenger table "
        "and the GBM. A long plus-money price can still be worse than this prior after juice."
    )
    return lines
