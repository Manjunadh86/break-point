"""Full-state in-play calibrator.

Every point of every point-by-point match becomes a training row: the Markov
engine's prior and dynamically-updated probabilities, the score state, live
serve/return performance, short-window momentum, carry-over fatigue, surface and
level. A gradient-boosted classifier per sex learns the corrections the iid chain
misses. Rows are built from both players' perspectives so the model is symmetric.

The same feature builder serves training (from pbp traces) and live forecasting
(from a scoreboard + running stats), so there is one definition of every feature.
"""

from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from datetime import date as _date

import numpy as np

from research.analyze import elo_win_prob
from research.config import DATA_DIR
from research.live import parse_sets
from research.markov import (
    fast_match_prob,
    importance,
    kappa_schedule,
    serve_return_from_match_prob,
    update_serve,
)
from research.pbp import C, RS
from research.prior import prior_prob
from research.replay import (
    PROGRESS_BINS,
    MatchTrace,
    TourConstants,
    _brier,
    _log_loss,
    build_trace,
    load_pbp,
    run_engines,
    slice_masks,
)

MODEL_DIR = DATA_DIR / "models"
# A "group" is one calibrator. The default units are the event segments from research.tiers
# (men: ATP / CH_BIG / CH_MID / CH_SMALL, women: WTA / ITF_BIG / ITF_SMALL); the umbrella
# "men" / "women" groups remain as a fallback when a segment model has not been trained.
GROUPS: dict[str, tuple[str, ...]] = {
    "ATP": ("ATP",), "CH_BIG": ("CH_BIG",), "CH_MID": ("CH_MID",), "CH_SMALL": ("CH_SMALL",),
    "WTA": ("WTA",), "ITF_BIG": ("ITF_BIG",), "ITF_SMALL": ("ITF_SMALL",),
    "men": ("ATP", "CH_BIG", "CH_MID", "CH_SMALL"),
    "women": ("WTA", "ITF_BIG", "ITF_SMALL"),
}
SEX_OF_GROUP = {g: ("men" if g in GROUPS["men"] or g == "men" else "women") for g in GROUPS}
SEGMENT_TOUR = {"ATP": "ATP", "CH_BIG": "CH", "CH_MID": "CH", "CH_SMALL": "CH",
                "WTA": "WTA", "ITF_BIG": "ITF", "ITF_SMALL": "ITF"}
ROUND_ORDER = {"Q1": 0, "Q2": 1, "Q3": 2, "R128": 3, "R64": 4, "R32": 5, "R16": 6, "QF": 7, "SF": 8, "F": 9, "RR": 5, "BR": 8}

FEATURE_NAMES = [
    # prior
    # absolute Elo levels are deliberately excluded: they drift between eras and the GBM
    # would learn the era instead of the matchup. Gaps and the Elo probability are kept.
    "elo_gap", "p_elo", "surf_elo_gap", "has_surf_elo",
    "prior_serve_a", "prior_serve_b",
    # score state (a's view)
    "sets_a", "sets_b", "games_a", "games_b", "pts_a", "pts_b", "a_serving", "in_tb",
    "points_played", "progress_est", "set_number", "prev_set_margin", "prev_set_tb",
    # engine outputs
    "p_prior", "p_dyn", "logit_prior", "logit_dyn", "dyn_minus_prior", "importance",
    "serve_dyn_a", "serve_dyn_b", "serve_shift_a", "serve_shift_b",
    # live serve / return performance
    "sv_pts_a", "sv_rate_a", "sv_pts_b", "sv_rate_b", "sv_excess_a", "sv_excess_b",
    "bp_faced_a", "bp_saved_a", "bp_faced_b", "bp_saved_b", "breaks_a", "breaks_b",
    "ace_a", "ace_b", "df_a", "df_b",
    # momentum
    "last4_a", "last8_a", "streak_a", "games_since_break", "last_break_by_a",
    # fatigue / carry-over
    "prev_games_a", "prev_games_b", "rest_days_a", "rest_days_b", "same_event_prev_a", "same_event_prev_b",
    "event_games_a", "event_games_b", "matches_14d_a", "matches_14d_b", "prev_minutes_a", "prev_minutes_b",
    # conditions / level
    "hard", "clay", "grass", "carpet", "level_top", "level_second", "qual",
]
F = {name: i for i, name in enumerate(FEATURE_NAMES)}


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


# --------------------------------------------------------------------------------------
# Fatigue index from the results table
# --------------------------------------------------------------------------------------


@dataclass
class PlayerLoad:
    prev_games: float = 0.0
    rest_days: float = 7.0
    same_event_prev: float = 0.0
    event_games: float = 0.0
    matches_14d: float = 0.0
    prev_minutes: float = 0.0


def _games_in_score(score: str | None) -> int:
    info = parse_sets(score)
    if not info:
        return 0
    total = sum(a + b for a, b in info["completed"])
    if info.get("unfinished"):
        total += sum(info["unfinished"])
    return total


def load_fatigue_index(conn: sqlite3.Connection, player_ids: set[int]) -> dict[int, list[tuple]]:
    """player_id -> sorted list of (tourney_date, round_order, tourney_id, games, minutes)."""
    if not player_ids:
        return {}
    index: dict[int, list[tuple]] = defaultdict(list)
    ids = list(player_ids)
    for start in range(0, len(ids), 800):
        chunk = ids[start:start + 800]
        q = ",".join("?" * len(chunk))
        rows = conn.execute(
            f"""
            SELECT tourney_date, round, tourney_id, score, minutes, winner_id, loser_id
            FROM matches
            WHERE (winner_id IN ({q}) OR loser_id IN ({q})) AND tourney_date IS NOT NULL
            """,
            chunk + chunk,
        ).fetchall()
        for r in rows:
            games = _games_in_score(r["score"])
            rec = (r["tourney_date"], ROUND_ORDER.get(r["round"] or "", 5), r["tourney_id"], games, r["minutes"] or 0)
            for pid in (r["winner_id"], r["loser_id"]):
                if pid in player_ids:
                    index[pid].append(rec)
    for pid in index:
        index[pid].sort()
    return index


def player_load(index: dict[int, list[tuple]], pid: int | None, match_date: str, tourney_id: str | None,
                round_order: int) -> PlayerLoad:
    """Workload before this match. Sackmann dates are tournament start dates, so
    within an event we order by round and treat the previous round as ~1 day ago."""
    load = PlayerLoad()
    if pid is None or pid not in index:
        return load
    recs = index[pid]
    try:
        d0 = _date.fromisoformat(match_date[:10])
    except ValueError:
        return load
    prev = None
    event_games = 0
    n14 = 0
    for rec in recs:
        rd, ro, tid, games, minutes = rec
        if tid == tourney_id:
            if ro < round_order:
                event_games += games
                n14 += 1
                prev = rec
            continue
        if rd >= match_date:
            continue
        try:
            gap = (d0 - _date.fromisoformat(rd[:10])).days
        except ValueError:
            continue
        if 0 <= gap <= 14:
            n14 += 1
        if prev is None or (prev[2] != tourney_id and rec >= prev):
            prev = rec
    load.event_games = float(event_games)
    load.matches_14d = float(n14)
    if prev is not None:
        load.prev_games = float(prev[3])
        load.prev_minutes = float(prev[4] or 0)
        if prev[2] == tourney_id:
            load.same_event_prev = 1.0
            load.rest_days = 1.0
        else:
            try:
                load.rest_days = float(min((d0 - _date.fromisoformat(prev[0][:10])).days, 30))
            except ValueError:
                load.rest_days = 7.0
    return load


# --------------------------------------------------------------------------------------
# Match context (per-match constants) and feature builder
# --------------------------------------------------------------------------------------


@dataclass
class MatchContext:
    elo: tuple[float, float]
    surf_elo: tuple[float | None, float | None]
    surface: str | None
    level: str | None
    qual: bool
    load: tuple[PlayerLoad, PlayerLoad]
    prior_serve: tuple[float, float]   # P(P1 wins point on own serve), P(P2 wins point on own serve)
    p_elo: float


def context_from_row(row: sqlite3.Row, consts: TourConstants, fatigue: dict[int, list[tuple]],
                     match_round: dict[int, tuple[str, int]]) -> MatchContext:
    p_elo = prior_prob(row["p1_elo"], row["p2_elo"], row["p1_surf_elo"], row["p2_surf_elo"],
                       row["segment"] if "segment" in row.keys() else None)
    s1, r1 = serve_return_from_match_prob(round(p_elo, 4), round(consts.serve_avg, 4))
    tid, ro = match_round.get(row["match_id"], (None, 5))
    return MatchContext(
        elo=(row["p1_elo"], row["p2_elo"]),
        surf_elo=(row["p1_surf_elo"], row["p2_surf_elo"]),
        surface=row["surface"],
        level=row["tourney_level"],
        qual=(row["draw"] or "").lower().startswith("q"),
        load=(player_load(fatigue, row["p1_id"], row["date"], tid, ro), player_load(fatigue, row["p2_id"], row["date"], tid, ro)),
        prior_serve=(s1, 1.0 - r1),
        p_elo=p_elo,
    )


def prev_set_info(states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per point: margin of the previous completed set from P1's view, and whether it was a TB."""
    n = len(states)
    margin = np.zeros(n, dtype=np.float32)
    tb = np.zeros(n, dtype=np.float32)
    last_margin = 0.0
    last_tb = 0.0
    for i in range(1, n):
        if states[i, C["set_idx"]] != states[i - 1, C["set_idx"]]:
            g1, g2 = int(states[i - 1, C["g1"]]), int(states[i - 1, C["g2"]])
            if states[i, C["set1"]] > states[i - 1, C["set1"]]:
                g1 += 1
            else:
                g2 += 1
            last_margin = float(g1 - g2)
            last_tb = 1.0 if (g1 == 7 and g2 == 6) or (g2 == 7 and g1 == 6) else 0.0
        margin[i] = last_margin
        tb[i] = last_tb
    return margin, tb


def _level_flags(level: str | None, tour_group: str) -> tuple[float, float]:
    """(level_top, level_second): men ATP vs Challenger; women WTA vs ITF."""
    lv = (level or "").upper()
    if tour_group == "men":
        top = 1.0 if lv in {"G", "M", "A", "F", "D", "O"} else 0.0
        second = 1.0 if lv == "C" else 0.0
    else:
        top = 1.0 if lv in {"G", "PM", "P", "I", "T1", "T2", "T3", "T4", "T5", "W", "O", "D", "F", "C"} else 0.0
        second = 1.0 if lv.isdigit() or (lv.startswith("W") and lv[1:].isdigit()) else 0.0
    return top, second


def feature_matrix(trace: MatchTrace, ctx: MatchContext, group: str, perspective: int = 1,
                   idx: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Rows for the points in idx (all if None) from the perspective of player `perspective`."""
    st_all = trace.states
    if idx is None:
        idx = np.arange(len(st_all))
    st = st_all[idx]
    rs = trace.stats[idx]
    a, b = (1, 2) if perspective == 1 else (2, 1)
    flip = perspective == 2

    p_prior = trace.probs["elo_markov"][idx].astype(np.float64)
    p_dyn = trace.probs["dyn"][idx].astype(np.float64)
    imp = trace.imp[idx].astype(np.float64) if trace.imp is not None else np.zeros(len(idx))
    serve_dyn = trace.serve_dyn[idx].astype(np.float64)
    margin_all, tb_all = prev_set_info(st_all)
    margin, tb = margin_all[idx], tb_all[idx]
    if flip:
        p_prior, p_dyn = 1 - p_prior, 1 - p_dyn
        margin = -margin

    def col(name_a: str, name_b: str):
        return (rs[:, RS[name_a]], rs[:, RS[name_b]]) if not flip else (rs[:, RS[name_b]], rs[:, RS[name_a]])

    sv_pts_a, sv_pts_b = col("sv_pts_1", "sv_pts_2")
    sv_won_a, sv_won_b = col("sv_won_1", "sv_won_2")
    bpf_a, bpf_b = col("bp_faced_1", "bp_faced_2")
    bps_a, bps_b = col("bp_saved_1", "bp_saved_2")
    brk_a, brk_b = col("breaks_1", "breaks_2")
    ace_a, ace_b = col("ace_1", "ace_2")
    df_a, df_b = col("df_1", "df_2")
    prior_a, prior_b = (ctx.prior_serve if not flip else ctx.prior_serve[::-1])
    sdyn_a = serve_dyn[:, 0] if not flip else serve_dyn[:, 1]
    sdyn_b = serve_dyn[:, 1] if not flip else serve_dyn[:, 0]
    sv_rate_a = np.where(sv_pts_a > 0, sv_won_a / np.maximum(sv_pts_a, 1), prior_a)
    sv_rate_b = np.where(sv_pts_b > 0, sv_won_b / np.maximum(sv_pts_b, 1), prior_b)

    points_played = st[:, C["idx"]].astype(np.float64)
    last4 = rs[:, RS["last4_1"]] if not flip else np.minimum(points_played, 4) - rs[:, RS["last4_1"]]
    last8 = rs[:, RS["last8_1"]] if not flip else np.minimum(points_played, 8) - rs[:, RS["last8_1"]]
    streak = rs[:, RS["streak"]] * (1 if not flip else -1)
    lbb = rs[:, RS["last_break_by"]]
    last_break_by_a = np.where(lbb == 0, 0.0, np.where(lbb == a, 1.0, -1.0))

    elo_a, elo_b = ctx.elo if not flip else ctx.elo[::-1]
    se_a, se_b = ctx.surf_elo if not flip else ctx.surf_elo[::-1]
    has_surf = 1.0 if (se_a is not None and se_b is not None) else 0.0
    surf_gap = (se_a - se_b) if has_surf else 0.0
    p_elo = ctx.p_elo if not flip else 1 - ctx.p_elo
    load_a, load_b = ctx.load if not flip else ctx.load[::-1]
    surf = (ctx.surface or "").lower()
    top, second = _level_flags(ctx.level, group)

    X = np.zeros((len(idx), len(FEATURE_NAMES)), dtype=np.float32)
    X[:, F["elo_gap"]] = elo_a - elo_b
    X[:, F["p_elo"]] = p_elo
    X[:, F["surf_elo_gap"]] = surf_gap
    X[:, F["has_surf_elo"]] = has_surf
    X[:, F["prior_serve_a"]] = prior_a
    X[:, F["prior_serve_b"]] = prior_b
    X[:, F["sets_a"]] = st[:, C[f"set{a}"]]
    X[:, F["sets_b"]] = st[:, C[f"set{b}"]]
    X[:, F["games_a"]] = st[:, C[f"g{a}"]]
    X[:, F["games_b"]] = st[:, C[f"g{b}"]]
    X[:, F["pts_a"]] = st[:, C[f"p{a}"]]
    X[:, F["pts_b"]] = st[:, C[f"p{b}"]]
    X[:, F["a_serving"]] = (st[:, C["server"]] == a)
    X[:, F["in_tb"]] = st[:, C["in_tb"]]
    X[:, F["points_played"]] = points_played
    X[:, F["progress_est"]] = trace.progress_est[idx]
    X[:, F["set_number"]] = st[:, C["set_idx"]] + 1
    X[:, F["prev_set_margin"]] = margin
    X[:, F["prev_set_tb"]] = tb
    X[:, F["p_prior"]] = p_prior
    X[:, F["p_dyn"]] = p_dyn
    X[:, F["logit_prior"]] = _logit(p_prior)
    X[:, F["logit_dyn"]] = _logit(p_dyn)
    X[:, F["dyn_minus_prior"]] = p_dyn - p_prior
    X[:, F["importance"]] = imp
    X[:, F["serve_dyn_a"]] = sdyn_a
    X[:, F["serve_dyn_b"]] = sdyn_b
    X[:, F["serve_shift_a"]] = sdyn_a - prior_a
    X[:, F["serve_shift_b"]] = sdyn_b - prior_b
    X[:, F["sv_pts_a"]] = sv_pts_a
    X[:, F["sv_rate_a"]] = sv_rate_a
    X[:, F["sv_pts_b"]] = sv_pts_b
    X[:, F["sv_rate_b"]] = sv_rate_b
    X[:, F["sv_excess_a"]] = (sv_rate_a - prior_a) * np.sqrt(sv_pts_a)
    X[:, F["sv_excess_b"]] = (sv_rate_b - prior_b) * np.sqrt(sv_pts_b)
    X[:, F["bp_faced_a"]] = bpf_a
    X[:, F["bp_saved_a"]] = bps_a
    X[:, F["bp_faced_b"]] = bpf_b
    X[:, F["bp_saved_b"]] = bps_b
    X[:, F["breaks_a"]] = brk_a
    X[:, F["breaks_b"]] = brk_b
    X[:, F["ace_a"]] = ace_a
    X[:, F["ace_b"]] = ace_b
    X[:, F["df_a"]] = df_a
    X[:, F["df_b"]] = df_b
    X[:, F["last4_a"]] = last4
    X[:, F["last8_a"]] = last8
    X[:, F["streak_a"]] = streak
    X[:, F["games_since_break"]] = rs[:, RS["games_since_break"]]
    X[:, F["last_break_by_a"]] = last_break_by_a
    X[:, F["prev_games_a"]] = load_a.prev_games
    X[:, F["prev_games_b"]] = load_b.prev_games
    X[:, F["rest_days_a"]] = load_a.rest_days
    X[:, F["rest_days_b"]] = load_b.rest_days
    X[:, F["same_event_prev_a"]] = load_a.same_event_prev
    X[:, F["same_event_prev_b"]] = load_b.same_event_prev
    X[:, F["event_games_a"]] = load_a.event_games
    X[:, F["event_games_b"]] = load_b.event_games
    X[:, F["matches_14d_a"]] = load_a.matches_14d
    X[:, F["matches_14d_b"]] = load_b.matches_14d
    X[:, F["prev_minutes_a"]] = load_a.prev_minutes
    X[:, F["prev_minutes_b"]] = load_b.prev_minutes
    X[:, F["hard"]] = surf == "hard"
    X[:, F["clay"]] = surf == "clay"
    X[:, F["grass"]] = surf == "grass"
    X[:, F["carpet"]] = surf == "carpet"
    X[:, F["level_top"]] = top
    X[:, F["level_second"]] = second
    X[:, F["qual"]] = 1.0 if ctx.qual else 0.0

    label = trace.label if not flip else 1 - trace.label
    y = np.full(len(idx), label, dtype=np.int8)
    return X, y


# --------------------------------------------------------------------------------------
# Dataset assembly and training
# --------------------------------------------------------------------------------------


def _load_constants(tour: str) -> TourConstants | None:
    path = MODEL_DIR / f"tour_constants_{tour}.json"
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    return TourConstants(tour=tour, serve_avg=d["serve_avg"], median_points=d["median_points"], kappa=tuple(d["kappa"]))


def _constants_for(conn: sqlite3.Connection, segment: str, train_until: str) -> TourConstants:
    """Serve average / median length / kappa for a segment; falls back to its tour, then recomputes."""
    consts = _load_constants(segment) or _load_constants(SEGMENT_TOUR.get(segment, segment))
    if consts is None:
        from research.replay import tour_constants

        rows = _load_segment_rows(conn, segment)
        consts = tour_constants(rows, segment, train_until)
    return consts


def _load_segment_rows(conn: sqlite3.Connection, segment: str, **kw) -> list[sqlite3.Row]:
    if segment in SEGMENT_TOUR:
        return load_pbp(conn, segments=(segment,), **kw)
    return load_pbp(conn, tours=(segment,), **kw)


def collect_dataset(
    conn: sqlite3.Connection,
    group: str = "men",
    train_until: str = "2016-01-01",
    stride: int = 3,
    include_qual: bool = True,
    limit_per_tour: int | None = None,
) -> dict:
    tours = GROUPS[group]
    X_tr, y_tr, X_te, y_te = [], [], [], []
    d_tr: list[np.ndarray] = []
    te_meta: list[dict] = []
    n_matches = {"train": 0, "test": 0}
    rng = np.random.default_rng(11)
    sex = SEX_OF_GROUP.get(group, "men")
    for tour in tours:
        consts = _constants_for(conn, tour, train_until)
        rows = _load_segment_rows(conn, tour, draw=None if include_qual else "Main", limit=limit_per_tour)
        if not rows:
            continue
        pids = {r["p1_id"] for r in rows} | {r["p2_id"] for r in rows}
        pids.discard(None)
        fatigue = load_fatigue_index(conn, pids)
        mids = [r["match_id"] for r in rows if r["match_id"] is not None]
        match_round: dict[int, tuple[str, int]] = {}
        for start in range(0, len(mids), 900):
            chunk = mids[start:start + 900]
            for mr in conn.execute(
                f"SELECT id, tourney_id, round FROM matches WHERE id IN ({','.join('?' * len(chunk))})", chunk
            ):
                match_round[mr["id"]] = (mr["tourney_id"], ROUND_ORDER.get(mr["round"] or "", 5))
        print(f"  {tour}: {len(rows)} matches, building features …", flush=True)
        for r in rows:
            trace = build_trace(r, consts)
            if trace is None:
                continue
            run_engines(trace, consts, with_importance=True)
            ctx = context_from_row(r, consts, fatigue, match_round)
            is_train = trace.date < train_until
            if is_train:
                offset = int(rng.integers(0, stride))
                idx = np.arange(offset, len(trace.states), stride)
                for persp in (1, 2):
                    X, y = feature_matrix(trace, ctx, sex, persp, idx)
                    X_tr.append(X)
                    y_tr.append(y)
                    d_tr.append(np.full(len(y), trace.date))
                n_matches["train"] += 1
            else:
                X, y = feature_matrix(trace, ctx, sex, 1, None)
                X_te.append(X)
                y_te.append(y)
                masks = slice_masks(trace)
                te_meta.append({"progress": trace.progress, "masks": masks, "tour": tour,
                                "p_dyn": trace.probs["dyn"], "p_prior": trace.probs["elo_markov"],
                                "p_elo": trace.probs["elo_static"]})
                n_matches["test"] += 1
    return {
        "X_train": np.concatenate(X_tr) if X_tr else np.zeros((0, len(FEATURE_NAMES)), np.float32),
        "y_train": np.concatenate(y_tr) if y_tr else np.zeros(0, np.int8),
        "date_train": np.concatenate(d_tr) if d_tr else np.zeros(0, dtype="<U10"),
        "X_test": np.concatenate(X_te) if X_te else np.zeros((0, len(FEATURE_NAMES)), np.float32),
        "y_test": np.concatenate(y_te) if y_te else np.zeros(0, np.int8),
        "test_meta": te_meta,
        "n_matches": n_matches,
    }


def _bucket_metrics(p: np.ndarray, y: np.ndarray) -> dict:
    return {"n": int(len(p)), "log_loss": round(_log_loss(p, y), 4), "brier": round(_brier(p, y), 4),
            "mean_pred": round(float(p.mean()), 4), "mean_actual": round(float(y.mean()), 4)}


def evaluate(proba: np.ndarray, data: dict) -> dict:
    """Compare the GBM to the engines it sits on, by progress bin and slice, on the test rows."""
    y = data["y_test"].astype(np.float64)
    progress = np.concatenate([m["progress"] for m in data["test_meta"]])
    p_dyn = np.concatenate([m["p_dyn"] for m in data["test_meta"]]).astype(np.float64)
    p_prior = np.concatenate([m["p_prior"] for m in data["test_meta"]]).astype(np.float64)
    p_elo = np.concatenate([m["p_elo"] for m in data["test_meta"]]).astype(np.float64)
    tours = np.concatenate([np.full(len(m["progress"]), m["tour"]) for m in data["test_meta"]])
    engines = {"elo_static": p_elo, "elo_markov": p_prior, "dyn": p_dyn, "gbm": proba.astype(np.float64)}
    out: dict = {"overall": {}, "progress": {}, "slices": {}, "by_tour": {}}
    for name, p in engines.items():
        out["overall"][name] = _bucket_metrics(p, y)
        for lo, hi in PROGRESS_BINS:
            m = (progress >= lo) & (progress < hi)
            if m.any():
                out["progress"].setdefault(f"{int(lo*100)}-{int(min(hi,1.0)*100)}", {})[name] = _bucket_metrics(p[m], y[m])
        for tour in np.unique(tours):
            m = tours == tour
            out["by_tour"].setdefault(str(tour), {})[name] = _bucket_metrics(p[m], y[m])
    slice_names = list(data["test_meta"][0]["masks"].keys()) if data["test_meta"] else []
    for sname in slice_names:
        m = np.concatenate([meta["masks"][sname] for meta in data["test_meta"]])
        if m.sum() < 200:
            continue
        for name, p in engines.items():
            out["slices"].setdefault(sname, {})[name] = _bucket_metrics(p[m], y[m])
    edges = np.linspace(0, 1, 11)
    cal = []
    pg = engines["gbm"]
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (pg >= lo) & ((pg < hi) if hi < 1 else (pg <= hi))
        if m.any():
            cal.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": int(m.sum()), "mean_pred": round(float(pg[m].mean()), 4),
                        "actual": round(float(y[m].mean()), 4)})
    out["calibration_gbm"] = cal
    return out


class OffsetBooster:
    """Calibrator that starts at the dynamic engine and learns only a correction.

    p = sigmoid(logit(p_dyn) + g(x)). g is fitted as a damped Newton step of the log loss around
    the engine: weighted least squares of the working response (y - p)/(p(1-p)) with weights
    p(1-p), using a gradient-boosted regressor. With g == 0 the model is exactly `dyn`, so a
    small or heavily shrunk ensemble cannot be much worse than the engine it sits on.
    """

    def __init__(self, params: dict, offset_col: int):
        self.params = dict(params)
        self.offset_col = offset_col
        self.reg = None
        self.n_iter_ = 0

    def _offset(self, X: np.ndarray) -> np.ndarray:
        return _logit(X[:, self.offset_col].astype(np.float64))

    @staticmethod
    def _working(z: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        p = np.clip(1.0 / (1.0 + np.exp(-z)), 0.02, 0.98)
        w = p * (1.0 - p)
        return (y - p) / w, w

    def fit(self, X: np.ndarray, y: np.ndarray, max_iter: int) -> "OffsetBooster":
        from sklearn.ensemble import HistGradientBoostingRegressor

        z = self._offset(X)
        r, w = self._working(z, y.astype(np.float64))
        self.reg = HistGradientBoostingRegressor(max_iter=max_iter, **self.params)
        self.reg.fit(X, r, sample_weight=w)
        self.n_iter_ = int(self.reg.n_iter_)
        return self

    def staged_log_loss(self, X: np.ndarray, y: np.ndarray) -> list[float]:
        z = self._offset(X)
        yy = y.astype(np.float64)
        out = []
        for g in self.reg.staged_predict(X):
            p = 1.0 / (1.0 + np.exp(-(z + g)))
            out.append(_log_loss(p, yy))
        return out

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        z = self._offset(np.asarray(X, dtype=np.float32))
        g = self.reg.predict(np.asarray(X, dtype=np.float32)) if self.reg is not None else 0.0
        p = 1.0 / (1.0 + np.exp(-(z + g)))
        return np.column_stack([1.0 - p, p])


def train_inplay_model(conn: sqlite3.Connection, group: str = "men", train_until: str = "2016-01-01",
                       stride: int = 3, limit_per_tour: int | None = None) -> dict:
    import joblib

    print(f"collecting {group} dataset (train < {train_until}) …", flush=True)
    data = collect_dataset(conn, group, train_until, stride, limit_per_tour=limit_per_tour)
    Xtr, ytr, Xte, yte = data["X_train"], data["y_train"], data["X_test"], data["y_test"]
    print(f"  train rows {len(ytr):,}  test rows {len(yte):,}  matches {data['n_matches']}", flush=True)
    if len(ytr) < 5000:
        raise RuntimeError(f"not enough training rows: {len(ytr)}")

    params = dict(
        learning_rate=0.05,
        max_depth=5,
        min_samples_leaf=1000,
        l2_regularization=5.0,
        max_bins=255,
        early_stopping=False,
        random_state=7,
    )

    # Chronological early stopping. A random row split would put points from the same match on
    # both sides and never stop; instead the last ~15% of training matches (by date) are held out,
    # the best iteration is picked on them, then the model is refit on all training rows.
    dates = data["date_train"]
    val_cut = str(np.sort(np.unique(dates))[int(0.85 * len(np.unique(dates)))])
    core, val = dates < val_cut, dates >= val_cut
    print(f"  validation from {val_cut}: {int(core.sum()):,} core rows, {int(val.sum()):,} val rows", flush=True)
    max_probe = 400
    probe = OffsetBooster(params, F["p_dyn"]).fit(Xtr[core], ytr[core], max_iter=max_probe)
    losses = probe.staged_log_loss(Xtr[val], ytr[val])
    best_iter = int(np.argmin(losses)) + 1
    yv = ytr[val].astype(np.float64)
    p_dyn_val = np.clip(Xtr[val][:, F["p_dyn"]].astype(np.float64), 1e-4, 1 - 1e-4)
    dyn_val = _log_loss(p_dyn_val, yv)
    print(f"  best iteration {best_iter}/{max_probe}: val log loss {losses[best_iter - 1]:.4f} "
          f"(dyn {dyn_val:.4f}, gain {dyn_val - losses[best_iter - 1]:+.4f})", flush=True)

    model = OffsetBooster(params, F["p_dyn"]).fit(Xtr, ytr, max_iter=max(best_iter, 10))
    proba = model.predict_proba(Xte)[:, 1] if len(yte) else np.zeros(0)
    metrics = evaluate(proba, data) if len(yte) else {}

    rng = np.random.default_rng(3)
    sample = rng.choice(len(Xte), size=min(40000, len(Xte)), replace=False) if len(Xte) else np.zeros(0, int)
    fi = {}
    if len(sample):
        base = model.predict_proba(Xte[sample])[:, 1]
        for name, j in F.items():
            Xs = Xte[sample].copy()
            Xs[:, j] = rng.permutation(Xs[:, j])
            fi[name] = round(float(np.mean(np.abs(model.predict_proba(Xs)[:, 1] - base))), 5)
        fi = dict(sorted(fi.items(), key=lambda kv: -kv[1]))

    result = {
        "group": group,
        "tours": GROUPS[group],
        "train_until": train_until,
        "n_train_rows": int(len(ytr)),
        "n_test_rows": int(len(yte)),
        "n_matches": data["n_matches"],
        "n_iter": int(model.n_iter_),
        "features": FEATURE_NAMES,
        "feature_sensitivity": fi,
        "metrics": metrics,
        "note": "Rows are every point (stride-sampled in training, both perspectives); chronological split by match date; "
                "monotone in the dynamic Markov probability.",
    }
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_DIR / f"inplay_{group}.joblib")
    (MODEL_DIR / f"inplay_{group}_metrics.json").write_text(json.dumps(result, indent=2, default=str))
    return result


def format_inplay_report(result: dict) -> str:
    m = result.get("metrics", {})
    engines = ("elo_static", "elo_markov", "dyn", "gbm")
    lines = [f"In-play model [{result['group']}] tours {result['tours']} — train rows {result['n_train_rows']:,}, "
             f"test rows {result['n_test_rows']:,}, matches {result['n_matches']}, iterations {result['n_iter']}"]
    lines.append("")
    lines.append(f"{'overall':>14} | " + " | ".join(f"{e:>10}" for e in engines) + "   (log loss)")
    if m.get("overall"):
        lines.append(f"{'':>14} | " + " | ".join(f"{m['overall'][e]['log_loss']:>10.4f}" for e in engines))
    lines.append("")
    lines.append(f"{'progress':>14} | " + " | ".join(f"{e:>10}" for e in engines))
    for key, recs in m.get("progress", {}).items():
        lines.append(f"{key:>14} | " + " | ".join(f"{recs[e]['log_loss']:>10.4f}" for e in engines))
    lines.append("")
    lines.append(f"{'tour':>14} | " + " | ".join(f"{e:>10}" for e in engines))
    for key, recs in m.get("by_tour", {}).items():
        lines.append(f"{key:>14} | " + " | ".join(f"{recs[e]['log_loss']:>10.4f}" for e in engines))
    lines.append("")
    lines.append(f"{'slice':>18} | {'n':>8} | " + " | ".join(f"{e:>10}" for e in engines))
    for key, recs in m.get("slices", {}).items():
        lines.append(f"{key:>18} | {recs['gbm']['n']:>8} | " + " | ".join(f"{recs[e]['log_loss']:>10.4f}" for e in engines))
    lines.append("")
    lines.append("calibration (gbm): bin, n, mean_pred, actual")
    for r in m.get("calibration_gbm", []):
        lines.append(f"  {r['bin']}  {r['n']:>8}  {r['mean_pred']:.3f}  {r['actual']:.3f}")
    lines.append("")
    lines.append("top feature sensitivities: " + ", ".join(f"{k} {v}" for k, v in list(result.get("feature_sensitivity", {}).items())[:12]))
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# Live forecasting from a scoreboard
# --------------------------------------------------------------------------------------


@dataclass
class LiveStats:
    """Running in-match tallies from player 1's view (all optional)."""
    sv_pts_1: int = 0
    sv_won_1: int = 0
    sv_pts_2: int = 0
    sv_won_2: int = 0
    bp_faced_1: int = 0
    bp_saved_1: int = 0
    bp_faced_2: int = 0
    bp_saved_2: int = 0
    breaks_1: int = 0
    breaks_2: int = 0
    ace_1: int = 0
    ace_2: int = 0
    df_1: int = 0
    df_2: int = 0
    last4_1: int = 2
    last8_1: int = 4
    streak: int = 0
    games_since_break: int = 0
    last_break_by: int = 0
    points_played: int | None = None
    prev_set_margin: int = 0
    prev_set_tb: int = 0


def _load_model(group: str):
    import joblib

    path = MODEL_DIR / f"inplay_{group}.joblib"
    if not path.exists():
        return None
    metrics_path = MODEL_DIR / f"inplay_{group}_metrics.json"
    if metrics_path.exists():
        overall = json.loads(metrics_path.read_text()).get("metrics", {}).get("overall", {})
        gbm, dyn = overall.get("gbm", {}).get("log_loss"), overall.get("dyn", {}).get("log_loss")
        if gbm is not None and dyn is not None and gbm > dyn:
            # the calibrator did not beat the engine it sits on for this segment (too little data):
            # refuse it so the caller falls back to the umbrella model or the engine itself
            return None
    model = joblib.load(path)
    est = getattr(model, "reg", model)
    n_in = getattr(est, "n_features_in_", len(FEATURE_NAMES))
    if n_in != len(FEATURE_NAMES):
        # stale artefact from an older feature layout: fall back to the engine rather than crash
        return None
    return model


def inplay_forecast(
    conn: sqlite3.Connection,
    p1_name: str,
    p2_name: str,
    sets: tuple[int, int] = (0, 0),
    games: tuple[int, int] = (0, 0),
    points: tuple[int | str, int | str] = (0, 0),
    server: int = 1,
    in_tb: bool = False,
    tour: str = "CH",
    surface: str | None = None,
    stats: LiveStats | None = None,
    match_date: str | None = None,
    elo_override: tuple[float, float] | None = None,
    tier: str | None = None,
    event: str | None = None,
) -> dict:
    """Probability bundle for a live best-of-three state.

    The quote is produced by the calibrator of the event *segment* (ATP / CH_BIG / CH_MID /
    CH_SMALL / WTA / ITF_BIG / ITF_SMALL). The segment comes from `tier`, else a named `event`,
    else the two players' latest rankings against this season's Challenger cutoffs.
    """
    from research.analyze import _rating, resolve_player
    from research.markov import parse_point
    from research.tiers import resolve_segment

    stats = stats or LiveStats()
    today = match_date or _date.today().isoformat()

    p1 = resolve_player(conn, p1_name) or {"player_id": None, "name": p1_name}
    p2 = resolve_player(conn, p2_name) or {"player_id": None, "name": p2_name}

    seg = resolve_segment(conn, tour, tier=tier, p1_id=p1.get("player_id"), p2_id=p2.get("player_id"), event=event)
    segment = seg["segment"]
    sex = SEX_OF_GROUP.get(segment, "men")
    consts = (_load_constants(segment) or _load_constants(tour)
              or TourConstants(tour=tour, serve_avg=0.62 if sex == "men" else 0.57, median_points=140))

    def _elo(player):
        if not player.get("player_id"):
            return 1500.0, None
        rate = _rating(conn, player["player_id"], surface)
        return rate["elo"], rate.get("surface_elo")

    if elo_override:
        elo_1, elo_2 = elo_override
        se_1 = se_2 = None
    else:
        (elo_1, se_1), (elo_2, se_2) = _elo(p1), _elo(p2)
    p_elo = prior_prob(elo_1, elo_2, se_1, se_2, segment)
    s1_prior, r1_prior = serve_return_from_match_prob(round(p_elo, 4), round(consts.serve_avg, 4))
    s2_prior = 1.0 - r1_prior

    fatigue = load_fatigue_index(conn, {pid for pid in (p1.get("player_id"), p2.get("player_id")) if pid})
    load1 = player_load(fatigue, p1.get("player_id"), today, None, 5)
    load2 = player_load(fatigue, p2.get("player_id"), today, None, 5)

    pts1, pts2 = parse_point(points[0]), parse_point(points[1])
    state = (sets[0], sets[1], games[0], games[1], pts1, pts2, server, int(in_tb))
    n_played = stats.points_played
    if n_played is None:
        n_played = int((sets[0] + sets[1]) * 60 + (games[0] + games[1]) * 6.3 + pts1 + pts2)
    progress_est = min(n_played / consts.median_points, 0.95)
    k = kappa_schedule(progress_est, *consts.kappa)
    s1 = update_serve(s1_prior, stats.sv_pts_1, stats.sv_won_1, k)
    s2 = update_serve(s2_prior, stats.sv_pts_2, stats.sv_won_2, k)
    p_prior = fast_match_prob(state, s1_prior, r1_prior)
    p_dyn = fast_match_prob(state, s1, 1.0 - s2)
    imp = importance(state, s1, 1.0 - s2)

    # One-point trace so the shared feature builder is used.
    st = np.zeros((1, len(C)), dtype=np.int16)
    st[0, C["idx"]] = n_played
    st[0, C["set1"]], st[0, C["set2"]] = sets
    st[0, C["g1"]], st[0, C["g2"]] = games
    st[0, C["p1"]], st[0, C["p2"]] = pts1, pts2
    st[0, C["in_tb"]] = int(in_tb)
    st[0, C["server"]] = server
    st[0, C["set_idx"]] = sets[0] + sets[1]
    rs = np.zeros((1, len(RS)), dtype=np.float32)
    for name in RS:
        if hasattr(stats, name):
            rs[0, RS[name]] = getattr(stats, name)
    trace = MatchTrace(pbp_id=0, tour=tour, date=today, label=0, p_elo=p_elo, best_of=3, states=st, stats=rs,
                       progress=np.array([progress_est], np.float32), progress_est=np.array([progress_est], np.float32))
    trace.probs = {"elo_markov": np.array([p_prior], np.float32), "dyn": np.array([p_dyn], np.float32),
                   "elo_static": np.array([p_elo], np.float32), "sym": np.array([0.5], np.float32)}
    trace.imp = np.array([imp], np.float32)
    trace.serve_dyn = np.array([[s1, s2]], np.float32)
    level = ("C" if tour == "CH" else "A") if sex == "men" else ("I" if tour == "WTA" else "25")
    ctx = MatchContext(elo=(elo_1, elo_2), surf_elo=(se_1, se_2), surface=surface, level=level,
                       qual=False, load=(load1, load2), prior_serve=(s1_prior, s2_prior), p_elo=p_elo)

    model, model_group = _load_model(segment), segment
    if model is None:
        model, model_group = _load_model(sex), sex
    p_model = p_dyn
    if model is not None:
        X, _ = feature_matrix(trace, ctx, sex, 1, None)
        X[0, F["prev_set_margin"]] = stats.prev_set_margin
        X[0, F["prev_set_tb"]] = stats.prev_set_tb
        Xf, _ = feature_matrix(trace, ctx, sex, 2, None)
        Xf[0, F["prev_set_margin"]] = -stats.prev_set_margin
        Xf[0, F["prev_set_tb"]] = stats.prev_set_tb
        p_a = float(model.predict_proba(X)[0, 1])
        p_b = float(model.predict_proba(Xf)[0, 1])
        p_model = 0.5 * (p_a + (1.0 - p_b))

    from research.markov import p_win_set

    if in_tb:
        p_set = p_win_set(games[0], games[1], 0, 0, server == 1, s1, 1.0 - s2, True, pts1, pts2)
    else:
        p_set = p_win_set(games[0], games[1], pts1, pts2, server == 1, s1, 1.0 - s2, False, 0, 0)
    return {
        "p_set": p_set,
        "p1": {"name": p1.get("name"), "id": p1.get("player_id"), "elo": elo_1, "surface_elo": se_1,
               "serve_prior": s1_prior, "serve_live": s1, "load": load1.__dict__},
        "p2": {"name": p2.get("name"), "id": p2.get("player_id"), "elo": elo_2, "surface_elo": se_2,
               "serve_prior": s2_prior, "serve_live": s2, "load": load2.__dict__},
        "state": {"sets": list(sets), "games": list(games), "points": [points[0], points[1]], "server": server,
                  "in_tb": in_tb, "tour": tour, "surface": surface, "points_played_est": n_played,
                  "progress_est": round(progress_est, 3), "kappa": round(k, 1)},
        "p_elo": p_elo,
        "p_prior": p_prior,
        "p_updated": p_dyn,
        "p_model": p_model,
        "importance": imp,
        "model_loaded": model is not None,
        "model_group": model_group if model is not None else None,
        "group": sex,
        "segment": segment,
        "tier": {k: v for k, v in seg.items() if k != "segment"},
    }


def set_paths(p_set_now: float, sets: tuple[int, int], p_set_fresh: float) -> dict[str, float]:
    """Distribution over the six best-of-three set sequences (c1..c6) from the current state.

    c1 T1,T1  c2 T2,T2  c3 T1,T2,T1  c4 T1,T2,T2  c5 T2,T1,T2  c6 T2,T1,T1
    p_set_now: P(T1 wins the set in progress); p_set_fresh: P(T1 wins a set from 0-0).
    Sets already decided get probability 1 on their branch.
    """
    s1, s2 = sets
    q_now = 1.0 - p_set_now
    q = 1.0 - p_set_fresh
    paths = {k: 0.0 for k in ("c1", "c2", "c3", "c4", "c5", "c6")}
    if (s1, s2) == (0, 0):
        paths["c1"] = p_set_now * p_set_fresh
        paths["c2"] = q_now * q
        paths["c3"] = p_set_now * q * p_set_fresh
        paths["c4"] = p_set_now * q * q
        paths["c5"] = q_now * p_set_fresh * q
        paths["c6"] = q_now * p_set_fresh * p_set_fresh
    elif (s1, s2) == (1, 0):
        paths["c1"] = p_set_now
        paths["c3"] = q_now * p_set_fresh
        paths["c4"] = q_now * q
    elif (s1, s2) == (0, 1):
        paths["c2"] = q_now
        paths["c5"] = p_set_now * q
        paths["c6"] = p_set_now * p_set_fresh
    else:  # 1-1: which path depends on who took set 1; expose both deciders
        paths["c3"] = paths["c6"] = p_set_now / 2
        paths["c4"] = paths["c5"] = q_now / 2
    return paths


def inplay_bundle(
    conn: sqlite3.Connection,
    p1_name: str,
    p2_name: str,
    tour: str = "CH",
    surface: str | None = None,
    sets: tuple[int, int] = (0, 0),
    games: tuple[int, int] = (0, 0),
    points: tuple[int | str, int | str] = (0, 0),
    server: int = 1,
    in_tb: bool = False,
    serve_1: tuple[int, int] = (0, 0),
    serve_2: tuple[int, int] = (0, 0),
    prev_set: tuple[int, int] | None = None,
    streak: int = 0,
    mto_1: float | None = None,
    mto_2: float | None = None,
    rule: str = "ball_served",
    price_1: float | None = None,
    price_2: float | None = None,
    log: bool = False,
    match_date: str | None = None,
    tier: str | None = None,
    event: str | None = None,
) -> dict:
    """Full output contract: probabilities, retirement overlay, settlement, market decision, paths."""
    from research import hazard as hz
    from research.market import decision, log_bundle

    stats = LiveStats(
        sv_won_1=serve_1[0], sv_pts_1=serve_1[1], sv_won_2=serve_2[0], sv_pts_2=serve_2[1],
        streak=streak,
        last4_1=min(4, max(0, 2 + streak)) if streak else 2,
        last8_1=min(8, max(0, 4 + streak)) if streak else 4,
        prev_set_margin=(prev_set[0] - prev_set[1]) if prev_set else 0,
        prev_set_tb=1 if prev_set and sorted(prev_set) == [6, 7] else 0,
    )
    fc = inplay_forecast(conn, p1_name, p2_name, sets=sets, games=games, points=points, server=server,
                         in_tb=in_tb, tour=tour, surface=surface, stats=stats, match_date=match_date,
                         tier=tier, event=event)
    group = fc["group"]
    bucket = fc["segment"]
    ov = hz.overlay(conn, group, bucket, surface, sets, games, fc["p_set"], fc["p_model"],
                    fc["p1"]["id"], fc["p2"]["id"], mto_1, mto_2, rule, as_of=match_date)
    fc["p_settled"] = ov["p_settled"]
    fc["retirement"] = ov

    s1, s2 = fc["p1"]["serve_live"], fc["p2"]["serve_live"]
    from research.markov import p_win_set

    p_set_fresh = 0.5 * (p_win_set(0, 0, 0, 0, True, s1, 1.0 - s2, False, 0, 0)
                         + p_win_set(0, 0, 0, 0, False, s1, 1.0 - s2, False, 0, 0))
    fc["paths"] = set_paths(fc["p_set"], sets, p_set_fresh)

    fc["market"] = None
    if price_1 or price_2:
        fc["market"] = {
            "p1": decision(fc["p_settled"], price_1, price_2) if price_1 else None,
            "p2": decision(1.0 - fc["p_settled"], price_2, price_1) if price_2 else None,
        }
    fc["flags"] = {
        "post_break": False,
        "mto_active": bool(mto_1 is not None or mto_2 is not None),
        "fatigue_carryover_1": fc["p1"]["load"]["rest_days"] <= 1 and fc["p1"]["load"]["prev_games"] >= 23,
        "fatigue_carryover_2": fc["p2"]["load"]["rest_days"] <= 1 and fc["p2"]["load"]["prev_games"] >= 23,
        "clear_favourite": abs(fc["p_elo"] - 0.5) >= 0.3,
    }
    if log:
        price = price_1
        fc["paper_log_id"] = log_bundle(conn, fc, price=price)
    return fc
