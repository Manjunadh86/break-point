"""Replay harness: score in-play engines point by point on historical matches.

For every point of every point-by-point match we compute P(player 1 wins the match)
under several engines and score them against the result:

  elo_static   pre-match Elo probability, never updated (the "static Elo" baseline)
  sym          symmetric iid Markov chain (tour-average serve, no prior)
  elo_markov   iid Markov chain with Elo-derived serve/return probabilities, fixed
  dyn          same prior, serve probabilities updated in-match (empirical Bayes)

Metrics are reported by match progress, by tour and by slice, on a chronological
split. This is the instrument every later model change is judged with.
"""

from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from dataclasses import dataclass, field
from statistics import median

import numpy as np

from research.analyze import elo_win_prob
from research.config import DATA_DIR
from research.markov import (
    fast_match_prob,
    importance,
    kappa_schedule,
    serve_return_from_match_prob,
    update_serve,
)
from research.pbp import C, RS, parse_match, running_stats

MODEL_DIR = DATA_DIR / "models"
PROGRESS_BINS = [(0.0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 0.9), (0.9, 1.01)]
ENGINES = ("elo_static", "sym", "elo_markov", "dyn")


@dataclass
class TourConstants:
    tour: str
    serve_avg: float          # P(server wins point), tour mean in the training period
    median_points: int        # typical completed bo3 length, for the non-leaky progress estimate
    kappa: tuple[float, float, float] = (640.0, 160.0, 40.0)
    n_matches: int = 0


@dataclass
class MatchTrace:
    pbp_id: int
    tour: str
    date: str
    label: int                     # 1 if player 1 won
    p_elo: float
    best_of: int
    states: np.ndarray
    stats: np.ndarray
    progress: np.ndarray           # true fraction of the match played (evaluation only)
    progress_est: np.ndarray       # live-computable estimate
    probs: dict[str, np.ndarray] = field(default_factory=dict)
    imp: np.ndarray | None = None
    serve_dyn: np.ndarray | None = None   # (n, 2) updated serve probs for P1 and P2


def load_pbp(
    conn: sqlite3.Connection,
    tours: tuple[str, ...] = ("CH",),
    draw: str | None = "Main",
    best_of: int = 3,
    linked_only: bool = True,
    limit: int | None = None,
    segments: tuple[str, ...] | None = None,
) -> list[sqlite3.Row]:
    """Point-by-point rows. `segments` (e.g. ("CH_SMALL",)) filters by event tier, see research.tiers."""
    sql = """
        SELECT id, date, tour, draw, p1, p2, winner, pbp, score, best_of,
               p1_elo, p2_elo, p1_surf_elo, p2_surf_elo, surface, tourney_level, match_id, p1_id, p2_id, segment
        FROM pbp_matches
        WHERE valid = 1 AND best_of = ?
    """
    params: list = [best_of]
    if segments:
        sql += f" AND segment IN ({','.join('?' * len(segments))})"
        params.extend(segments)
    else:
        sql += f" AND tour IN ({','.join('?' * len(tours))})"
        params.extend(tours)
    if draw:
        sql += " AND draw = ?"
        params.append(draw)
    if linked_only:
        sql += " AND p1_elo IS NOT NULL AND p2_elo IS NOT NULL"
    sql += " ORDER BY date, id"
    if limit:
        sql += f" LIMIT {int(limit)}"
    return conn.execute(sql, params).fetchall()


def tour_constants(rows: list[sqlite3.Row], tour: str, train_until: str) -> TourConstants:
    sv_pts = sv_won = 0
    lengths = []
    n = 0
    for r in rows:
        if r["date"] >= train_until:
            continue
        parsed = parse_match(r["pbp"], 1, r["score"], r["winner"])
        if not parsed.valid:
            continue
        st = parsed.states
        sv_pts += len(st)
        sv_won += int((st[:, C["server"]] == st[:, C["winner"]]).sum())
        lengths.append(len(st))
        n += 1
    serve_avg = sv_won / sv_pts if sv_pts else 0.62
    return TourConstants(tour=tour, serve_avg=serve_avg, median_points=int(median(lengths)) if lengths else 150, n_matches=n)


def build_trace(row: sqlite3.Row, consts: TourConstants) -> MatchTrace | None:
    parsed = parse_match(row["pbp"], 1, row["score"], row["winner"])
    if not parsed.valid or parsed.best_of != 3:
        return None
    st = parsed.states
    n = len(st)
    stats = running_stats(st)
    from research.prior import prior_prob

    keys = row.keys()
    p_elo = prior_prob(row["p1_elo"], row["p2_elo"],
                       row["p1_surf_elo"] if "p1_surf_elo" in keys else None,
                       row["p2_surf_elo"] if "p2_surf_elo" in keys else None,
                       row["segment"] if "segment" in keys else None)
    progress = (np.arange(n) / max(n, 1)).astype(np.float32)
    progress_est = np.minimum(np.arange(n) / consts.median_points, 0.95).astype(np.float32)
    return MatchTrace(
        pbp_id=row["id"],
        tour=row["tour"],
        date=row["date"],
        label=1 if row["winner"] == 1 else 0,
        p_elo=p_elo,
        best_of=parsed.best_of,
        states=st,
        stats=stats,
        progress=progress,
        progress_est=progress_est,
    )


def _state(row: np.ndarray) -> tuple[int, ...]:
    return (
        int(row[C["set1"]]), int(row[C["set2"]]), int(row[C["g1"]]), int(row[C["g2"]]),
        int(row[C["p1"]]), int(row[C["p2"]]), int(row[C["server"]]), int(row[C["in_tb"]]),
    )


def run_engines(trace: MatchTrace, consts: TourConstants, kappa: tuple[float, float, float] | None = None,
                with_importance: bool = False) -> MatchTrace:
    """Fill trace.probs for every engine. kappa overrides the tour default."""
    kappa = kappa or consts.kappa
    n = len(trace.states)
    sym_s, sym_r = consts.serve_avg, 1.0 - consts.serve_avg
    s1_prior, r1_prior = serve_return_from_match_prob(round(trace.p_elo, 4), round(consts.serve_avg, 4))
    s2_prior = 1.0 - r1_prior

    p_sym = np.empty(n, dtype=np.float32)
    p_elo_m = np.empty(n, dtype=np.float32)
    p_dyn = np.empty(n, dtype=np.float32)
    serve_dyn = np.empty((n, 2), dtype=np.float32)
    imp = np.empty(n, dtype=np.float32) if with_importance else None

    for i in range(n):
        state = _state(trace.states[i])
        p_sym[i] = fast_match_prob(state, sym_s, sym_r)
        p_elo_m[i] = fast_match_prob(state, s1_prior, r1_prior)
        k = kappa_schedule(float(trace.progress_est[i]), *kappa)
        st = trace.stats[i]
        s1 = update_serve(s1_prior, int(st[RS["sv_pts_1"]]), int(st[RS["sv_won_1"]]), k)
        s2 = update_serve(s2_prior, int(st[RS["sv_pts_2"]]), int(st[RS["sv_won_2"]]), k)
        serve_dyn[i, 0], serve_dyn[i, 1] = s1, s2
        p_dyn[i] = fast_match_prob(state, s1, 1.0 - s2)
        if with_importance:
            imp[i] = importance(state, s1, 1.0 - s2)

    trace.probs = {
        "elo_static": np.full(n, trace.p_elo, dtype=np.float32),
        "sym": p_sym,
        "elo_markov": p_elo_m,
        "dyn": p_dyn,
    }
    trace.serve_dyn = serve_dyn
    trace.imp = imp
    return trace


# --------------------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------------------


def _log_loss(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p.astype(np.float64), 1e-4, 1 - 1e-4)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p.astype(np.float64) - y) ** 2))


def _acc(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p >= 0.5) == (y == 1)))


def slice_masks(trace: MatchTrace) -> dict[str, np.ndarray]:
    st = trace.states
    stats = trace.stats
    set1, set2 = st[:, C["set1"]], st[:, C["set2"]]
    fav_is_p1 = trace.p_elo >= 0.5
    fav_sets = set1 if fav_is_p1 else set2
    dog_sets = set2 if fav_is_p1 else set1
    clear_fav = abs(trace.p_elo - 0.5) >= 0.15
    return {
        "all": np.ones(len(st), dtype=bool),
        "fav_down_a_set": (fav_sets < dog_sets) & clear_fav,
        "fav_up_a_set": (fav_sets > dog_sets) & clear_fav,
        "decider": (set1 + set2) == 2,
        "post_break_game": (stats[:, RS["games_since_break"]] == 0) & (stats[:, RS["last_break_by"]] > 0),
        "tiebreak": st[:, C["in_tb"]] == 1,
        "heavy_fav": np.full(len(st), abs(trace.p_elo - 0.5) >= 0.3),
    }


def score_traces(traces: list[MatchTrace], engines: tuple[str, ...] = ENGINES) -> dict:
    """Aggregate log loss / Brier / accuracy by progress bin, overall and by slice."""
    buckets: dict[tuple, list] = defaultdict(lambda: [[], []])
    for tr in traces:
        y = np.full(len(tr.states), tr.label, dtype=np.float64)
        masks = slice_masks(tr)
        for eng in engines:
            p = tr.probs[eng]
            for lo, hi in PROGRESS_BINS:
                m = (tr.progress >= lo) & (tr.progress < hi)
                if m.any():
                    b = buckets[(eng, "progress", f"{int(lo*100)}-{int(min(hi,1.0)*100)}")]
                    b[0].append(p[m])
                    b[1].append(y[m])
            for name, m in masks.items():
                if m.any():
                    b = buckets[(eng, "slice", name)]
                    b[0].append(p[m])
                    b[1].append(y[m])
    out: dict = {"n_matches": len(traces), "n_points": int(sum(len(t.states) for t in traces)), "progress": {}, "slices": {}}
    for (eng, kind, key), (ps, ys) in buckets.items():
        p = np.concatenate(ps)
        y = np.concatenate(ys)
        rec = {"n": int(len(p)), "log_loss": round(_log_loss(p, y), 4), "brier": round(_brier(p, y), 4),
               "accuracy": round(_acc(p, y), 4), "mean_pred": round(float(p.mean()), 4), "mean_actual": round(float(y.mean()), 4)}
        target = out["progress"] if kind == "progress" else out["slices"]
        target.setdefault(key, {})[eng] = rec
    return out


def calibration_table(traces: list[MatchTrace], engine: str = "dyn", bins: int = 10) -> list[dict]:
    p = np.concatenate([t.probs[engine] for t in traces]).astype(np.float64)
    y = np.concatenate([np.full(len(t.states), t.label) for t in traces]).astype(np.float64)
    edges = np.linspace(0, 1, bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi) if hi < 1 else (p >= lo) & (p <= hi)
        if m.sum() == 0:
            continue
        rows.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": int(m.sum()), "mean_pred": round(float(p[m].mean()), 4),
                     "actual": round(float(y[m].mean()), 4)})
    return rows


# --------------------------------------------------------------------------------------
# Kappa fitting
# --------------------------------------------------------------------------------------

KAPPA_GRID = [
    (e, m, l)
    for e in (160.0, 320.0, 640.0, 1280.0, 1e9)
    for m in (40.0, 80.0, 160.0, 320.0, 640.0)
    for l in (10.0, 20.0, 40.0, 80.0, 160.0)
    if e >= m >= l
]


def fit_kappa(traces: list[MatchTrace], consts: TourConstants, sample: int = 1200, seed: int = 7) -> dict:
    """Grid-search the (early, mid, late) pseudo-counts that minimise dyn log loss."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(traces), size=min(sample, len(traces)), replace=False)
    sub = [traces[i] for i in idx]
    y_all = np.concatenate([np.full(len(t.states), t.label, dtype=np.float64) for t in sub])
    results = []
    for kappa in KAPPA_GRID:
        ps = []
        for tr in sub:
            run_engines(tr, consts, kappa)
            ps.append(tr.probs["dyn"])
        ll = _log_loss(np.concatenate(ps), y_all)
        results.append((ll, kappa))
    results.sort()
    best_ll, best = results[0]
    # static reference on the same sample
    ps = np.concatenate([t.probs["elo_markov"] for t in sub])
    return {
        "best": {"early": best[0], "mid": best[1], "late": best[2], "log_loss": round(best_ll, 4)},
        "elo_markov_log_loss": round(_log_loss(ps, y_all), 4),
        "top5": [{"kappa": k, "log_loss": round(ll, 4)} for ll, k in results[:5]],
        "n_sample_matches": len(sub),
    }


# --------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------


def run_replay(
    conn: sqlite3.Connection,
    tour: str = "CH",
    train_until: str = "2016-01-01",
    fit: bool = True,
    limit: int | None = None,
    save: bool = True,
) -> dict:
    from research.tiers import ALL_SEGMENTS

    tour = tour.upper()
    if tour in ALL_SEGMENTS:
        rows = load_pbp(conn, segments=(tour,), limit=limit)
    else:
        tours = (tour,) if tour != "ALL" else ("ATP", "CH", "WTA", "ITF")
        rows = load_pbp(conn, tours=tours, limit=limit)
    consts = tour_constants(rows, tour, train_until)
    traces = [t for t in (build_trace(r, consts) for r in rows) if t is not None]
    train = [t for t in traces if t.date < train_until]
    test = [t for t in traces if t.date >= train_until]
    print(f"{tour}: {len(traces)} matches ({len(train)} train / {len(test)} test), "
          f"serve avg {consts.serve_avg:.4f}, median points {consts.median_points}", flush=True)

    kappa_fit = None
    if fit and train:
        print(f"fitting kappa on {min(1200, len(train))} training matches x {len(KAPPA_GRID)} grid points …", flush=True)
        kappa_fit = fit_kappa(train, consts)
        consts.kappa = (kappa_fit["best"]["early"], kappa_fit["best"]["mid"], kappa_fit["best"]["late"])
        print(f"  best kappa {consts.kappa} log loss {kappa_fit['best']['log_loss']} "
              f"(static elo_markov {kappa_fit['elo_markov_log_loss']})", flush=True)

    for tr in test:
        run_engines(tr, consts)
    metrics = score_traces(test) if test else {}
    result = {
        "tour": tour,
        "train_until": train_until,
        "constants": {"serve_avg": round(consts.serve_avg, 4), "median_points": consts.median_points,
                      "kappa": list(consts.kappa), "n_train_matches": len(train)},
        "kappa_fit": kappa_fit,
        "test": metrics,
        "calibration_dyn": calibration_table(test, "dyn") if test else [],
        "calibration_elo_markov": calibration_table(test, "elo_markov") if test else [],
    }
    if save:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        (MODEL_DIR / f"replay_{tour}.json").write_text(json.dumps(result, indent=2))
        (MODEL_DIR / f"tour_constants_{tour}.json").write_text(json.dumps(result["constants"], indent=2))
    return result


def format_report(result: dict) -> str:
    lines = [f"Replay {result['tour']} — test from {result['train_until']}; "
             f"{result['test'].get('n_matches', 0)} matches, {result['test'].get('n_points', 0)} points"]
    c = result["constants"]
    lines.append(f"tour serve avg {c['serve_avg']}, median points {c['median_points']}, kappa {c['kappa']}")
    lines.append("")
    lines.append(f"{'progress':>10} | " + " | ".join(f"{e:>11}" for e in ENGINES) + "   (log loss)")
    for key, recs in result["test"].get("progress", {}).items():
        lines.append(f"{key:>10} | " + " | ".join(f"{recs[e]['log_loss']:>11.4f}" for e in ENGINES if e in recs))
    lines.append("")
    lines.append(f"{'slice':>18} | {'n':>8} | " + " | ".join(f"{e:>11}" for e in ENGINES) + "   (log loss)")
    for key, recs in result["test"].get("slices", {}).items():
        n = recs[ENGINES[0]]["n"]
        lines.append(f"{key:>18} | {n:>8} | " + " | ".join(f"{recs[e]['log_loss']:>11.4f}" for e in ENGINES if e in recs))
    lines.append("")
    lines.append("calibration (dyn): bin, n, mean_pred, actual")
    for r in result.get("calibration_dyn", []):
        lines.append(f"  {r['bin']}  {r['n']:>8}  {r['mean_pred']:.3f}  {r['actual']:.3f}")
    return "\n".join(lines)
