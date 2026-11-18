"""Pre-match prior: overall Elo blended with surface Elo, per segment.

The point engine inherits whatever error the prior carries for the first half of a match,
so the prior is worth tuning. Plain overall Elo is beaten in every segment by

    d = scale * ( w * (elo_a - elo_b) + (1 - w) * (surf_a - surf_b) )
    p = 1 / (1 + 10 ** (-d / 400))

`w` and `scale` are fitted per segment on matches before the chronological split. The scale
matters most on the women's ITF circuit, where thousands of players enter near 1500 and the
rating gap understates the real skill gap.
"""

from __future__ import annotations

import json
import sqlite3

import numpy as np

from research.config import DATA_DIR

MODEL_DIR = DATA_DIR / "models"
PARAMS_PATH = MODEL_DIR / "prior_params.json"

DEFAULT = {"w_overall": 1.0, "scale": 1.0}
W_GRID = (0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
SCALE_GRID = (0.9, 1.0, 1.1, 1.2, 1.3, 1.4)

_cache: dict | None = None


def load_params() -> dict:
    global _cache
    if _cache is None:
        _cache = json.loads(PARAMS_PATH.read_text()) if PARAMS_PATH.exists() else {}
    return _cache


def prior_prob(elo_a: float, elo_b: float, surf_a: float | None, surf_b: float | None,
               segment: str | None = None) -> float:
    """P(a beats b) from the blended prior. Falls back to plain Elo when surface Elo is missing."""
    prm = load_params().get(segment or "", DEFAULT)
    w = prm["w_overall"]
    if surf_a is None or surf_b is None:
        w = 1.0
        surf_a = surf_b = 0.0
    d = prm["scale"] * (w * (elo_a - elo_b) + (1.0 - w) * (surf_a - surf_b))
    return float(1.0 / (1.0 + 10 ** (-d / 400.0)))


def _log_loss(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def fit_prior_params(conn: sqlite3.Connection, train_until: str = "2016-01-01", since: str = "2010-01-01") -> dict:
    """Grid-fit (w, scale) per segment on completed best-of-3 main-draw matches before `train_until`."""
    global _cache
    rows = conn.execute(
        """
        SELECT et.segment, m.winner_elo w, m.loser_elo l, m.winner_surf_elo ws, m.loser_surf_elo ls
        FROM matches m JOIN event_tiers et ON et.tourney_id = m.tourney_id
        WHERE m.tourney_date >= ? AND m.tourney_date < ? AND m.completion = 'Completed'
          AND m.round NOT LIKE 'Q%' AND COALESCE(m.best_of, 3) = 3 AND m.winner_elo IS NOT NULL
        """,
        (since, train_until),
    ).fetchall()
    by_seg: dict[str, list] = {}
    for r in rows:
        by_seg.setdefault(r["segment"], []).append(r)

    out = {}
    rng = np.random.default_rng(0)
    for seg, rs in sorted(by_seg.items()):
        w_ = np.array([r["w"] for r in rs], float)
        l_ = np.array([r["l"] for r in rs], float)
        ws = np.array([r["ws"] if r["ws"] is not None else r["w"] for r in rs], float)
        ls = np.array([r["ls"] if r["ls"] is not None else r["l"] for r in rs], float)
        flip = rng.random(len(rs)) < 0.5
        a, b = np.where(flip, l_, w_), np.where(flip, w_, l_)
        sa, sb = np.where(flip, ls, ws), np.where(flip, ws, ls)
        y = (~flip).astype(float)
        base = _log_loss(1 / (1 + 10 ** ((b - a) / 400)), y)
        best = (base, 1.0, 1.0)
        for w in W_GRID:
            for scale in SCALE_GRID:
                d = scale * (w * (a - b) + (1 - w) * (sa - sb))
                v = _log_loss(1 / (1 + 10 ** (-d / 400)), y)
                if v < best[0] - 1e-6:
                    best = (v, w, scale)
        out[seg] = {"w_overall": best[1], "scale": best[2], "n": len(rs),
                    "log_loss": round(best[0], 4), "elo_only_log_loss": round(base, 4)}
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    PARAMS_PATH.write_text(json.dumps(out, indent=2))
    _cache = out
    return out
