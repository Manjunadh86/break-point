"""Market comparison and staking.

The market is the benchmark, never a feature of the probability model. This module
turns a model probability and a price into edge, a capped fractional-Kelly stake,
closing-line value, and a paper-trading log that doubles as the next training set.

Nothing here places a bet or talks to a sportsbook.
"""

from __future__ import annotations

import bz2
import json
import math
import sqlite3
from datetime import datetime
from pathlib import Path

from research.analyze import devig_two_way, implied_from_decimal

COMMISSION = 0.05           # exchange commission on net winnings
KELLY_FRACTION = 0.25
MAX_STAKE_FRACTION = 0.03   # of bankroll, per bet
MIN_EDGE = 0.03             # probability points, before uncertainty scaling


def edge(p: float, price: float, commission: float = COMMISSION) -> float:
    """Expected return per unit staked at decimal `price` with commission on winnings."""
    if price <= 1.0:
        return -1.0
    net_win = (price - 1.0) * (1.0 - commission)
    return p * net_win - (1.0 - p)


def kelly_fraction(p: float, price: float, fraction: float = KELLY_FRACTION, cap: float = MAX_STAKE_FRACTION,
                   commission: float = COMMISSION) -> float:
    """Fractional Kelly stake as a share of bankroll, capped. 0 when there is no edge."""
    if price <= 1.0:
        return 0.0
    b = (price - 1.0) * (1.0 - commission)
    q = 1.0 - p
    full = (b * p - q) / b
    if full <= 0:
        return 0.0
    return min(full * fraction, cap)


def expected_log_growth(p: float, price: float, stake: float, commission: float = COMMISSION) -> float:
    b = (price - 1.0) * (1.0 - commission)
    if stake <= 0 or stake >= 1:
        return 0.0
    return p * math.log1p(b * stake) + (1.0 - p) * math.log1p(-stake)


def edge_threshold(p_low: float, p_high: float, base: float = MIN_EDGE) -> float:
    """Require more edge when the model is less sure (wider interval)."""
    return base + 0.5 * max(0.0, p_high - p_low)


def decision(p_model: float, price_back: float, price_lay: float | None = None, p_low: float | None = None,
             p_high: float | None = None, commission: float = COMMISSION) -> dict:
    """Compare the model to a two-way market and size a stake, or explain why not."""
    market = devig_two_way(price_back, price_lay) if price_lay else {"prob_a": implied_from_decimal(price_back)}
    market_p = market.get("prob_a")
    e = edge(p_model, price_back, commission)
    thr = edge_threshold(p_low if p_low is not None else p_model, p_high if p_high is not None else p_model)
    prob_edge = (p_model - market_p) if market_p is not None else None
    ok = (prob_edge is None or prob_edge >= thr) and e > 0
    stake = kelly_fraction(p_model, price_back, commission=commission) if ok else 0.0
    return {
        "price": price_back,
        "market_p": market_p,
        "overround": market.get("overround"),
        "edge_return": round(e, 4),
        "edge_prob": round(prob_edge, 4) if prob_edge is not None else None,
        "threshold_prob": round(thr, 4),
        "stake_fraction": round(stake, 4),
        "expected_log_growth": round(expected_log_growth(p_model, price_back, stake, commission), 6) if stake else 0.0,
        "action": "back" if stake > 0 else "pass",
    }


def closing_line_value(entry_price: float, closing_price: float) -> float:
    """CLV in probability points: implied(close) - implied(entry). Positive = beat the close."""
    ie, ic = implied_from_decimal(entry_price), implied_from_decimal(closing_price)
    if ie is None or ic is None:
        return 0.0
    return ic - ie


# --------------------------------------------------------------------------------------
# Paper-trading log
# --------------------------------------------------------------------------------------


def _ensure_paper_columns(conn: sqlite3.Connection) -> None:
    cols = {r[1] for r in conn.execute("PRAGMA table_info(paper_log)")}
    for name, typ in (("segment", "TEXT"), ("winner", "INTEGER"), ("closing_price", "REAL"), ("settled_at", "TEXT")):
        if name not in cols:
            conn.execute(f"ALTER TABLE paper_log ADD COLUMN {name} {typ}")


def _now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def log_bundle(conn: sqlite3.Connection, bundle: dict, event_id: str | None = None, price: float | None = None,
               market_p: float | None = None, note: str | None = None) -> int:
    _ensure_paper_columns(conn)
    dec = decision(bundle["p_model"], price) if price else None
    cur = conn.execute(
        """
        INSERT INTO paper_log (logged_at, event_id, p1, p2, tour, surface, state,
                               p_prior, p_updated, p_model, p_settled, market_p, price, edge, stake_frac, note, segment)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            _now(),
            event_id,
            bundle["p1"]["name"],
            bundle["p2"]["name"],
            bundle["state"].get("tour"),
            bundle["state"].get("surface"),
            json.dumps(bundle["state"]),
            bundle.get("p_prior"),
            bundle.get("p_updated"),
            bundle.get("p_model"),
            bundle.get("p_settled"),
            market_p if market_p is not None else (dec["market_p"] if dec else None),
            price,
            dec["edge_prob"] if dec else None,
            dec["stake_fraction"] if dec else None,
            note,
            bundle.get("segment"),
        ),
    )
    conn.commit()
    return int(cur.lastrowid)


def settle_paper(conn: sqlite3.Connection, log_id: int, winner: int, closing_price: float | None = None) -> dict:
    """Record who actually won (1 = P1, 2 = P2) and, optionally, the closing price on P1."""
    if winner not in (1, 2):
        raise ValueError("winner must be 1 or 2")
    _ensure_paper_columns(conn)
    cur = conn.execute(
        "UPDATE paper_log SET winner = ?, closing_price = COALESCE(?, closing_price), settled_at = ? WHERE id = ?",
        (winner, closing_price, _now(), log_id),
    )
    conn.commit()
    if cur.rowcount == 0:
        raise KeyError(f"paper_log id {log_id} not found")
    return paper_rows(conn, ids=(log_id,))[0]


def paper_rows(conn: sqlite3.Connection, limit: int = 50, ids: tuple[int, ...] | None = None) -> list[dict]:
    _ensure_paper_columns(conn)
    sql = "SELECT * FROM paper_log"
    params: list = []
    if ids:
        sql += f" WHERE id IN ({','.join('?' * len(ids))})"
        params.extend(ids)
    sql += " ORDER BY id DESC LIMIT ?"
    params.append(limit)
    out = []
    for r in conn.execute(sql, params):
        d = dict(r)
        try:
            d["state"] = json.loads(d["state"]) if d.get("state") else None
        except json.JSONDecodeError:
            pass
        out.append(d)
    return out


def _score(ps: list[float], ys: list[int]) -> dict | None:
    if not ps:
        return None
    ll = -sum(y * math.log(max(p, 1e-4)) + (1 - y) * math.log(max(1 - p, 1e-4)) for p, y in zip(ps, ys)) / len(ps)
    brier = sum((p - y) ** 2 for p, y in zip(ps, ys)) / len(ps)
    return {"n": len(ps), "log_loss": round(ll, 4), "brier": round(brier, 4),
            "mean_pred": round(sum(ps) / len(ps), 4), "hit_rate": round(sum(ys) / len(ys), 4)}


def paper_summary(conn: sqlite3.Connection) -> dict:
    """Scores the log on the quotes you have settled: model vs market vs prior, flat and Kelly P&L, CLV."""
    _ensure_paper_columns(conn)
    row = conn.execute(
        "SELECT COUNT(*) n, AVG(p_model) avg_p, AVG(edge) avg_edge, SUM(stake_frac > 0) bets, "
        "SUM(winner IS NOT NULL) settled, MIN(logged_at) first, MAX(logged_at) last FROM paper_log"
    ).fetchone()
    out = {k: row[k] for k in row.keys()}
    settled = conn.execute("SELECT * FROM paper_log WHERE winner IS NOT NULL").fetchall()
    y = [1 if r["winner"] == 1 else 0 for r in settled]
    out["model"] = _score([r["p_model"] for r in settled if r["p_model"] is not None], y)
    out["prior"] = _score([r["p_prior"] for r in settled if r["p_prior"] is not None], y)
    mk = [(r["market_p"], 1 if r["winner"] == 1 else 0) for r in settled if r["market_p"] is not None]
    out["market"] = _score([m for m, _ in mk], [yy for _, yy in mk])

    flat = kelly = 0.0
    n_bets = 0
    clv = []
    for r in settled:
        if r["price"] and r["price"] > 1 and r["stake_frac"] and r["stake_frac"] > 0:
            n_bets += 1
            won = r["winner"] == 1
            ret = (r["price"] - 1) * (1 - COMMISSION) if won else -1.0
            flat += ret
            kelly += r["stake_frac"] * ret
        if r["price"] and r["closing_price"]:
            clv.append(closing_line_value(r["price"], r["closing_price"]))
    out["bets_settled"] = n_bets
    out["flat_stake_units"] = round(flat, 3)
    out["kelly_bankroll_growth"] = round(kelly, 4)
    out["avg_clv"] = round(sum(clv) / len(clv), 4) if clv else None
    out["note"] = ("P&L is on P1 backs with 5% commission; CLV needs a closing price typed at settlement. "
                   "Fewer than ~50 settled quotes says nothing reliable about the model.")
    return out


# --------------------------------------------------------------------------------------
# Betfair historical (Basic tier) loader
# --------------------------------------------------------------------------------------


def load_betfair_market(path: str | Path) -> list[dict]:
    """Read one Betfair historical market file (bz2 or plain JSON lines) into
    [{t: iso, runner_id, ltp, status, in_play}]. Basic tier = last traded price at 1-min."""
    p = Path(path)
    opener = bz2.open if p.suffix == ".bz2" else open
    out: list[dict] = []
    with opener(p, "rt", encoding="utf-8") as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            t = datetime.utcfromtimestamp(rec.get("pt", 0) / 1000).isoformat(timespec="seconds") + "Z"
            for mc in rec.get("mc", []):
                md = mc.get("marketDefinition", {})
                for rc in mc.get("rc", []):
                    if "ltp" in rc:
                        out.append({"t": t, "runner_id": rc.get("id"), "ltp": rc["ltp"],
                                    "status": md.get("status"), "in_play": md.get("inPlay")})
    return out
