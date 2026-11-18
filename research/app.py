from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from research.analyze import (
    add_watch,
    challenger_context,
    delete_watch,
    find_players,
    list_watchlist,
    market_report,
    matchup_report,
    recent_challengers,
    snapshot_curves,
    stats_overview,
)
from research.live import live_quote, live_study
from research.live_feed import enrich, get_match, live_slate, slate_meta
from research.predictor import METRICS_PATH, forecast
from research.db import session

WEB = Path(__file__).resolve().parent / "web"

app = FastAPI(title="Break Point", description="Local tennis research desk. No betting automation.")
app.mount("/static", StaticFiles(directory=WEB), name="static")


class WatchIn(BaseModel):
    sport: str = "challenger"
    player_a: str
    player_b: str
    surface: str | None = None
    tournament: str | None = None
    start_time: str | None = None
    notes: str | None = None
    line_a: float | None = None
    line_b: float | None = None


@app.get("/")
def index():
    return FileResponse(WEB / "index.html")


@app.get("/api/overview")
def overview():
    with session() as conn:
        return stats_overview(conn)


@app.get("/api/search")
def search(q: str):
    with session() as conn:
        return find_players(conn, q)


@app.get("/api/matchup")
def matchup(
    a: str,
    b: str,
    surface: str | None = None,
    line_a: float | None = None,
    line_b: float | None = None,
):
    with session() as conn:
        report = matchup_report(conn, a, b, surface=surface, line_a=line_a, line_b=line_b)
    if report.get("error"):
        raise HTTPException(404, report)
    return report


@app.get("/api/challenger")
def challenger(surface: str | None = None, limit: int = 40):
    with session() as conn:
        return recent_challengers(conn, surface=surface, limit=limit)


@app.get("/api/context")
def context():
    with session() as conn:
        return challenger_context(conn)


@app.get("/api/market")
def market():
    with session() as conn:
        return market_report(conn)


@app.get("/api/snapshots")
def snapshots():
    with session() as conn:
        return snapshot_curves(conn)


@app.get("/api/live-study")
def live_study_api(level: str = "C", surface: str | None = None):
    with session() as conn:
        return live_study(conn, level=level, surface=surface or None)


@app.get("/api/live-quote")
def live_quote_api(
    a: str,
    b: str,
    sets_a: int = 0,
    sets_b: int = 0,
    surface: str | None = None,
    set1: str | None = None,
    line_a: float | None = None,
    line_b: float | None = None,
    level: str = "C",
):
    with session() as conn:
        report = live_quote(
            conn,
            a,
            b,
            sets_a=sets_a,
            sets_b=sets_b,
            surface=surface,
            set1_games=set1,
            line_a=line_a,
            line_b=line_b,
            level=level,
        )
    if report.get("error"):
        raise HTTPException(404, report)
    return report


@app.get("/api/slate")
def slate(all: bool = False):
    with session() as conn:
        rows = enrich(conn, live_slate(include_finals=all))
    return {"meta": slate_meta(), "matches": rows}


@app.get("/api/slate/{match_id}")
def slate_match(match_id: str, fresh: bool = True):
    """One match from the public feed, re-fetched so a follower sees the newest sets/games/server."""
    row = get_match(match_id, fresh=fresh)
    if not row:
        raise HTTPException(404, {"error": "match_not_found"})
    with session() as conn:
        enrich(conn, [row])
    return row


@app.get("/api/slate/{match_id}/forecast")
def slate_forecast(match_id: str):
    row = get_match(match_id)
    if not row:
        raise HTTPException(404, {"error": "match_not_found"})
    with session() as conn:
        report = forecast(
            conn,
            row["p1"],
            row["p2"],
            sets_1=int(row["sets"].split("-")[0]),
            sets_2=int(row["sets"].split("-")[1]),
            games_1=int(row["games"].split("-")[0]),
            games_2=int(row["games"].split("-")[1]),
            points_1=row["points"].split("-")[0],
            points_2=row["points"].split("-")[1],
            server=row["server"],
            surface=row.get("surface"),
            set1=row.get("set1"),
        )
    if report.get("error"):
        raise HTTPException(404, report)
    return {"board": row, "forecast": report}


@app.get("/api/forecast")
def forecast_api(
    p1: str,
    p2: str,
    sets: str = "0-0",
    games: str = "0-0",
    points: str = "0-0",
    server: str = "p1",
    surface: str | None = None,
    set1: str | None = None,
    line_1: float | None = None,
    line_2: float | None = None,
):
    def split(text: str):
        a, b = text.replace(":", "-").split("-", 1)
        return a, b

    s1, s2 = (int(x) for x in split(sets))
    g1, g2 = (int(x) for x in split(games))
    pt1, pt2 = split(points)
    with session() as conn:
        report = forecast(
            conn,
            p1,
            p2,
            sets_1=s1,
            sets_2=s2,
            games_1=g1,
            games_2=g2,
            points_1=pt1,
            points_2=pt2,
            server=server,
            surface=surface,
            set1=set1,
            line_1=line_1,
            line_2=line_2,
        )
    if report.get("error"):
        raise HTTPException(404, report)
    return report


@app.get("/api/inplay")
def inplay_api(
    p1: str,
    p2: str,
    tour: str = "CH",
    surface: str | None = None,
    sets: str = "0-0",
    games: str = "0-0",
    points: str = "0-0",
    server: int = 1,
    tb: bool = False,
    serve_1: str = "0/0",
    serve_2: str = "0/0",
    prev_set: str | None = None,
    streak: int = 0,
    mto_1: float | None = None,
    mto_2: float | None = None,
    rule: str = "ball_served",
    price_1: float | None = None,
    price_2: float | None = None,
    log: bool = False,
    tier: str | None = None,
    event: str | None = None,
):
    """Dynamic in-play forecast: prior -> live-updated Markov -> GBM -> retirement overlay -> market view.

    Read-only research output; nothing here places a bet or talks to a sportsbook.
    """
    from research.inplay import inplay_bundle

    def pair(text: str, sep: str = "-") -> tuple[str, str]:
        parts = text.replace(":", sep).split(sep, 1)
        if len(parts) != 2:
            raise HTTPException(422, {"error": f"expected 'a{sep}b', got {text!r}"})
        return parts[0].strip(), parts[1].strip()

    try:
        s = tuple(int(x) for x in pair(sets))
        g = tuple(int(x) for x in pair(games))
        pt = pair(points)
        sv1 = tuple(int(x) for x in pair(serve_1, "/"))
        sv2 = tuple(int(x) for x in pair(serve_2, "/"))
        ps = tuple(int(x) for x in pair(prev_set)) if prev_set else None
    except ValueError as exc:
        raise HTTPException(422, {"error": str(exc)})
    if server not in (1, 2):
        raise HTTPException(422, {"error": "server must be 1 or 2"})
    with session() as conn:
        report = inplay_bundle(
            conn,
            p1,
            p2,
            tour=tour.upper(),
            surface=surface,
            sets=s,
            games=g,
            points=pt,
            server=server,
            in_tb=tb,
            serve_1=sv1,
            serve_2=sv2,
            prev_set=ps,
            streak=streak,
            mto_1=mto_1,
            mto_2=mto_2,
            rule=rule,
            price_1=price_1,
            price_2=price_2,
            log=log,
            tier=tier,
            event=event,
        )
    if report.get("error"):
        raise HTTPException(404, report)
    return report


@app.get("/api/tiers")
def tiers_api(since: str = "2011-01-01"):
    from research.tiers import tier_report

    with session() as conn:
        return tier_report(conn, since=since)


@app.get("/api/inplay/model")
def inplay_model_metrics(group: str = "CH_SMALL"):
    import json

    from research.inplay import MODEL_DIR

    path = MODEL_DIR / f"inplay_{group}_metrics.json"
    if not path.exists():
        return {"trained": False, "hint": f"Run python -m research train-inplay --group {group}"}
    return {"trained": True, **json.loads(path.read_text())}


@app.get("/api/paper")
def paper_api():
    from research.market import paper_summary

    with session() as conn:
        return paper_summary(conn)


@app.get("/api/paper/rows")
def paper_rows_api(limit: int = 50):
    from research.market import paper_rows

    with session() as conn:
        return paper_rows(conn, limit=limit)


class SettleIn(BaseModel):
    winner: int
    closing_price: float | None = None


@app.post("/api/paper/{log_id}/settle")
def paper_settle_api(log_id: int, body: SettleIn):
    from research.market import settle_paper

    with session() as conn:
        try:
            return settle_paper(conn, log_id, body.winner, body.closing_price)
        except KeyError as exc:
            raise HTTPException(404, {"error": str(exc)})
        except ValueError as exc:
            raise HTTPException(422, {"error": str(exc)})


@app.get("/api/model")
def model_metrics():
    if not METRICS_PATH.exists():
        return {"trained": False, "hint": "Run python -m research train"}
    import json

    return {"trained": True, **json.loads(METRICS_PATH.read_text())}


@app.get("/api/watchlist")
def watchlist():
    with session() as conn:
        return list_watchlist(conn)


@app.post("/api/watchlist")
def watchlist_add(item: WatchIn):
    with session() as conn:
        return add_watch(conn, item.model_dump())


@app.delete("/api/watchlist/{watch_id}")
def watchlist_delete(watch_id: int):
    with session() as conn:
        delete_watch(conn, watch_id)
    return {"ok": True}
