"""Public live tennis slate. No sportsbook scrape.

Primary source is LiveScore's public scoreboard JSON (ATP all levels, ATP Challenger, WTA,
WTA 125). It carries sets, games, who is serving and start time; it does not carry the point
score inside the game or serve-point counts, so those stay manual in the Point engine.
ESPN's scoreboard (ATP/WTA main tour only) is kept as a fallback if LiveScore is unreachable.
ITF World Tour events have no reachable public feed from here and are not listed.
"""

from __future__ import annotations

import re
import sqlite3
from datetime import datetime, timedelta, timezone
from time import time

import httpx

LIVESCORE_LIVE = "https://prod-public-api.livescore.com/v1/api/app/live/tennis/5.30?MD=1"
LIVESCORE_DATE = "https://prod-public-api.livescore.com/v1/api/app/date/tennis/{ymd}/5.30?MD=1"
LIVESCORE_EVENT = "https://prod-public-api.livescore.com/v1/api/app/scoreboard/tennis/{eid}"
ESPN_SCOREBOARDS = (
    "https://site.web.api.espn.com/apis/site/v2/sports/tennis/atp/scoreboard",
    "https://site.web.api.espn.com/apis/site/v2/sports/tennis/wta/scoreboard",
)
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.9",
}
POINT_WORDS = {"0", "15", "30", "40", "AD", "A"}

# LiveScore event types: 10 men's singles, 11 men's doubles, 15 women's singles, 16 women's doubles.
MEN_SINGLES, WOMEN_SINGLES = 10, 15
PRE_STATES = {"NS"}
POST_STATES = {"FT", "Canc", "Postp", "Ret", "WO", "Aband", "Abd", "Awarded"}

_cache: dict[str, tuple[float, object]] = {}
_surface_cache: dict[tuple[str, str], tuple[str | None, str | None]] = {}


def _cached(key: str, ttl: float, loader):
    now = time()
    hit = _cache.get(key)
    if hit and now - hit[0] < ttl:
        return hit[1]
    value = loader()
    _cache[key] = (now, value)
    return value


def _get(url: str, referer: str) -> dict:
    with httpx.Client(timeout=20.0, headers={**HEADERS, "Referer": referer}, follow_redirects=True) as client:
        resp = client.get(url)
        resp.raise_for_status()
        return resp.json()


# ----------------------------------------------------------------------------- LiveScore


def _ls_datetime(raw) -> str | None:
    """Esd is yyyymmddHHMMSS in UTC."""
    text = str(raw or "")
    if len(text) < 12:
        return None
    try:
        dt = datetime.strptime(text[:14].ljust(14, "0"), "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return None
    return dt.isoformat(timespec="seconds")


def _ls_level(stage: dict) -> tuple[str, str, str, str]:
    """-> (tour for the engine, sex, display level, tier hint)."""
    cat = (stage.get("Cnm") or stage.get("Csnm") or "").strip()
    low = cat.lower()
    if "challenger" in low:
        if low.startswith("wta"):
            return "ITF", "women", "WTA 125", "big"     # WTA 125 sits in the ITF_BIG segment
        return "CH", "men", "ATP Challenger", ""
    if low.startswith("wta"):
        return "WTA", "women", cat, "main"
    if low.startswith("atp") or "grand slam" in low or "davis" in low or "laver" in low or "united cup" in low:
        return "ATP", "men", cat, "main"
    if "itf" in low:
        sex = "women" if "women" in low or low.startswith("w") else "men"
        return ("ITF" if sex == "women" else "CH"), sex, cat, "small"
    return "ATP", "men", cat or "Tennis", "main"


def _ls_event_name(stage: dict) -> tuple[str, bool]:
    """'Santa Cruz 2, Bolivia' -> ('Santa Cruz', False); 'Almaty Open: Qualification' -> ('Almaty', True)."""
    raw = (stage.get("Snm") or stage.get("Sdn") or "").strip()
    qual = "qualif" in raw.lower()
    name = raw.split(":")[0]
    paren = re.search(r"\(([^)]+)\)", name)
    if paren:
        name = paren.group(1)
    name = name.split(",")[0]
    name = re.sub(r"\b(open|masters|rolex|cup|international|classic|championships?|tennis)\b", " ", name, flags=re.I)
    name = re.sub(r"\b\d+\b", " ", name)
    name = re.sub(r"\s+", " ", name).strip(" -")
    return (name or raw), qual


def _ls_scores(ev: dict) -> dict:
    sets = (int(ev.get("Tr1") or 0), int(ev.get("Tr2") or 0))
    per_set = []
    for i in range(1, 6):
        a, b = ev.get(f"Tr1S{i}"), ev.get(f"Tr2S{i}")
        if a is None and b is None:
            break
        per_set.append((int(a or 0), int(b or 0)))
    eps = str(ev.get("Eps") or "")
    live_set = int(eps[1:]) if re.fullmatch(r"S\d", eps) else None
    games = (0, 0)
    completed: list[tuple[int, int]] = []
    for i, (a, b) in enumerate(per_set, start=1):
        if live_set is not None and i == live_set:
            games = (a, b)
        elif i <= sets[0] + sets[1]:
            completed.append((a, b))
        else:
            games = (a, b)
    set1 = f"{completed[0][0]}-{completed[0][1]}" if completed else None
    return {"sets": sets, "games": games, "set1": set1, "per_set": per_set, "live_set": live_set}


def _ls_normalize(ev: dict, stage: dict) -> dict | None:
    et = ev.get("Et")
    if et not in (MEN_SINGLES, WOMEN_SINGLES):
        return None
    t1, t2 = ev.get("T1") or [], ev.get("T2") or []
    if not t1 or not t2:
        return None
    p1, p2 = (t1[0].get("Nm") or "").strip(), (t2[0].get("Nm") or "").strip()
    if not p1 or not p2 or "/" in p1 or "/" in p2:
        return None
    tour, sex, level, tier = _ls_level(stage)
    event, qual = _ls_event_name(stage)
    eps = str(ev.get("Eps") or "NS").strip(". ")
    if eps in PRE_STATES:
        state = "pre"
    elif eps in POST_STATES or any(eps.startswith(x) for x in ("FT", "Ret", "Canc", "Postp", "WO", "Aband")):
        state = "post"
    else:
        state = "in"
    sc = _ls_scores(ev)
    srv = ev.get("Esrv")
    server = "p1" if srv == 1 else "p2" if srv == 2 else None
    return {
        "id": str(ev.get("Eid")),
        "source": "livescore",
        "league": "WTA" if sex == "women" else "ATP",
        "tour": tour,
        "sex": sex,
        "level": level,
        "tier_hint": tier,
        "tournament": (stage.get("Snm") or "").strip(),
        "event": event,
        "qualifying": qual,
        "round": None,
        "court": None,
        "p1": p1,
        "p2": p2,
        "p1_country": (t1[0].get("CoId") or None),
        "p2_country": (t2[0].get("CoId") or None),
        "sets": f"{sc['sets'][0]}-{sc['sets'][1]}",
        "games": f"{sc['games'][0]}-{sc['games'][1]}",
        "points": "0-0",
        "points_known": False,
        "set1": sc["set1"],
        "per_set": [f"{a}-{b}" for a, b in sc["per_set"]],
        "tiebreak": sc["games"] == (6, 6) and state == "in",
        "server": server or "p1",
        "server_known": server is not None,
        "surface": None,
        "surface_source": None,
        "state": state,
        "status": eps,
        "start": _ls_datetime(ev.get("Esd")),
        "challenger": tour in {"CH", "ITF"},
        "live": state == "in",
    }


def _ls_rows(payload: dict) -> list[dict]:
    rows = []
    for stage in payload.get("Stages") or []:
        for ev in stage.get("Events") or []:
            row = _ls_normalize(ev, stage)
            if row:
                rows.append(row)
    return rows


def _slate_livescore() -> list[dict]:
    today = datetime.now(timezone.utc)
    rows: dict[str, dict] = {}
    # live first so a live row wins over its schedule copy
    for url in (
        LIVESCORE_LIVE,
        LIVESCORE_DATE.format(ymd=today.strftime("%Y%m%d")),
        LIVESCORE_DATE.format(ymd=(today + timedelta(days=1)).strftime("%Y%m%d")),
    ):
        try:
            data = _get(url, "https://www.livescore.com/en/tennis/")
        except Exception:
            continue
        for row in _ls_rows(data):
            if row["id"] not in rows or (row["live"] and not rows[row["id"]]["live"]):
                rows[row["id"]] = row
    return list(rows.values())


def _ls_event(eid: str) -> dict | None:
    try:
        ev = _get(LIVESCORE_EVENT.format(eid=eid), "https://www.livescore.com/en/tennis/")
    except Exception:
        return None
    stage = ev.get("Stg") or {}
    return _ls_normalize(ev, stage)


# ----------------------------------------------------------------------------- ESPN fallback


def _name(comp: dict) -> str:
    athlete = comp.get("athlete") or {}
    return (
        athlete.get("displayName")
        or athlete.get("fullName")
        or comp.get("displayName")
        or comp.get("name")
        or ""
    ).strip()


def _parse_score_blob(text: str | None) -> dict:
    """Best-effort parse of '6-4 3-2' or '6-4 3-2 30-15'."""
    out = {"sets": (0, 0), "games": (0, 0), "points": None, "set1": None}
    if not text:
        return out
    chunks = re.findall(r"(\d+|AD|A)\s*[-–]\s*(\d+|AD|A)", text.upper())
    if not chunks:
        return out
    completed = []
    current_games = None
    points = None
    for a, b in chunks:
        if a in POINT_WORDS and b in POINT_WORDS:
            points = (a if a != "A" else "AD", b if b != "A" else "AD")
            continue
        ga, gb = int(a), int(b)
        finished = (max(ga, gb) >= 6 and abs(ga - gb) >= 2) or max(ga, gb) == 7
        if finished:
            completed.append((ga, gb))
        else:
            current_games = (ga, gb)
    out["sets"] = (sum(1 for a, b in completed if a > b), sum(1 for a, b in completed if b > a))
    out["games"] = current_games or (0, 0)
    out["points"] = points
    if completed:
        out["set1"] = f"{completed[0][0]}-{completed[0][1]}"
    return out


def _from_linescores(home: dict, away: dict, period: int | None) -> dict:
    h_ls = home.get("linescores") or []
    a_ls = away.get("linescores") or []
    n = max(len(h_ls), len(a_ls), period or 0)
    completed = []
    games = (0, 0)
    for i in range(n):
        hv = int((h_ls[i] or {}).get("value") or (h_ls[i] or {}).get("score") or 0) if i < len(h_ls) else 0
        av = int((a_ls[i] or {}).get("value") or (a_ls[i] or {}).get("score") or 0) if i < len(a_ls) else 0
        hw = bool((h_ls[i] or {}).get("winner")) if i < len(h_ls) else False
        aw = bool((a_ls[i] or {}).get("winner")) if i < len(a_ls) else False
        live_set = period is not None and i == period - 1
        finished = hw or aw or (max(hv, av) >= 6 and abs(hv - av) >= 2) or max(hv, av) == 7
        if live_set and not (hw or aw):
            games = (hv, av)
        elif finished:
            completed.append((hv, av))
        else:
            games = (hv, av)
    s1 = sum(1 for a, b in completed if a > b)
    s2 = sum(1 for a, b in completed if b > a)
    set1 = f"{completed[0][0]}-{completed[0][1]}" if completed else None
    return {"sets": (s1, s2), "games": games, "set1": set1, "points": None}


def _espn_normalize(item: dict, tournament: str, league: str) -> dict | None:
    comps = sorted(item.get("competitors") or [], key=lambda c: c.get("order") or 0)
    if len(comps) < 2:
        return None
    a, b = comps[0], comps[1]
    names = [_name(a), _name(b)]
    slug = ((item.get("type") or {}).get("slug") or "").lower()
    if not names[0] or not names[1] or "double" in slug or any("/" in n for n in names):
        return None
    status = (item.get("status") or {}).get("type") or {}
    state = status.get("state") or "pre"
    parsed = _from_linescores(a, b, (item.get("status") or {}).get("period"))
    pts = None
    for cand in (a.get("score"), b.get("score"), status.get("detail"), status.get("shortDetail")):
        tok = _parse_score_blob(str(cand) if cand is not None else None)
        if tok.get("points"):
            pts = tok["points"]
    server = None
    for raw in (item.get("situation") or {}, status, item):
        if not isinstance(raw, dict):
            continue
        text = " ".join(str(raw.get(k) or "") for k in ("possession", "detail", "description")).lower()
        if "serv" in text:
            if names[0].split()[-1].lower() in text:
                server = "p1"
            elif names[1].split()[-1].lower() in text:
                server = "p2"
    chall = bool(re.search(r"challenger|\bch\b|wta\s*125|\b125\b|\bitf\b", tournament, re.I))
    sex = "women" if league == "WTA" else "men"
    tour = ("ITF" if sex == "women" else "CH") if chall else league
    event, qual = _ls_event_name({"Snm": tournament})
    return {
        "id": str(item.get("id") or item.get("uid")),
        "source": "espn",
        "league": league,
        "tour": tour,
        "sex": sex,
        "level": f"{league} ({'Challenger' if chall else 'main tour'})",
        "tier_hint": "" if chall else "main",
        "tournament": tournament,
        "event": event,
        "qualifying": qual,
        "round": (item.get("round") or {}).get("displayName"),
        "court": (item.get("venue") or {}).get("court"),
        "p1": names[0],
        "p2": names[1],
        "p1_country": None,
        "p2_country": None,
        "sets": f"{parsed['sets'][0]}-{parsed['sets'][1]}",
        "games": f"{parsed['games'][0]}-{parsed['games'][1]}",
        "points": f"{pts[0]}-{pts[1]}" if pts else "0-0",
        "points_known": pts is not None,
        "set1": parsed.get("set1"),
        "per_set": [],
        "tiebreak": False,
        "server": server or "p1",
        "server_known": server is not None,
        "surface": None,
        "surface_source": None,
        "state": state,
        "status": status.get("shortDetail") or status.get("description") or state,
        "start": item.get("startDate") or item.get("date"),
        "challenger": chall,
        "live": state == "in",
    }


def _slate_espn() -> list[dict]:
    rows, seen = [], set()
    for url, league in zip(ESPN_SCOREBOARDS, ("ATP", "WTA")):
        try:
            data = _get(url, "https://www.espn.com/tennis/scoreboard")
        except Exception:
            continue
        for event in data.get("events") or []:
            tournament = event.get("name") or event.get("shortName") or league
            for group in event.get("groupings") or []:
                for item in group.get("competitions") or []:
                    row = _espn_normalize(item, tournament, league)
                    if row and row["id"] not in seen:
                        seen.add(row["id"])
                        rows.append(row)
    return rows


# ----------------------------------------------------------------------------- public API


def _slate_uncached() -> list[dict]:
    rows = _slate_livescore()
    source = "livescore"
    if not rows:
        rows = _slate_espn()
        source = "espn"
    order = {"in": 0, "pre": 1, "post": 2}
    rows.sort(key=lambda r: (order.get(r["state"], 9), r.get("start") or "", r["tournament"], r["p1"]))
    _cache["slate_source"] = (time(), source)
    return rows


def live_slate(include_finals: bool = False) -> list[dict]:
    rows = _cached("slate", 8.0, _slate_uncached)
    if include_finals:
        return rows
    return [r for r in rows if r["state"] in {"in", "pre"}]


def get_match(match_id: str, fresh: bool = False) -> dict | None:
    """One match. `fresh=True` hits the per-event endpoint so a follower gets the newest score."""
    cached = next((r for r in _cached("slate", 8.0, _slate_uncached) if r["id"] == str(match_id)), None)
    if fresh and (cached is None or cached.get("source") == "livescore"):
        live = _ls_event(str(match_id))
        if live:
            if cached:
                live["surface"], live["surface_source"] = cached.get("surface"), cached.get("surface_source")
            return live
    if cached:
        return cached
    _cache.pop("slate", None)
    return next((r for r in _slate_uncached() if r["id"] == str(match_id)), None)


def guess_surface(conn: sqlite3.Connection, event: str, sex: str, tour: str) -> tuple[str | None, str | None]:
    """Most common surface for a Sackmann event whose name contains the city, last two seasons."""
    key = (event.lower(), sex)
    if key in _surface_cache:
        return _surface_cache[key]
    since = f"{datetime.now(timezone.utc).year - 2}-01-01"
    sources = ("wta", "qual_itf") if sex == "women" else ("atp", "qual_chall")
    rows = conn.execute(
        """
        SELECT tourney_name, surface, COUNT(*) AS n FROM matches
        WHERE tourney_name LIKE ? AND tourney_date >= ? AND surface IS NOT NULL AND surface <> ''
          AND source IN (?, ?)
        GROUP BY tourney_name, surface ORDER BY n DESC
        """,
        (f"%{event}%", since, *sources),
    ).fetchall()
    if not rows:
        rows = conn.execute(
            """
            SELECT tourney_name, surface, COUNT(*) AS n FROM matches
            WHERE tourney_name LIKE ? AND tourney_date >= ? AND surface IS NOT NULL AND surface <> ''
            GROUP BY tourney_name, surface ORDER BY n DESC
            """,
            (f"%{event}%", since),
        ).fetchall()
    pick = None
    if rows:
        want_ch = tour in {"CH", "ITF"}
        pref = [r for r in rows if (" CH" in r[0] or "125" in r[0] or r[0].startswith("W")) == want_ch] or rows
        pick = pref[0]
    out = (pick[1], f"event:{pick[0]}") if pick else (None, None)
    _surface_cache[key] = out
    return out


def enrich(conn: sqlite3.Connection, rows: list[dict]) -> list[dict]:
    for r in rows:
        if r.get("surface") is None and r.get("event"):
            r["surface"], r["surface_source"] = guess_surface(conn, r["event"], r["sex"], r["tour"])
    return rows


def slate_meta() -> dict:
    rows = _cached("slate", 8.0, _slate_uncached)
    source = (_cache.get("slate_source") or (0, "livescore"))[1]
    return {
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": (
            "LiveScore public scoreboard (ATP, Challenger, WTA, WTA 125). Not Stake."
            if source == "livescore" else "ESPN public scoreboard (ATP/WTA main tour). Not Stake."
        ),
        "provider": source,
        "live": sum(1 for r in rows if r["live"]),
        "upcoming": sum(1 for r in rows if r["state"] == "pre"),
        "note": (
            "The feed carries sets, games and who is serving. The point score inside the game and the "
            "serve-point counts are not in it: type those yourself. ITF World Tour events are not listed."
        ),
    }
