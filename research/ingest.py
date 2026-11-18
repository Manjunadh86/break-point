from __future__ import annotations

import time
from datetime import datetime
from io import BytesIO
from pathlib import Path

import httpx
import pandas as pd

from research.analyze import parse_completion, rebuild_ratings
from research.config import (
    DEFAULT_YEARS,
    PBP_BASE,
    PBP_FILES,
    RAW_DIR,
    SACKMANN_BASE,
    SACKMANN_WTA_BASE,
    TENNIS_DATA_XLSX,
)
from research.db import ensure_dirs, init_db

# source tag -> (base url, filename pattern)
SACKMANN_SOURCES = {
    "atp": (SACKMANN_BASE, "atp_matches_{year}.csv"),
    "qual_chall": (SACKMANN_BASE, "atp_matches_qual_chall_{year}.csv"),
    "wta": (SACKMANN_WTA_BASE, "wta_matches_{year}.csv"),
    "qual_itf": (SACKMANN_WTA_BASE, "wta_matches_qual_itf_{year}.csv"),
}
DEFAULT_SOURCES = ("atp", "qual_chall", "wta", "qual_itf")
WOMEN_SOURCES = ("wta", "qual_itf")
# Sackmann ATP ids (100xxx-21xxxx) and WTA ids (200xxx-27xxxx) overlap. Women's ids are
# offset so one player_id is one human across the whole database.
WOMEN_ID_OFFSET = 10_000_000

UA = {"User-Agent": "BreakPointResearch/0.1 (local tennis research; +https://github.com)"}


def _get(url: str, dest: Path | None = None, retries: int = 3) -> bytes:
    ensure_dirs()
    if dest and dest.exists() and dest.stat().st_size > 2000:
        return dest.read_bytes()
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            with httpx.Client(timeout=60.0, follow_redirects=True, headers=UA) as client:
                resp = client.get(url)
                if resp.status_code in {429, 503}:
                    wait = min(int(resp.headers.get("Retry-After", 6 + attempt * 4)), 12)
                    print(f"  {resp.status_code} from {url} — retry in {wait}s", flush=True)
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                data = resp.content
                if dest:
                    dest.write_bytes(data)
                return data
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            time.sleep(2 + attempt * 2)
    raise RuntimeError(f"Failed to download {url}: {last_err}")


def _to_date(value) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(int(value)) if isinstance(value, (int, float)) and not isinstance(value, bool) else str(value)
    text = text.strip()
    if len(text) == 8 and text.isdigit():
        return f"{text[0:4]}-{text[4:6]}-{text[6:8]}"
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%Y/%m/%d", "%m/%d/%Y"):
        try:
            return datetime.strptime(text[:10], fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    try:
        return pd.to_datetime(value).strftime("%Y-%m-%d")
    except Exception:
        return None


def _num(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, str) and not value.strip():
        return None
    try:
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, str) and "." not in value:
            return int(value)
        number = float(value)
        return int(number) if number.is_integer() else number
    except (TypeError, ValueError):
        return None


def _text(value) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    return text or None


MATCH_COLS = [
    "tourney_id", "tourney_name", "surface", "draw_size", "tourney_level", "tourney_date",
    "match_num", "round", "best_of", "score", "minutes",
    "winner_id", "winner_name", "winner_hand", "winner_ht", "winner_ioc", "winner_age",
    "winner_seed", "winner_entry", "winner_rank", "winner_rank_points",
    "loser_id", "loser_name", "loser_hand", "loser_ht", "loser_ioc", "loser_age",
    "loser_seed", "loser_entry", "loser_rank", "loser_rank_points",
    "w_ace", "w_df", "w_svpt", "w_1stIn", "w_1stWon", "w_2ndWon", "w_SvGms", "w_bpSaved", "w_bpFaced",
    "l_ace", "l_df", "l_svpt", "l_1stIn", "l_1stWon", "l_2ndWon", "l_SvGms", "l_bpSaved", "l_bpFaced",
    "completion", "source",
]


def ingest_sackmann(
    years: tuple[int, ...] = DEFAULT_YEARS,
    sources: tuple[str, ...] = DEFAULT_SOURCES,
    rebuild: bool = True,
) -> dict:
    conn = init_db()
    inserted = 0
    files = 0
    for year in years:
        for kind in sources:
            base, pattern = SACKMANN_SOURCES[kind]
            fname = pattern.format(year=year)
            url = f"{base}/{fname}"
            dest = RAW_DIR / fname
            print(f"Downloading {fname} …", flush=True)
            try:
                raw = _get(url, dest)
            except Exception as exc:  # noqa: BLE001
                print(f"  skip {fname}: {exc}")
                continue
            files += 1
            df = pd.read_csv(BytesIO(raw), low_memory=False)
            offset = WOMEN_ID_OFFSET if kind in WOMEN_SOURCES else 0

            def _pid(value):
                pid = _num(value)
                return pid + offset if isinstance(pid, int) else pid

            rows = []
            for rec in df.to_dict("records"):
                rows.append(
                    (
                        _text(rec.get("tourney_id")),
                        _text(rec.get("tourney_name")),
                        _text(rec.get("surface")),
                        _num(rec.get("draw_size")),
                        _text(rec.get("tourney_level")),
                        _to_date(rec.get("tourney_date")),
                        _num(rec.get("match_num")),
                        _text(rec.get("round")),
                        _num(rec.get("best_of")),
                        _text(rec.get("score")),
                        _num(rec.get("minutes")),
                        _pid(rec.get("winner_id")),
                        _text(rec.get("winner_name")),
                        _text(rec.get("winner_hand")),
                        _num(rec.get("winner_ht")),
                        _text(rec.get("winner_ioc")),
                        _num(rec.get("winner_age")),
                        _text(rec.get("winner_seed")),
                        _text(rec.get("winner_entry")),
                        _num(rec.get("winner_rank")),
                        _num(rec.get("winner_rank_points")),
                        _pid(rec.get("loser_id")),
                        _text(rec.get("loser_name")),
                        _text(rec.get("loser_hand")),
                        _num(rec.get("loser_ht")),
                        _text(rec.get("loser_ioc")),
                        _num(rec.get("loser_age")),
                        _text(rec.get("loser_seed")),
                        _text(rec.get("loser_entry")),
                        _num(rec.get("loser_rank")),
                        _num(rec.get("loser_rank_points")),
                        _num(rec.get("w_ace")),
                        _num(rec.get("w_df")),
                        _num(rec.get("w_svpt")),
                        _num(rec.get("w_1stIn")),
                        _num(rec.get("w_1stWon")),
                        _num(rec.get("w_2ndWon")),
                        _num(rec.get("w_SvGms")),
                        _num(rec.get("w_bpSaved")),
                        _num(rec.get("w_bpFaced")),
                        _num(rec.get("l_ace")),
                        _num(rec.get("l_df")),
                        _num(rec.get("l_svpt")),
                        _num(rec.get("l_1stIn")),
                        _num(rec.get("l_1stWon")),
                        _num(rec.get("l_2ndWon")),
                        _num(rec.get("l_SvGms")),
                        _num(rec.get("l_bpSaved")),
                        _num(rec.get("l_bpFaced")),
                        parse_completion(_text(rec.get("score"))),
                        kind,
                    )
                )
            before = conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
            conn.executemany(
                f"INSERT OR IGNORE INTO matches ({', '.join(MATCH_COLS)}) VALUES ({', '.join('?' * len(MATCH_COLS))})",
                rows,
            )
            conn.commit()
            after = conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
            inserted += after - before
            print(f"  {fname}: {after - before} new matches")

    conn.execute(
        """
        INSERT OR REPLACE INTO players (player_id, name, hand, ioc, ht)
        SELECT winner_id, winner_name, winner_hand, winner_ioc, winner_ht FROM matches
        WHERE winner_id IS NOT NULL
        """
    )
    conn.execute(
        """
        INSERT OR IGNORE INTO players (player_id, name, hand, ioc, ht)
        SELECT loser_id, loser_name, loser_hand, loser_ioc, loser_ht FROM matches
        WHERE loser_id IS NOT NULL
        """
    )
    if rebuild:
        print("Rebuilding Elo …")
        rebuild_ratings(conn)
    conn.execute(
        "INSERT OR REPLACE INTO meta(key, value) VALUES ('last_ingest', ?)",
        (datetime.utcnow().isoformat(timespec="seconds") + "Z",),
    )
    conn.commit()
    n = conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
    chall = conn.execute("SELECT COUNT(*) FROM matches WHERE tourney_level='C'").fetchone()[0]
    wta = conn.execute("SELECT COUNT(*) FROM matches WHERE source IN ('wta','qual_itf')").fetchone()[0]
    conn.close()
    return {"files": files, "inserted": inserted, "matches": n, "challenger": chall, "women": wta}


# --------------------------------------------------------------------------------------
# Point-by-point
# --------------------------------------------------------------------------------------

PBP_COLS = [
    "date", "tny_name", "tour", "draw", "p1", "p2", "winner", "pbp", "score", "adf_flag",
    "best_of", "n_points", "valid", "parse_error", "source_file",
]


def _pbp_date(text) -> str | None:
    try:
        return datetime.strptime(str(text).strip(), "%d %b %y").strftime("%Y-%m-%d")
    except (ValueError, TypeError):
        return _to_date(text)


def ingest_pbp(files: tuple[str, ...] = PBP_FILES, link: bool = True) -> dict:
    """Download Sackmann point-by-point files, validate every string, store, and link to matches."""
    from research.pbp import parse_match

    conn = init_db()
    pbp_dir = RAW_DIR / "pbp"
    pbp_dir.mkdir(parents=True, exist_ok=True)
    stats: dict[str, dict] = {}
    total_new = 0
    for tag in files:
        fname = f"pbp_matches_{tag}.csv"
        dest = pbp_dir / fname
        print(f"Downloading {fname} …", flush=True)
        try:
            raw = _get(f"{PBP_BASE}/{fname}", dest)
        except Exception as exc:  # noqa: BLE001
            print(f"  skip {fname}: {exc}")
            continue
        df = pd.read_csv(BytesIO(raw), low_memory=False)
        rows = []
        valid = 0
        for rec in df.to_dict("records"):
            pbp = _text(rec.get("pbp"))
            p1, p2 = _text(rec.get("server1")), _text(rec.get("server2"))
            if not pbp or not p1 or not p2:
                continue
            winner = _num(rec.get("winner"))
            parsed = parse_match(pbp, 1, _text(rec.get("score")), winner)
            valid += int(parsed.valid)
            rows.append(
                (
                    _pbp_date(rec.get("date")),
                    _text(rec.get("tny_name")),
                    _text(rec.get("tour")),
                    _text(rec.get("draw")),
                    p1,
                    p2,
                    winner,
                    pbp,
                    _text(rec.get("score")),
                    _num(rec.get("adf_flag")),
                    parsed.best_of,
                    parsed.n_points,
                    int(parsed.valid),
                    parsed.error,
                    tag,
                )
            )
        before = conn.execute("SELECT COUNT(*) FROM pbp_matches").fetchone()[0]
        conn.executemany(
            f"INSERT OR IGNORE INTO pbp_matches ({', '.join(PBP_COLS)}) VALUES ({', '.join('?' * len(PBP_COLS))})",
            rows,
        )
        conn.commit()
        after = conn.execute("SELECT COUNT(*) FROM pbp_matches").fetchone()[0]
        total_new += after - before
        stats[tag] = {"rows": len(rows), "valid": valid, "new": after - before}
        print(f"  {fname}: {len(rows)} matches, {valid} valid, {after - before} new")

    linked = link_pbp_to_matches(conn) if link else None
    if link:
        from research.tiers import build_event_tiers

        tiers = build_event_tiers(conn)
        print(f"  event tiers rebuilt; pbp matches per segment: {tiers['pbp_matches']}")
    conn.execute(
        "INSERT OR REPLACE INTO meta(key, value) VALUES ('last_pbp_ingest', ?)",
        (datetime.utcnow().isoformat(timespec="seconds") + "Z",),
    )
    conn.commit()
    n = conn.execute("SELECT COUNT(*) FROM pbp_matches WHERE valid=1").fetchone()[0]
    conn.close()
    return {"files": stats, "new": total_new, "valid_total": n, "linked": linked}


def link_pbp_to_matches(conn) -> dict:
    """Attach each pbp match to its Sackmann result row by player names and date proximity.

    The link gives us pre-match Elo, surface, tourney level and the ability to compute
    fatigue carry-over from the player's previous match.
    """
    from collections import defaultdict
    from datetime import date as _date

    from research.names import fold

    index: dict[tuple[str, str], list] = defaultdict(list)
    rows = conn.execute(
        """
        SELECT id, tourney_date, winner_id, loser_id, winner_name, loser_name,
               winner_elo, loser_elo, winner_surf_elo, loser_surf_elo, surface, tourney_level
        FROM matches WHERE tourney_date IS NOT NULL
        """
    ).fetchall()
    for r in rows:
        key = (fold(r["winner_name"] or ""), fold(r["loser_name"] or ""))
        index[key].append(r)

    def _d(text: str | None):
        try:
            return _date.fromisoformat(text[:10]) if text else None
        except ValueError:
            return None

    pbp_rows = conn.execute(
        "SELECT id, date, p1, p2, winner FROM pbp_matches WHERE valid=1 AND match_id IS NULL"
    ).fetchall()
    updates = []
    linked = 0
    for r in pbp_rows:
        d = _d(r["date"])
        if d is None:
            continue
        w_name, l_name = (r["p1"], r["p2"]) if r["winner"] == 1 else (r["p2"], r["p1"])
        cands = index.get((fold(w_name), fold(l_name)), [])
        best = None
        best_gap = 99
        for c in cands:
            cd = _d(c["tourney_date"])
            if cd is None:
                continue
            gap = abs((d - cd).days)
            if gap <= 14 and gap < best_gap:
                best, best_gap = c, gap
        if best is None:
            continue
        linked += 1
        if r["winner"] == 1:
            p1_id, p2_id = best["winner_id"], best["loser_id"]
            p1_elo, p2_elo = best["winner_elo"], best["loser_elo"]
            p1_s, p2_s = best["winner_surf_elo"], best["loser_surf_elo"]
        else:
            p1_id, p2_id = best["loser_id"], best["winner_id"]
            p1_elo, p2_elo = best["loser_elo"], best["winner_elo"]
            p1_s, p2_s = best["loser_surf_elo"], best["winner_surf_elo"]
        updates.append(
            (best["id"], p1_id, p2_id, p1_elo, p2_elo, p1_s, p2_s, best["surface"], best["tourney_level"], r["id"])
        )
    conn.executemany(
        """
        UPDATE pbp_matches
        SET match_id=?, p1_id=?, p2_id=?, p1_elo=?, p2_elo=?, p1_surf_elo=?, p2_surf_elo=?,
            surface=?, tourney_level=?
        WHERE id=?
        """,
        updates,
    )
    conn.commit()
    total = conn.execute("SELECT COUNT(*) FROM pbp_matches WHERE valid=1").fetchone()[0]
    with_link = conn.execute("SELECT COUNT(*) FROM pbp_matches WHERE valid=1 AND match_id IS NOT NULL").fetchone()[0]
    print(f"  linked {linked} new; {with_link}/{total} valid pbp matches now carry a Sackmann link")
    return {"newly_linked": linked, "linked": with_link, "valid": total}


def ingest_tennis_data(years: tuple[int, ...] = DEFAULT_YEARS) -> dict:
    conn = init_db()
    inserted = 0
    for year in years:
        url = TENNIS_DATA_XLSX.format(year=year)
        dest = RAW_DIR / f"tennis_data_{year}.xlsx"
        print(f"Downloading tennis-data {year} …", flush=True)
        try:
            raw = _get(url, dest)
        except Exception as exc:  # noqa: BLE001
            print(f"  skip {year}: {exc}")
            continue
        try:
            df = pd.read_excel(BytesIO(raw), engine="openpyxl")
        except Exception as exc:  # noqa: BLE001
            print(f"  could not parse {year}: {exc}")
            continue
        rows = []
        for rec in df.to_dict("records"):
            rows.append(
                (
                    _to_date(rec.get("Date")),
                    _text(rec.get("Tournament")),
                    _text(rec.get("Series")),
                    _text(rec.get("Surface")),
                    _text(rec.get("Round")),
                    _text(rec.get("Winner")),
                    _text(rec.get("Loser")),
                    _num(rec.get("WRank")),
                    _num(rec.get("LRank")),
                    _text(rec.get("Comment")),
                    _num(rec.get("B365W")),
                    _num(rec.get("B365L")),
                    _num(rec.get("PSW")),
                    _num(rec.get("PSL")),
                    _num(rec.get("MaxW")),
                    _num(rec.get("MaxL")),
                    _num(rec.get("AvgW")),
                    _num(rec.get("AvgL")),
                    None,
                )
            )
        before = conn.execute("SELECT COUNT(*) FROM closing_odds").fetchone()[0]
        conn.executemany(
            """
            INSERT OR IGNORE INTO closing_odds (
                match_date, tournament, series, surface, round,
                winner_name, loser_name, winner_rank, loser_rank, comment,
                b365_w, b365_l, ps_w, ps_l, max_w, max_l, avg_w, avg_l, match_id
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            rows,
        )
        conn.commit()
        after = conn.execute("SELECT COUNT(*) FROM closing_odds").fetchone()[0]
        inserted += after - before
        print(f"  {year}: {after - before} closing lines")

    conn.commit()
    n = conn.execute("SELECT COUNT(*) FROM closing_odds").fetchone()[0]
    conn.close()
    return {"inserted": inserted, "closing_odds": n}


def snapshot_odds() -> dict:
    from research.config import ODDS_API_KEY, ODDS_REGIONS, ODDS_SPORT_KEYS

    if not ODDS_API_KEY:
        raise RuntimeError("Set ODDS_API_KEY in .env to snapshot live tennis lines.")

    from research.analyze import implied_from_decimal

    conn = init_db()
    stored = 0
    now = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    with httpx.Client(timeout=40.0, headers=UA) as client:
        for sport in ODDS_SPORT_KEYS:
            url = f"https://api.the-odds-api.com/v4/sports/{sport}/odds"
            resp = client.get(
                url,
                params={
                    "apiKey": ODDS_API_KEY,
                    "regions": ODDS_REGIONS,
                    "markets": "h2h",
                    "oddsFormat": "decimal",
                },
            )
            if resp.status_code == 404:
                print(f"  {sport}: not in season")
                continue
            resp.raise_for_status()
            events = resp.json()
            print(f"  {sport}: {len(events)} events")
            for event in events:
                home = event.get("home_team")
                away = event.get("away_team")
                for book in event.get("bookmakers") or []:
                    for market in book.get("markets") or []:
                        if market.get("key") != "h2h":
                            continue
                        for outcome in market.get("outcomes") or []:
                            price = outcome.get("price")
                            conn.execute(
                                """
                                INSERT INTO odds_snapshots (
                                    snapped_at, sport_key, event_id, commence_time, tournament,
                                    player_a, player_b, bookmaker, market, player_name, price, implied
                                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                                """,
                                (
                                    now,
                                    sport,
                                    event.get("id"),
                                    event.get("commence_time"),
                                    event.get("sport_title"),
                                    home,
                                    away,
                                    book.get("title") or book.get("key"),
                                    "h2h",
                                    outcome.get("name"),
                                    price,
                                    implied_from_decimal(price),
                                ),
                            )
                            stored += 1
    conn.commit()
    conn.close()
    return {"snapped_at": now, "rows": stored}
