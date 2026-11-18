"""Event tiers and segments.

The in-play model is not one model. Men and women are separate, and inside each the
events are split by size / reputation, because the game played at a $10K ITF or a weak
Challenger is not the game played at a 125 with ten top-100 players in the draw.

Segments
    men    : ATP | CH_BIG | CH_MID | CH_SMALL
    women  : WTA | ITF_BIG | ITF_SMALL

Challengers carry no prize tier in the Sackmann files, so *field strength* is the proxy:
the median ATP ranking of the main-draw players at that event. Terciles are cut per
season (rankings and the Challenger calendar drift), so "big" always means the strongest
third of that year's Challengers. Women's ITF events carry the prize money directly in
`tourney_level` ($10K ... $100K); WTA 125s are grouped with the big ITFs.

Nothing in here can observe match fixing. What it can do is show, per segment, where
results deviate from what the engine expects: retirements, favourites collapsing after
winning the first set, heavy favourites losing more often than their price says. That is
the evidence; the interpretation is yours.
"""

from __future__ import annotations

import json
import sqlite3
from statistics import median

import numpy as np

from research.config import DATA_DIR

MODEL_DIR = DATA_DIR / "models"
CUTOFFS_PATH = MODEL_DIR / "tier_cutoffs.json"

MEN_SOURCES = ("atp", "qual_chall")
MEN_SEGMENTS = ("ATP", "CH_BIG", "CH_MID", "CH_SMALL")
WOMEN_SEGMENTS = ("WTA", "ITF_BIG", "ITF_SMALL")
ALL_SEGMENTS = MEN_SEGMENTS + WOMEN_SEGMENTS

TOUR_SEGMENTS = {
    "ATP": ("ATP",),
    "CH": ("CH_BIG", "CH_MID", "CH_SMALL"),
    "WTA": ("WTA",),
    "ITF": ("ITF_BIG", "ITF_SMALL"),
}

ITF_BIG_MIN_PRIZE = 35  # $35K and up count as the big ITF circuit; 10/15/25K are the small leagues


def sex_of_segment(segment: str) -> str:
    return "men" if segment in MEN_SEGMENTS else "women"


def segments_for_tour(tour: str) -> tuple[str, ...]:
    return TOUR_SEGMENTS.get(tour.upper(), ("CH_MID",))


def _is_itf_level(level: str | None) -> bool:
    lv = (level or "").upper()
    return lv.isdigit() or (lv.startswith("W") and lv[1:].isdigit())


def _prize(level: str | None) -> int | None:
    lv = (level or "").upper().lstrip("W")
    return int(lv) if lv.isdigit() else None


def segment_from_level(source: str, level: str | None, ch_tier: str | None = None) -> str:
    """Segment for a Sackmann row. `ch_tier` is 'big'/'mid'/'small' for men's Challengers."""
    lv = (level or "").upper()
    if source in MEN_SOURCES:
        if lv == "C":
            return f"CH_{(ch_tier or 'mid').upper()}"
        if _is_itf_level(lv):
            return "CH_SMALL"  # men's Futures/ITF rows (rare in these files) sit with the small leagues
        return "ATP"
    if _is_itf_level(lv):
        prize = _prize(lv) or 0
        return "ITF_BIG" if prize >= ITF_BIG_MIN_PRIZE else "ITF_SMALL"
    if lv == "C":
        return "ITF_BIG"  # WTA 125
    return "WTA"


# --------------------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------------------


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS event_tiers (
            tourney_id TEXT PRIMARY KEY,
            source TEXT,
            year INTEGER,
            level TEXT,
            field_rank REAL,      -- median ranking of main-draw players (men's Challengers)
            n_players INTEGER,
            tier TEXT,            -- big / mid / small (Challengers, ITF) or main
            segment TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_event_tiers_segment ON event_tiers(segment);
        """
    )
    cols = {r[1] for r in conn.execute("PRAGMA table_info(pbp_matches)")}
    if "segment" not in cols:
        conn.execute("ALTER TABLE pbp_matches ADD COLUMN segment TEXT")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_pbp_segment ON pbp_matches(segment)")


def build_event_tiers(conn: sqlite3.Connection) -> dict:
    """Tier every event, stamp `pbp_matches.segment`, save the yearly Challenger cutoffs."""
    _ensure_schema(conn)
    rows = conn.execute(
        """
        SELECT tourney_id, source, substr(tourney_date, 1, 4) AS year, tourney_level,
               winner_rank, loser_rank, round
        FROM matches
        WHERE tourney_id IS NOT NULL
        """
    ).fetchall()

    events: dict[str, dict] = {}
    for r in rows:
        ev = events.setdefault(
            r["tourney_id"],
            {"source": r["source"], "year": int(r["year"]), "level": r["tourney_level"], "ranks": []},
        )
        if (r["round"] or "").startswith("Q"):
            continue
        for rank in (r["winner_rank"], r["loser_rank"]):
            if rank is not None and rank > 0:
                ev["ranks"].append(float(rank))

    # yearly terciles of Challenger field strength
    by_year: dict[int, list[float]] = {}
    for ev in events.values():
        if ev["source"] in MEN_SOURCES and (ev["level"] or "").upper() == "C" and len(ev["ranks"]) >= 8:
            ev["field_rank"] = float(median(ev["ranks"]))
            by_year.setdefault(ev["year"], []).append(ev["field_rank"])
        else:
            ev["field_rank"] = float(median(ev["ranks"])) if ev["ranks"] else None
    cutoffs = {}
    for year, vals in sorted(by_year.items()):
        arr = np.array(vals)
        cutoffs[str(year)] = {
            "big_max": round(float(np.quantile(arr, 1 / 3)), 1),
            "small_min": round(float(np.quantile(arr, 2 / 3)), 1),
            "n_events": int(len(arr)),
        }

    def ch_tier(year: int, field_rank: float | None) -> str:
        c = cutoffs.get(str(year)) or (cutoffs[max(cutoffs)] if cutoffs else None)
        if field_rank is None or c is None:
            return "mid"
        if field_rank <= c["big_max"]:
            return "big"
        if field_rank >= c["small_min"]:
            return "small"
        return "mid"

    out_rows = []
    counts: dict[str, int] = {}
    for tid, ev in events.items():
        is_ch = ev["source"] in MEN_SOURCES and (ev["level"] or "").upper() == "C"
        tier = ch_tier(ev["year"], ev["field_rank"]) if is_ch else None
        seg = segment_from_level(ev["source"], ev["level"], tier)
        if tier is None:
            tier = {"ITF_BIG": "big", "ITF_SMALL": "small"}.get(seg, "main")
        out_rows.append((tid, ev["source"], ev["year"], ev["level"], ev["field_rank"], len(set(ev["ranks"])), tier, seg))
        counts[seg] = counts.get(seg, 0) + 1

    conn.execute("DELETE FROM event_tiers")
    conn.executemany("INSERT INTO event_tiers VALUES (?,?,?,?,?,?,?,?)", out_rows)
    conn.execute(
        """
        UPDATE pbp_matches
        SET segment = (
            SELECT et.segment FROM matches m JOIN event_tiers et ON et.tourney_id = m.tourney_id
            WHERE m.id = pbp_matches.match_id
        )
        WHERE match_id IS NOT NULL
        """
    )
    conn.commit()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    CUTOFFS_PATH.write_text(json.dumps({"CH": cutoffs, "itf_big_min_prize": ITF_BIG_MIN_PRIZE}, indent=2))
    pbp_counts = dict(
        conn.execute(
            "SELECT segment, COUNT(*) FROM pbp_matches WHERE valid = 1 AND segment IS NOT NULL GROUP BY segment"
        ).fetchall()
    )
    return {"events": counts, "pbp_matches": pbp_counts, "ch_cutoffs": cutoffs}


def load_cutoffs() -> dict:
    if CUTOFFS_PATH.exists():
        return json.loads(CUTOFFS_PATH.read_text())
    return {}


# --------------------------------------------------------------------------------------
# Live resolution
# --------------------------------------------------------------------------------------


def latest_rank(conn: sqlite3.Connection, player_id: int | None) -> int | None:
    if not player_id:
        return None
    row = conn.execute(
        """
        SELECT CASE WHEN winner_id = ? THEN winner_rank ELSE loser_rank END AS rank
        FROM matches
        WHERE (winner_id = ? OR loser_id = ?)
        ORDER BY tourney_date DESC, match_num DESC
        LIMIT 1
        """,
        (player_id, player_id, player_id),
    ).fetchone()
    return int(row["rank"]) if row and row["rank"] else None


def resolve_segment(
    conn: sqlite3.Connection,
    tour: str,
    tier: str | None = None,
    p1_id: int | None = None,
    p2_id: int | None = None,
    event: str | None = None,
) -> dict:
    """Pick the segment for a live quote.

    Priority: explicit `tier` > named `event` (looked up in event_tiers) > the two players'
    latest rankings against the most recent season's Challenger cutoffs.
    """
    tour = tour.upper()
    if tour == "ATP":
        return {"segment": "ATP", "tier": "main", "tier_source": "tour"}
    if tour == "WTA":
        return {"segment": "WTA", "tier": "main", "tier_source": "tour"}

    if tier:
        t = tier.lower()
        if tour == "ITF":
            seg = "ITF_BIG" if t in {"big", "main"} else "ITF_SMALL"
        else:
            seg = f"CH_{t.upper()}" if t in {"big", "mid", "small"} else "CH_MID"
        return {"segment": seg, "tier": t, "tier_source": "given"}

    if event:
        row = conn.execute(
            """
            SELECT et.tier, et.segment, et.field_rank, m.tourney_name
            FROM event_tiers et JOIN matches m ON m.tourney_id = et.tourney_id
            WHERE m.tourney_name LIKE ? ORDER BY et.year DESC LIMIT 1
            """,
            (f"%{event}%",),
        ).fetchone()
        if row:
            return {"segment": row["segment"], "tier": row["tier"], "tier_source": f"event:{row['tourney_name']}",
                    "field_rank": row["field_rank"]}

    if tour == "ITF":
        return {"segment": "ITF_SMALL", "tier": "small", "tier_source": "default"}

    ranks = [r for r in (latest_rank(conn, p1_id), latest_rank(conn, p2_id)) if r]
    cut = load_cutoffs().get("CH") or {}
    if ranks and cut:
        latest = cut[max(cut)]
        proxy = float(np.mean(ranks))
        t = "big" if proxy <= latest["big_max"] else "small" if proxy >= latest["small_min"] else "mid"
        return {"segment": f"CH_{t.upper()}", "tier": t, "tier_source": "player_ranks", "field_rank": proxy}
    return {"segment": "CH_MID", "tier": "mid", "tier_source": "default"}


# --------------------------------------------------------------------------------------
# Report: where do the small leagues behave differently?
# --------------------------------------------------------------------------------------


def tier_report(conn: sqlite3.Connection, since: str = "2011-01-01") -> dict:
    from research.analyze import elo_win_prob
    from research.live import parse_sets

    _ensure_schema(conn)
    rows = conn.execute(
        """
        SELECT et.segment, et.field_rank, m.completion, m.score, m.winner_elo, m.loser_elo
        FROM matches m JOIN event_tiers et ON et.tourney_id = m.tourney_id
        WHERE m.tourney_date >= ? AND m.round NOT LIKE 'Q%' AND m.winner_elo IS NOT NULL
          AND COALESCE(m.best_of, 3) = 3
        """,
        (since,),
    ).fetchall()

    agg: dict[str, dict] = {}
    for r in rows:
        a = agg.setdefault(
            r["segment"],
            {"n": 0, "retired": 0, "walkover": 0, "field": [], "fav_p": [], "fav_won": [],
             "heavy_p": [], "heavy_won": [], "fav_s1_won": 0, "fav_s1_then_lost": 0, "three_sets": 0, "decided": 0},
        )
        a["n"] += 1
        if r["field_rank"] is not None:
            a["field"].append(r["field_rank"])
        if r["completion"] == "Walkover":
            a["walkover"] += 1
            continue
        if r["completion"] == "Retired":
            a["retired"] += 1
        p_w = elo_win_prob(r["winner_elo"], r["loser_elo"])
        fav_won = 1 if p_w >= 0.5 else 0
        p_fav = max(p_w, 1 - p_w)
        if p_fav >= 0.65:
            a["fav_p"].append(p_fav)
            a["fav_won"].append(fav_won)
        if p_fav >= 0.85:
            a["heavy_p"].append(p_fav)
            a["heavy_won"].append(fav_won)
        if r["completion"] == "Completed":
            info = parse_sets(r["score"])
            if info and info["completed"]:
                a["decided"] += 1
                s1_a, s1_b = info["completed"][0]
                winner_took_s1 = s1_a > s1_b
                fav_took_s1 = winner_took_s1 if fav_won else not winner_took_s1
                if p_fav >= 0.65 and fav_took_s1:
                    a["fav_s1_won"] += 1
                    if not fav_won:
                        a["fav_s1_then_lost"] += 1
                if len(info["completed"]) == 3:
                    a["three_sets"] += 1

    def rate(num, den):
        return round(num / den, 4) if den else None

    out = {}
    for seg in ALL_SEGMENTS:
        a = agg.get(seg)
        if not a:
            continue
        fav_exp = float(np.mean(a["fav_p"])) if a["fav_p"] else None
        fav_act = float(np.mean(a["fav_won"])) if a["fav_won"] else None
        heavy_exp = float(np.mean(a["heavy_p"])) if a["heavy_p"] else None
        heavy_act = float(np.mean(a["heavy_won"])) if a["heavy_won"] else None
        out[seg] = {
            "matches": a["n"],
            "median_field_rank": round(float(median(a["field"])), 0) if a["field"] else None,
            "retire_rate": rate(a["retired"], a["n"] - a["walkover"]),
            "walkover_rate": rate(a["walkover"], a["n"]),
            "three_set_rate": rate(a["three_sets"], a["decided"]),
            "fav65_n": len(a["fav_p"]),
            "fav65_expected": round(fav_exp, 4) if fav_exp is not None else None,
            "fav65_actual": round(fav_act, 4) if fav_act is not None else None,
            "fav65_gap_pp": round(100 * (fav_act - fav_exp), 2) if fav_exp is not None else None,
            "heavy85_n": len(a["heavy_p"]),
            "heavy85_expected": round(heavy_exp, 4) if heavy_exp is not None else None,
            "heavy85_actual": round(heavy_act, 4) if heavy_act is not None else None,
            "heavy85_gap_pp": round(100 * (heavy_act - heavy_exp), 2) if heavy_exp is not None else None,
            "fav_up_a_set_then_lost": rate(a["fav_s1_then_lost"], a["fav_s1_won"]),
        }
    return {"since": since, "segments": out}


def format_tier_report(rep: dict) -> str:
    def pct(x):
        return f"{100 * x:5.1f}" if x is not None else "    -"

    lines = [f"Segment behaviour since {rep['since']} (best-of-3, main draw, pre-match Elo)", ""]
    lines.append(f"{'segment':>10} | {'matches':>8} | {'field':>5} | {'ret%':>5} | {'w/o%':>5} | {'3set%':>5} | "
                 f"{'fav65 exp/act (gap)':>22} | {'heavy85 exp/act (gap)':>22} | {'fav up-set lost%':>16}")
    for seg, a in rep["segments"].items():
        fav = (f"{100 * a['fav65_expected']:5.1f}/{100 * a['fav65_actual']:5.1f} ({a['fav65_gap_pp']:+5.1f})"
               if a["fav65_expected"] is not None else "-")
        heavy = (f"{100 * a['heavy85_expected']:5.1f}/{100 * a['heavy85_actual']:5.1f} ({a['heavy85_gap_pp']:+5.1f})"
                 if a["heavy85_expected"] is not None else "-")
        field = f"{int(a['median_field_rank']):5d}" if a["median_field_rank"] else "    -"
        lines.append(f"{seg:>10} | {a['matches']:8d} | {field} | {pct(a['retire_rate'])} | {pct(a['walkover_rate'])} | "
                     f"{pct(a['three_set_rate'])} | {fav:>22} | {heavy:>22} | {pct(a['fav_up_a_set_then_lost']):>16}")
    lines += ["", "gap = actual favourite win rate minus the Elo-expected rate, in percentage points.",
              "A negative gap means favourites in that segment lose more often than their rating says."]
    return "\n".join(lines)
