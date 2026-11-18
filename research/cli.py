from __future__ import annotations

import argparse
import json

from research.config import DEFAULT_YEARS


def _years(text: str | None) -> tuple[int, ...]:
    if not text:
        return DEFAULT_YEARS
    if "-" in text and "," not in text:
        a, b = text.split("-", 1)
        return tuple(range(int(a), int(b) + 1))
    return tuple(int(p) for p in text.split(","))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Break Point — Challenger / ATP research desk (no automated betting)."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    ingest = sub.add_parser("ingest", help="Download public match files and rebuild ratings")
    ingest.add_argument("--years", default="2022-2026", help="e.g. 2022-2026 or 2024,2025,2026")
    ingest.add_argument("--odds", action="store_true", help="Also pull ATP closing odds from tennis-data.co.uk")
    ingest.add_argument("--odds-only", action="store_true")
    ingest.add_argument(
        "--sources",
        default="atp,qual_chall,wta,qual_itf",
        help="Comma list of Sackmann files: atp, qual_chall, wta, qual_itf",
    )
    ingest.add_argument("--pbp", action="store_true", help="Also download + link point-by-point matches")

    pbp = sub.add_parser("pbp", help="Download Sackmann point-by-point files and link them to results")
    pbp.add_argument("--files", help="Comma list, e.g. ch_main_archive,wta_main_archive (default: all)")
    pbp.add_argument("--no-link", action="store_true")

    sub.add_parser("snapshot", help="Snapshot live tennis lines via The Odds API")

    matchup = sub.add_parser("matchup", help="Print a research card for two players")
    matchup.add_argument("player_a")
    matchup.add_argument("player_b")
    matchup.add_argument("--surface")
    matchup.add_argument("--line-a", type=float)
    matchup.add_argument("--line-b", type=float)

    live = sub.add_parser("live", help="Print Challenger in-play probability study")
    live.add_argument("--level", default="C", choices=["C", "ATP"])
    live.add_argument("--surface")

    quote = sub.add_parser("quote", help="Live-state prior for two players")
    quote.add_argument("player_a")
    quote.add_argument("player_b")
    quote.add_argument("--sets", default="0-0", help="Set score from A's view, e.g. 1-0 or 1-1")
    quote.add_argument("--set1", help="First-set games, e.g. 7-6 or 6-1")
    quote.add_argument("--surface")
    quote.add_argument("--line-a", type=float)
    quote.add_argument("--line-b", type=float)
    quote.add_argument("--level", default="C")

    sub.add_parser("train", help="Fit the Challenger live-state gradient boosting model")

    replay = sub.add_parser("replay", help="Score in-play engines point by point on historical pbp matches")
    replay.add_argument("--tour", default="CH", choices=["CH", "ATP", "WTA", "ITF", "ALL", "ATP", "CH_BIG", "CH_MID", "CH_SMALL", "WTA", "ITF_BIG", "ITF_SMALL"],
                        help="A tour, or an event segment (see `tiers`)")
    replay.add_argument("--train-until", default="2016-01-01", help="Chronological split date")
    replay.add_argument("--no-fit", action="store_true", help="Skip the kappa grid search")
    replay.add_argument("--limit", type=int)
    replay.add_argument("--json", action="store_true", help="Print full JSON instead of the table")

    ti = sub.add_parser("train-inplay", help="Fit the in-play calibrator for one event segment (or an umbrella group)")
    ti.add_argument("--group", default="CH_SMALL", choices=["ATP", "CH_BIG", "CH_MID", "CH_SMALL", "WTA", "ITF_BIG", "ITF_SMALL", "men", "women"])
    ti.add_argument("--train-until", default="2016-01-01")
    ti.add_argument("--stride", type=int, default=3, help="Keep every k-th point per training match")
    ti.add_argument("--limit-per-tour", type=int)
    ti.add_argument("--json", action="store_true")

    ip = sub.add_parser("inplay", help="Live in-play probability bundle from a scoreboard")
    ip.add_argument("p1")
    ip.add_argument("p2")
    ip.add_argument("--tour", default="CH", choices=["CH", "ATP", "WTA", "ITF"])
    ip.add_argument("--surface")
    ip.add_argument("--tier", choices=["big", "mid", "small"], help="Event size for CH / ITF; inferred from rankings if omitted")
    ip.add_argument("--event", help="Tournament name to look the tier up, e.g. 'Pune'")
    ip.add_argument("--sets", default="0-0")
    ip.add_argument("--games", default="0-0")
    ip.add_argument("--points", default="0-0", help="e.g. 30-40 or 3-5 inside a tiebreak")
    ip.add_argument("--server", default="1", choices=["1", "2"])
    ip.add_argument("--tb", action="store_true", help="Currently in a tiebreak")
    ip.add_argument("--serve-1", help="P1 serve points won/played so far, e.g. 28/40")
    ip.add_argument("--serve-2", help="P2 serve points won/played so far, e.g. 22/38")
    ip.add_argument("--prev-set", help="Previous set games from P1 view, e.g. 4-6")
    ip.add_argument("--streak", type=int, default=0, help="Current point streak (+P1 / -P2)")
    ip.add_argument("--mto-1", type=float, help="Games since P1 took a medical timeout")
    ip.add_argument("--mto-2", type=float, help="Games since P2 took a medical timeout")
    ip.add_argument("--rule", default="ball_served", choices=["ball_served", "one_set", "match"])
    ip.add_argument("--price-1", type=float, help="Decimal price on P1")
    ip.add_argument("--price-2", type=float, help="Decimal price on P2")
    ip.add_argument("--log", action="store_true", help="Append the bundle to the paper-trading log")

    tiers = sub.add_parser("tiers", help="Tier events by size (Challenger field strength, ITF prize) and show how each segment behaves")
    tiers.add_argument("--since", default="2011-01-01")
    tiers.add_argument("--no-build", action="store_true", help="Report only; do not rebuild the event_tiers table")
    tiers.add_argument("--json", action="store_true")

    sub.add_parser("hazard-rates", help="Rebuild retirement base rates from the results table")
    paper = sub.add_parser("paper", help="Paper-trading log: summary, recent rows, settle results")
    paper.add_argument("action", nargs="?", default="summary", choices=["summary", "list", "settle"])
    paper.add_argument("id", nargs="?", type=int, help="paper_log id (for settle)")
    paper.add_argument("--winner", type=int, choices=[1, 2], help="Who won: 1 = P1, 2 = P2")
    paper.add_argument("--close", type=float, help="Closing price on P1 (for CLV)")
    paper.add_argument("--limit", type=int, default=30)

    fc = sub.add_parser("forecast", help="Live match forecast from a scoreboard")
    fc.add_argument("p1")
    fc.add_argument("p2")
    fc.add_argument("--sets", default="0-0")
    fc.add_argument("--games", default="0-0")
    fc.add_argument("--points", default="0-0")
    fc.add_argument("--server", default="p1")
    fc.add_argument("--surface")
    fc.add_argument("--set1")
    fc.add_argument("--line-1", type=float)
    fc.add_argument("--line-2", type=float)

    serve = sub.add_parser("serve", help="Open the local research dashboard")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)

    args = parser.parse_args(argv)

    if args.cmd == "ingest":
        from research.ingest import ingest_sackmann, ingest_tennis_data

        years = _years(args.years)
        result = {}
        if not args.odds_only:
            sources = tuple(s.strip() for s in args.sources.split(",") if s.strip())
            result["matches"] = ingest_sackmann(years, sources=sources)
        if args.odds or args.odds_only:
            result["odds"] = ingest_tennis_data(years)
        if args.pbp:
            from research.ingest import ingest_pbp

            result["pbp"] = ingest_pbp()
        print(json.dumps(result, indent=2))
        return

    if args.cmd == "pbp":
        from research.config import PBP_FILES
        from research.ingest import ingest_pbp

        files = tuple(f.strip() for f in args.files.split(",")) if args.files else PBP_FILES
        print(json.dumps(ingest_pbp(files, link=not args.no_link), indent=2))
        return

    if args.cmd == "snapshot":
        from research.ingest import snapshot_odds

        print(json.dumps(snapshot_odds(), indent=2))
        return

    if args.cmd == "matchup":
        from research.analyze import matchup_report
        from research.db import session

        with session() as conn:
            report = matchup_report(
                conn,
                args.player_a,
                args.player_b,
                surface=args.surface,
                line_a=args.line_a,
                line_b=args.line_b,
            )
        print(json.dumps(report, indent=2, default=str))
        return

    if args.cmd == "live":
        from research.db import session
        from research.live import live_study

        with session() as conn:
            print(json.dumps(live_study(conn, level=args.level, surface=args.surface), indent=2))
        return

    if args.cmd == "quote":
        from research.db import session
        from research.live import live_quote

        sa, sb = (int(p) for p in args.sets.replace(":", "-").split("-", 1))
        with session() as conn:
            print(
                json.dumps(
                    live_quote(
                        conn,
                        args.player_a,
                        args.player_b,
                        sets_a=sa,
                        sets_b=sb,
                        surface=args.surface,
                        set1_games=args.set1,
                        line_a=args.line_a,
                        line_b=args.line_b,
                        level=args.level,
                    ),
                    indent=2,
                    default=str,
                )
            )
        return

    if args.cmd == "train":
        from research.db import session
        from research.predictor import train_live_model

        with session() as conn:
            print(json.dumps(train_live_model(conn), indent=2))
        return

    if args.cmd == "replay":
        from research.db import session
        from research.replay import format_report, run_replay

        with session() as conn:
            result = run_replay(conn, tour=args.tour, train_until=args.train_until, fit=not args.no_fit, limit=args.limit)
        print(json.dumps(result, indent=2) if args.json else format_report(result))
        return

    if args.cmd == "train-inplay":
        from research.db import session
        from research.inplay import format_inplay_report, train_inplay_model

        with session() as conn:
            result = train_inplay_model(conn, group=args.group, train_until=args.train_until, stride=args.stride,
                                        limit_per_tour=args.limit_per_tour)
        print(json.dumps(result, indent=2, default=str) if args.json else format_inplay_report(result))
        return

    if args.cmd == "inplay":
        from research.db import session
        from research.inplay import inplay_bundle

        def pair(text):
            a, b = text.replace(":", "-").split("-", 1)
            return a.strip(), b.strip()

        def frac(text):
            if not text:
                return 0, 0
            won, played = text.split("/", 1)
            return int(won), int(played)

        s1, s2 = (int(x) for x in pair(args.sets))
        g1, g2 = (int(x) for x in pair(args.games))
        pt1, pt2 = pair(args.points)
        sv1_won, sv1_pts = frac(args.serve_1)
        sv2_won, sv2_pts = frac(args.serve_2)
        prev = tuple(int(x) for x in pair(args.prev_set)) if args.prev_set else None
        with session() as conn:
            out = inplay_bundle(
                conn, args.p1, args.p2, tour=args.tour, surface=args.surface,
                sets=(s1, s2), games=(g1, g2), points=(pt1, pt2), server=int(args.server), in_tb=args.tb,
                serve_1=(sv1_won, sv1_pts), serve_2=(sv2_won, sv2_pts), prev_set=prev, streak=args.streak,
                mto_1=args.mto_1, mto_2=args.mto_2, rule=args.rule,
                price_1=args.price_1, price_2=args.price_2, log=args.log,
                tier=args.tier, event=args.event,
            )
        print(json.dumps(out, indent=2, default=str))
        return

    if args.cmd == "tiers":
        from research.db import session
        from research.tiers import build_event_tiers, format_tier_report, tier_report

        from research.prior import fit_prior_params

        with session() as conn:
            built = None if args.no_build else build_event_tiers(conn)
            if built is not None:
                built["prior"] = fit_prior_params(conn)
                print("blended prior per segment (w_overall, scale, train log loss vs Elo-only):")
                for seg, pr in built["prior"].items():
                    print(f"  {seg:>10}: w={pr['w_overall']:.1f} scale={pr['scale']:.1f}  "
                          f"{pr['log_loss']:.4f} vs {pr['elo_only_log_loss']:.4f}")
            rep = tier_report(conn, since=args.since)
        if args.json:
            print(json.dumps({"build": built, "report": rep}, indent=2, default=str))
        else:
            if built:
                print("events per segment:", built["events"])
                print("point-by-point matches per segment:", built["pbp_matches"])
                latest = max(built["ch_cutoffs"])
                print(f"Challenger cutoffs {latest}: big = median field rank <= {built['ch_cutoffs'][latest]['big_max']}, "
                      f"small >= {built['ch_cutoffs'][latest]['small_min']}\n")
            print(format_tier_report(rep))
        return

    if args.cmd == "hazard-rates":
        from research.db import session
        from research.hazard import build_rates

        with session() as conn:
            rates = build_rates(conn)
        print(json.dumps({k: v for k, v in rates.items() if k.endswith("|*")}, indent=2))
        return

    if args.cmd == "paper":
        from research.db import session
        from research.market import paper_rows, paper_summary, settle_paper

        with session() as conn:
            if args.action == "settle":
                if args.id is None or args.winner is None:
                    raise SystemExit("usage: paper settle <id> --winner 1|2 [--close 1.50]")
                print(json.dumps(settle_paper(conn, args.id, args.winner, args.close), indent=2, default=str))
            elif args.action == "list":
                for r in paper_rows(conn, limit=args.limit):
                    st = r["state"] or {}
                    res = f"P{r['winner']} won" if r["winner"] else "open"
                    print(f"#{r['id']:<4} {r['logged_at']}  {r['p1']} v {r['p2']}  [{r.get('segment') or r['tour']}] "
                          f"sets {'-'.join(map(str, st.get('sets', [])))} games {'-'.join(map(str, st.get('games', [])))}  "
                          f"p_model {r['p_model']:.3f}  price {r['price'] or '-'}  {res}")
            else:
                print(json.dumps(paper_summary(conn), indent=2, default=str))
        return

    if args.cmd == "forecast":
        from research.db import session
        from research.predictor import forecast

        def pair(text):
            a, b = text.replace(":", "-").split("-", 1)
            return a, b

        s1, s2 = (int(x) for x in pair(args.sets))
        g1, g2 = (int(x) for x in pair(args.games))
        pt1, pt2 = pair(args.points)
        with session() as conn:
            print(
                json.dumps(
                    forecast(
                        conn,
                        args.p1,
                        args.p2,
                        sets_1=s1,
                        sets_2=s2,
                        games_1=g1,
                        games_2=g2,
                        points_1=pt1,
                        points_2=pt2,
                        server=args.server,
                        surface=args.surface,
                        set1=args.set1,
                        line_1=args.line_1,
                        line_2=args.line_2,
                    ),
                    indent=2,
                    default=str,
                )
            )
        return

    if args.cmd == "serve":
        import uvicorn

        uvicorn.run("research.app:app", host=args.host, port=args.port, reload=False)
