# Break Point

Local research desk for **ATP Challenger** (and ATP tour) tennis. It does **not** log into Stake, scrape a sportsbook, or place bets. You look at the card and decide.

## What it does

- Pulls public match histories (Jeff Sackmann archive: ATP tour + Challenger/qualifying, WTA + ITF) and the point-by-point archive
- Rebuilds overall and surface Elo
- Builds a matchup card: form, H2H, hold%, retirements, 14-day workload
- Optionally loads **ATP closing odds** from [tennis-data.co.uk](http://www.tennis-data.co.uk) and checks whether favorites cash at the de-vigged rate
- Lets you type the two prices you see on a board and compare them to the Elo prior
- Optional live **snapshots** via [The Odds API](https://the-odds-api.com) (ATP/WTA coverage, rarely Challenger)
- Manual watchlist for Challenger cards and **SRL** notes

SRL on Stake is Simulated Reality League (virtual cricket / sims), not Challenger tennis. There is no public result feed worth training on. The watchlist is a notebook, not a model.

## Setup

```bash
cd /Users/manjunadh/Desktop/stake-automation
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Load the last several seasons (Challenger + ATP). Add `--odds` when you want ATP closing-line calibration:

```bash
python -m research ingest --years 2022-2026
python -m research ingest --odds-only --years 2022-2026
```

Open the desk:

```bash
python -m research serve
```

Then http://127.0.0.1:8765 (pass `--port 8788` or any free port if 8765 is taken by something else).
The first panel, **Point engine**, starts with the public scoreboard (LiveScore: ATP, Challenger, WTA, WTA 125; ESPN main tour as fallback; no ITF World Tour feed is reachable). Press *Load* on a match and the names, tour, event, surface (looked up from the Sackmann history), sets, games, server and previous set fill in; a live match is then followed and re-priced whenever sets, games or server change (auto re-prices never write to the paper log). The feed does not carry the point score inside the game or serve-point counts, so those stay yours to type. You can also type everything by hand: the two players, the score, what you have seen on serve so far, optional prices, and it shows the quote step by step (Elo → segment prior → serve-updated → calibrated → settlement-adjusted) with the per-segment held-out log-loss table underneath.

Live assistant (Markov + Challenger GBM). Train once after ingest:

```bash
python -m research train
python -m research forecast "Alexander Donski" "Philip Henning" --sets 0-1 --games 3-4 --points 30-40 --server p2 --surface Hard --set1 3-6 --line-1 5.5 --line-2 1.15
```

In-play study (set-1 close-out, clocks, live quote):

```bash
python -m research live --level C
python -m research quote "Emilio Nava" "Henri Squire" --sets 1-0 --set1 6-4 --surface Clay --line-a 1.22 --line-b 4.20
```

One-off from the terminal:

```bash
python -m research matchup "Learner Tien" "Ethan Quinn" --surface Hard --line-a 1.72 --line-b 2.15
```

## Dynamic in-play desk (point-by-point)

The second generation of the live model learns from ~94k real point-by-point matches (ATP, Challenger, WTA, ITF; 2011–2017) instead of only set scores. Men and women are ingested together; WTA/ITF player ids are offset by 10,000,000 so one `player_id` is one human.

```bash
# 1. full history, men + women, plus the point-by-point archive (linked to Sackmann rows: Elo, surface, fatigue)
python -m research ingest --years 2009-2026 --pbp

# 2. tier every event by size and see how each segment behaves (favourite collapse, retirements ...)
python -m research tiers

# 3. replay every point through the engines and fit the serve-update prior kappa, per segment
for s in ATP CH_BIG CH_MID CH_SMALL WTA ITF_BIG ITF_SMALL; do python -m research replay --tour $s; done
python -m research hazard-rates          # retirement base rates by segment / surface

# 4. train one calibrator per segment (run them one at a time; ~0.5-2 GB each)
for s in ATP CH_BIG CH_MID CH_SMALL WTA ITF_BIG ITF_SMALL; do python -m research train-inplay --group $s; done

# 5. ask for a quote at any score, with what you have observed so far
python -m research inplay "Learner Tien" "Ethan Quinn" --tour CH --surface Hard \
  --sets 0-1 --games 3-4 --points 30-40 --server 2 --serve-1 25/40 --serve-2 30/42 \
  --prev-set 3-6 --price-1 3.4 --price-2 1.35 --tier small      # or --event "Pune"
```

### Segments

Men and women never share a model, and inside each the events are split by size:

| segment | what it is | how it is decided |
|---|---|---|
| `ATP` | tour-level men (250/500/1000, Slams bo3 excluded) | `tourney_level` |
| `CH_BIG` / `CH_MID` / `CH_SMALL` | Challengers, strongest / middle / weakest third of each season | median ranking of the main-draw field (no prize tier in the data) |
| `WTA` | tour-level women | `tourney_level` |
| `ITF_BIG` / `ITF_SMALL` | $35K+ (and WTA 125) / $10K-$25K | prize money in `tourney_level` |

Every segment gets its own pre-match prior (overall Elo blended with surface Elo, weight and scale fitted per segment: `data/models/prior_params.json`), serve average, κ, calibrator and retirement rates. A live quote picks its segment from `--tier`, else `--event`, else the two players' latest rankings against this season's cutoffs (`tier_source` in the output tells you which). If a segment model is missing it falls back to the `men` / `women` umbrella (`model_group` in the output).

`python -m research tiers` prints, per segment, retirement and walkover rates, 3-set rate, how often Elo favourites actually win versus what their rating says, and how often a favourite that took set one still lost. That is where "small leagues behave differently" shows up if it does; the model cannot see fixing, only deviation.

What a quote contains, in order:

- `p_prior` — Elo-derived match probability inverted into serve/return points.
- `p_updated` — the same chain after an empirical-Bayes update of each player's serve % with the points seen so far (`p = (won + κπ)/(n + κ)`, κ shrinking from ~640 early to ~40 late, fitted by `replay`).
- `p_model` — GBM calibration on top, when `train-inplay` has run (`model_loaded: true`).
- `retirement` / `p_settled` — hazard overlay (base rate × player history × medical-timeout spike) and the probability you actually get paid under `--rule ball_served|one_set|match`.
- `paths` — the six set-sequence paths (c1–c6) from the current state.
- `market` — edge after 5% commission, quarter-Kelly capped at 3%, only if edge clears 3pp plus the model's own uncertainty. `--log` (or the *Write to paper log* tick box in the panel) writes the row to `paper_log` with its segment. When the match is over, record the result — `python -m research paper settle <id> --winner 1|2 [--close 1.20]` or the *P1 won / P2 won* buttons under the panel — and `python -m research paper` (or `/api/paper`) scores everything you settled: model vs prior vs market log loss and Brier, flat-stake and quarter-Kelly P&L on the suggested backs after commission, and average CLV where you typed a closing price. `paper list` prints the rows. Nothing places a bet.

Same thing over HTTP: `GET /api/inplay?p1=...&p2=...&tour=CH&sets=0-1&games=3-4&points=30-40&server=2&serve_1=25/40&serve_2=30/42`, plus `/api/slate` (public scoreboard, surface-enriched), `/api/slate/{id}` (one match, re-fetched), `/api/inplay/model?group=CH_SMALL`, `/api/paper`, `/api/paper/rows` and `POST /api/paper/{id}/settle` (`{"winner": 1|2, "closing_price": optional}`).

## Optional live snapshots

Copy `.env.example` to `.env`, add a free Odds API key, then run `python -m research snapshot` more than once (hours apart). Movement only exists if you store more than one print.

## Honest limits

- Challenger **opening** lines are not in tennis-data. Public history is results + ATP closes.
- Elo vs a soft book is a disagreement, not an edge. Juice still sits in every two-way market.
- Thin Challenger samples and retirements will fake “signals.” Read the flags.
- The point-by-point archive stops in September 2017. The dynamic engine is validated on that era and applied to today's players through current Elo; there is no 2018+ point-level test set.
- The medical-timeout spike in the retirement hazard is a prior (×8, decaying over 3 games), not a fitted number. The paper log has no exchange tape behind it, so CLV is only computed if you type a closing price.

Data credits: Jeff Sackmann / tennis_atp and tennis_wta (archive mirror), tennis-data.co.uk, and the `tennis_pointbypoint` archive (Sackmann; CC BY-NC-SA 4.0 — non-commercial, attribution, share-alike).
