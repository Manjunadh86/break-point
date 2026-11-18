"""Parse Sackmann point-by-point strings into per-point score states.

Format (tennis_pointbypoint): one character per point from the server's view,
S = server won, R = returner won, A = ace, D = double fault. Games are split by
';', sets by '.', and '/' marks a change of serve inside a tiebreak.

Every row produced here is a *pre-point* state: the score before the point is
played, who serves it, and who won it. That is the row an in-play model is
trained on, and the row the replay harness scores.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

# Column layout of the state matrix returned by parse_match().
COLS = (
    "idx",       # 0  point index in match (0-based)
    "set1",      # 1  sets won by player 1 before the point
    "set2",      # 2
    "g1",        # 3  games in current set
    "g2",        # 4
    "p1",        # 5  points in current game (0..4 where 4 = advantage) or TB points
    "p2",        # 6
    "in_tb",     # 7  1 if the current game is a tiebreak
    "server",    # 8  1 or 2
    "winner",    # 9  1 or 2, who won this point
    "ace",       # 10
    "df",        # 11
    "set_idx",   # 12 0-based set number
    "game_idx",  # 13 0-based game number within the match (TB counts as one)
)
C = {name: i for i, name in enumerate(COLS)}

SET_RE = re.compile(r"(\d+)-(\d+)(?:\((\d+)\))?")


@dataclass
class ParsedMatch:
    states: np.ndarray                 # (n_points, len(COLS)) int16
    set_scores: list[tuple[int, int]]  # games per set from player 1's view
    winner: int                        # 1 or 2 as implied by the point string
    valid: bool
    error: str | None = None
    n_points: int = 0
    tb_sets: list[int] = field(default_factory=list)
    best_of: int = 3

    @property
    def sets(self) -> tuple[int, int]:
        s1 = sum(1 for a, b in self.set_scores if a > b)
        s2 = sum(1 for a, b in self.set_scores if b > a)
        return s1, s2


def _game_points(p1: int, p2: int) -> tuple[int, int, bool]:
    """Advance game score after player 1 wins the point (swap args for player 2)."""
    # 0..3 = 0,15,30,40; 4 = advantage
    if p1 >= 3 and p2 >= 3:
        if p1 == p2:
            return p1 + 1, p2, False          # advantage to winner
        if p1 > p2:
            return p1, p2, True               # game
        return p1, p2 - 1, False              # back to deuce (p2 had advantage)
    if p1 == 3:
        return p1, p2, True                   # game
    return p1 + 1, p2, False


def _tb_server(first_server: int, points_played: int) -> int:
    """Who serves TB point number `points_played` (0-based)."""
    if points_played == 0:
        return first_server
    block = (points_played - 1) // 2
    other = 3 - first_server
    return other if block % 2 == 0 else first_server


def _set_finished(g1: int, g2: int) -> bool:
    hi, lo = max(g1, g2), min(g1, g2)
    return (hi >= 6 and hi - lo >= 2) or (hi == 7 and lo == 6)


def parse_match(pbp: str, server1: int = 1, score: str | None = None, winner: int | None = None) -> ParsedMatch:
    """Walk a point string and return one pre-point state per point.

    server1: 1 or 2, who served the first game. With Sackmann files this is
    always player 1 (columns are server1/server2), but it is exposed for
    other feeds.
    """
    rows: list[tuple[int, ...]] = []
    set_scores: list[tuple[int, int]] = []
    tb_sets: list[int] = []
    s1 = s2 = 0
    idx = 0
    game_idx = 0
    error = None

    sets_raw = [s for s in (pbp or "").strip().split(".") if s]
    for set_i, set_str in enumerate(sets_raw):
        g1 = g2 = 0
        games = [g for g in set_str.split(";") if g]
        for game_str in games:
            server = server1 if game_idx % 2 == 0 else 3 - server1
            is_tb = (g1 == 6 and g2 == 6) or ("/" in game_str)
            tokens = game_str.replace("/", "")
            p1 = p2 = 0
            game_over = False
            game_winner = 0
            tb_points = 0
            for ch in tokens:
                if ch not in "SRAD":
                    error = f"bad token {ch!r}"
                    break
                if game_over:
                    error = f"points after game over in set {set_i + 1}"
                    break
                srv = _tb_server(server, tb_points) if is_tb else server
                server_won = ch in "SA"
                pw = srv if server_won else 3 - srv
                rows.append(
                    (idx, s1, s2, g1, g2, p1, p2, int(is_tb), srv, pw, int(ch == "A"), int(ch == "D"), set_i, game_idx)
                )
                idx += 1
                if is_tb:
                    tb_points += 1
                    if pw == 1:
                        p1 += 1
                    else:
                        p2 += 1
                    if max(p1, p2) >= 7 and abs(p1 - p2) >= 2:
                        game_over = True
                        game_winner = 1 if p1 > p2 else 2
                else:
                    if pw == 1:
                        p1, p2, game_over = _game_points(p1, p2)
                    else:
                        p2, p1, game_over = _game_points(p2, p1)
                    if game_over:
                        game_winner = pw
            if error:
                break
            if not game_over:
                error = f"unfinished game in set {set_i + 1}"
                break
            if game_winner == 1:
                g1 += 1
            else:
                g2 += 1
            if is_tb:
                tb_sets.append(set_i)
            game_idx += 1
            if _set_finished(g1, g2):
                break
        if error:
            break
        set_scores.append((g1, g2))
        if not _set_finished(g1, g2):
            error = f"set {set_i + 1} ended {g1}-{g2}"
            break
        if g1 > g2:
            s1 += 1
        else:
            s2 += 1

    # Best-of-five if anyone needed more than two sets.
    best_of = 5 if (s1 + s2 > 3 or max(s1, s2) == 3) else 3
    need = best_of // 2 + 1
    implied_winner = 1 if s1 > s2 else 2 if s2 > s1 else 0
    valid = error is None and implied_winner in (1, 2) and max(s1, s2) == need and min(s1, s2) < need
    if valid and score:
        expected = [(int(a), int(b)) for a, b, _ in SET_RE.findall(score)]
        if expected and expected != set_scores:
            # Sackmann scores are from the winner's view; flip if winner was player 2.
            flipped = [(b, a) for a, b in set_scores]
            if expected != flipped:
                valid = False
                error = f"score mismatch {score} vs {set_scores}"
    if valid and winner in (1, 2) and winner != implied_winner:
        valid = False
        error = f"winner mismatch {winner} vs {implied_winner}"

    states = np.array(rows, dtype=np.int16) if rows else np.zeros((0, len(COLS)), dtype=np.int16)
    return ParsedMatch(
        states=states,
        set_scores=set_scores,
        winner=implied_winner,
        valid=valid,
        error=error,
        n_points=len(rows),
        tb_sets=tb_sets,
        best_of=best_of,
    )


# Running-stat column layout returned by running_stats().
RS_COLS = (
    "sv_pts_1", "sv_won_1", "sv_pts_2", "sv_won_2",
    "last4_1", "last8_1", "streak",
    "bp_faced_1", "bp_faced_2", "bp_saved_1", "bp_saved_2",
    "breaks_1", "breaks_2", "games_since_break", "last_break_by",
    "ace_1", "ace_2", "df_1", "df_2",
)
RS = {name: i for i, name in enumerate(RS_COLS)}


def running_stats(states: np.ndarray) -> np.ndarray:
    """Running tallies *before* each point, from player 1's view (float32, n x len(RS_COLS))."""
    n = len(states)
    out = np.zeros((n, len(RS_COLS)), dtype=np.float32)
    sv_pts = [0, 0, 0]
    sv_won = [0, 0, 0]
    bp_faced = [0, 0, 0]
    bp_saved = [0, 0, 0]
    breaks = [0, 0, 0]
    aces = [0, 0, 0]
    dfs = [0, 0, 0]
    hist: list[int] = []
    streak = 0
    games_since_break = 0
    last_break_by = 0
    prev_game = -1
    prev_server = 0
    prev_tb = False
    prev_score = (0, 0, 0, 0)  # set1, set2, g1, g2 at the previous point
    for i in range(n):
        st = states[i]
        game_idx = int(st[C["game_idx"]])
        srv = int(st[C["server"]])
        score = (int(st[C["set1"]]), int(st[C["set2"]]), int(st[C["g1"]]), int(st[C["g2"]]))
        if game_idx != prev_game and prev_game >= 0:
            # A game just ended; who won it? Set change => set winner took the last game.
            if score[0] > prev_score[0]:
                game_winner = 1
            elif score[1] > prev_score[1]:
                game_winner = 2
            elif score[2] > prev_score[2]:
                game_winner = 1
            else:
                game_winner = 2
            if not prev_tb and game_winner != prev_server:
                breaks[game_winner] += 1
                games_since_break = 0
                last_break_by = game_winner
            else:
                games_since_break += 1
        prev_game = game_idx
        prev_server = srv
        prev_tb = bool(st[C["in_tb"]])
        prev_score = score

        out[i, RS["sv_pts_1"]] = sv_pts[1]
        out[i, RS["sv_won_1"]] = sv_won[1]
        out[i, RS["sv_pts_2"]] = sv_pts[2]
        out[i, RS["sv_won_2"]] = sv_won[2]
        out[i, RS["last4_1"]] = sum(hist[-4:])
        out[i, RS["last8_1"]] = sum(hist[-8:])
        out[i, RS["streak"]] = streak
        out[i, RS["bp_faced_1"]] = bp_faced[1]
        out[i, RS["bp_faced_2"]] = bp_faced[2]
        out[i, RS["bp_saved_1"]] = bp_saved[1]
        out[i, RS["bp_saved_2"]] = bp_saved[2]
        out[i, RS["breaks_1"]] = breaks[1]
        out[i, RS["breaks_2"]] = breaks[2]
        out[i, RS["games_since_break"]] = games_since_break
        out[i, RS["last_break_by"]] = last_break_by
        out[i, RS["ace_1"]] = aces[1]
        out[i, RS["ace_2"]] = aces[2]
        out[i, RS["df_1"]] = dfs[1]
        out[i, RS["df_2"]] = dfs[2]

        pw = int(st[C["winner"]])
        p_srv = int(st[C["p1"]] if srv == 1 else st[C["p2"]])
        p_ret = int(st[C["p2"]] if srv == 1 else st[C["p1"]])
        if not st[C["in_tb"]] and p_ret >= 3 and p_ret > p_srv:
            bp_faced[srv] += 1
            if pw == srv:
                bp_saved[srv] += 1
        sv_pts[srv] += 1
        if pw == srv:
            sv_won[srv] += 1
        if st[C["ace"]]:
            aces[srv] += 1
        if st[C["df"]]:
            dfs[srv] += 1
        hist.append(1 if pw == 1 else 0)
        if pw == 1:
            streak = streak + 1 if streak >= 0 else 1
        else:
            streak = streak - 1 if streak <= 0 else -1
    return out
