"""Point → game → set → match probabilities. Advantage scoring, TB at 6-6, BO3."""

from __future__ import annotations

from functools import lru_cache

PointLike = int | str


def parse_point(value: PointLike | None) -> int:
    if value is None or value == "":
        return 0
    if isinstance(value, str):
        token = value.strip().upper()
        table = {"0": 0, "15": 1, "30": 2, "40": 3, "AD": 4, "A": 4}
        if token in table:
            return table[token]
        return int(token)
    return int(value)


@lru_cache(maxsize=None)
def p_win_game(p1: int, p2: int, p_point: float) -> float:
    """P(player 1 wins the game) when p_point is P(P1 wins the next point)."""
    p, q = p_point, 1.0 - p_point
    if p1 >= 3 and p2 >= 3:
        deuce = (p * p) / (p * p + q * q) if (p * p + q * q) else 0.5
        if p1 == p2:
            return deuce
        if p1 > p2:
            return p + q * deuce
        return p * deuce
    if p1 >= 4:
        return 1.0
    if p2 >= 4:
        return 0.0
    return p * p_win_game(p1 + 1, p2, p_point) + q * p_win_game(p1, p2 + 1, p_point)


def _tb_server(points_played: int, first_is_p1: bool) -> bool:
    if points_played == 0:
        return first_is_p1
    block = (points_played - 1) // 2
    return (not first_is_p1) if block % 2 == 0 else first_is_p1


@lru_cache(maxsize=None)
def p_win_tb(p1: int, p2: int, first_is_p1: bool, p_serve: float, p_return: float) -> float:
    """Tiebreak to 7, margin 2. first_is_p1 is who served point 0 of this TB."""
    if p1 >= 7 and p1 - p2 >= 2:
        return 1.0
    if p2 >= 7 and p2 - p1 >= 2:
        return 0.0
    if p1 >= 6 and p2 >= 6:
        p = p_serve if _tb_server(p1 + p2, first_is_p1) else p_return
        q = 1.0 - p
        deuce = (p * p) / (p * p + q * q) if (p * p + q * q) else 0.5
        if p1 == p2:
            return deuce
        if p1 > p2:
            return p + q * deuce
        return p * deuce
    p1_serves = _tb_server(p1 + p2, first_is_p1)
    p = p_serve if p1_serves else p_return
    return p * p_win_tb(p1 + 1, p2, first_is_p1, p_serve, p_return) + (1 - p) * p_win_tb(
        p1, p2 + 1, first_is_p1, p_serve, p_return
    )


@lru_cache(maxsize=None)
def p_win_set(
    g1: int,
    g2: int,
    pts1: int,
    pts2: int,
    p1_serves: bool,
    p_serve: float,
    p_return: float,
    in_tb: bool,
    tb1: int,
    tb2: int,
) -> float:
    if g1 >= 6 and g1 - g2 >= 2:
        return 1.0
    if g2 >= 6 and g2 - g1 >= 2:
        return 0.0
    if g1 == 7 and g2 == 6:
        return 1.0
    if g2 == 7 and g1 == 6:
        return 0.0
    if in_tb or (g1 == 6 and g2 == 6):
        played = tb1 + tb2
        first = True
        for cand in (True, False):
            if _tb_server(played, cand) == p1_serves:
                first = cand
                break
        return p_win_tb(tb1, tb2, first, p_serve, p_return)

    p_point = p_serve if p1_serves else p_return
    p_game = p_win_game(pts1, pts2, p_point)
    win = p_game * p_win_set(g1 + 1, g2, 0, 0, not p1_serves, p_serve, p_return, False, 0, 0)
    lose = (1 - p_game) * p_win_set(g1, g2 + 1, 0, 0, not p1_serves, p_serve, p_return, False, 0, 0)
    return win + lose


@lru_cache(maxsize=None)
def p_win_match(
    s1: int,
    s2: int,
    g1: int,
    g2: int,
    pts1: int,
    pts2: int,
    p1_serves: bool,
    p_serve: float,
    p_return: float,
    need: int = 2,
    in_tb: bool = False,
    tb1: int = 0,
    tb2: int = 0,
) -> float:
    if s1 >= need:
        return 1.0
    if s2 >= need:
        return 0.0
    p_set = p_win_set(g1, g2, pts1, pts2, p1_serves, p_serve, p_return, in_tb, tb1, tb2)
    nxt_win = 0.5 * (
        p_win_match(s1 + 1, s2, 0, 0, 0, 0, True, p_serve, p_return, need)
        + p_win_match(s1 + 1, s2, 0, 0, 0, 0, False, p_serve, p_return, need)
    )
    nxt_lose = 0.5 * (
        p_win_match(s1, s2 + 1, 0, 0, 0, 0, True, p_serve, p_return, need)
        + p_win_match(s1, s2 + 1, 0, 0, 0, 0, False, p_serve, p_return, need)
    )
    return p_set * nxt_win + (1 - p_set) * nxt_lose


def match_from_state(
    sets_1: int,
    sets_2: int,
    games_1: int = 0,
    games_2: int = 0,
    points_1: PointLike = 0,
    points_2: PointLike = 0,
    p1_serves: bool = True,
    p_serve: float = 0.64,
    p_return: float = 0.36,
    best_of: int = 3,
    in_tb: bool = False,
    tb_1: int = 0,
    tb_2: int = 0,
) -> dict[str, float]:
    need = best_of // 2 + 1
    pts1, pts2 = parse_point(points_1), parse_point(points_2)
    p_point = p_serve if p1_serves else p_return
    p_game = 1.0 if in_tb else p_win_game(pts1, pts2, p_point)
    p_set = p_win_set(games_1, games_2, pts1, pts2, p1_serves, p_serve, p_return, in_tb, tb_1, tb_2)
    p_match = p_win_match(
        sets_1, sets_2, games_1, games_2, pts1, pts2, p1_serves, p_serve, p_return, need, in_tb, tb_1, tb_2
    )
    return {
        "point": p_point,
        "game": p_game,
        "set": p_set,
        "match": p_match,
    }


# --------------------------------------------------------------------------------------
# State helpers shared by the replay harness and the in-play model.
# A state is (set1, set2, g1, g2, p1, p2, server, in_tb) with p1/p2 holding tiebreak
# points when in_tb is true (the layout produced by research.pbp).
# --------------------------------------------------------------------------------------

State = tuple[int, int, int, int, int, int, int, int]


def match_prob(state: State, p_serve: float, p_return: float, best_of: int = 3) -> float:
    """P(player 1 wins the match) from a pbp-style state."""
    s1, s2, g1, g2, p1, p2, server, in_tb = state
    need = best_of // 2 + 1
    if s1 >= need:
        return 1.0
    if s2 >= need:
        return 0.0
    if in_tb:
        return p_win_match(s1, s2, g1, g2, 0, 0, server == 1, p_serve, p_return, need, True, p1, p2)
    return p_win_match(s1, s2, g1, g2, p1, p2, server == 1, p_serve, p_return, need, False, 0, 0)


def advance(state: State, p1_won: bool, best_of: int = 3) -> State:
    """Score state after player 1 wins (or loses) the next point. Handles TB rotation."""
    s1, s2, g1, g2, p1, p2, server, in_tb = state
    if in_tb:
        tb_first = _tb_first_server(server, p1 + p2)
        if p1_won:
            p1 += 1
        else:
            p2 += 1
        if max(p1, p2) >= 7 and abs(p1 - p2) >= 2:
            # set over; next set's first server is the one who received TB point 0
            new_server = 3 - tb_first
            if p1 > p2:
                return (s1 + 1, s2, 0, 0, 0, 0, new_server, 0)
            return (s1, s2 + 1, 0, 0, 0, 0, new_server, 0)
        played = p1 + p2
        nxt = (3 - tb_first) if ((played - 1) // 2) % 2 == 0 else tb_first
        return (s1, s2, g1, g2, p1, p2, nxt, 1)
    # regular game
    if p1_won:
        if p1 >= 3 and p2 >= 3:
            if p1 == p2:
                return (s1, s2, g1, g2, p1 + 1, p2, server, 0)
            if p1 < p2:
                return (s1, s2, g1, g2, 3, 3, server, 0)
            g1 += 1
        elif p1 == 3:
            g1 += 1
        else:
            return (s1, s2, g1, g2, p1 + 1, p2, server, 0)
    else:
        if p1 >= 3 and p2 >= 3:
            if p1 == p2:
                return (s1, s2, g1, g2, p1, p2 + 1, server, 0)
            if p2 < p1:
                return (s1, s2, g1, g2, 3, 3, server, 0)
            g2 += 1
        elif p2 == 3:
            g2 += 1
        else:
            return (s1, s2, g1, g2, p1, p2 + 1, server, 0)
    # game over
    new_server = 3 - server
    hi, lo = max(g1, g2), min(g1, g2)
    if (hi >= 6 and hi - lo >= 2) or (hi == 7 and lo == 6):
        if g1 > g2:
            return (s1 + 1, s2, 0, 0, 0, 0, new_server, 0)
        return (s1, s2 + 1, 0, 0, 0, 0, new_server, 0)
    if g1 == 6 and g2 == 6:
        return (s1, s2, g1, g2, 0, 0, new_server, 1)
    return (s1, s2, g1, g2, 0, 0, new_server, 0)


def _tb_first_server(current_server: int, points_before: int) -> int:
    """Given who serves TB point `points_before` (0-based), recover who served point 0."""
    if points_before <= 0:
        return current_server
    block = (points_before - 1) // 2
    return (3 - current_server) if block % 2 == 0 else current_server


def importance(state: State, p_serve: float, p_return: float, best_of: int = 3) -> float:
    """Morris / Klaassen-Magnus importance: P(win match | win point) - P(win match | lose point)."""
    win = fast_match_prob(advance(state, True, best_of), p_serve, p_return, best_of)
    lose = fast_match_prob(advance(state, False, best_of), p_serve, p_return, best_of)
    return win - lose


# --------------------------------------------------------------------------------------
# Fast evaluator for replay: set/match tables per quantised (p_serve, p_return).
# Equivalent to match_prob() but with a small explicit cache so millions of points
# with slowly drifting serve probabilities stay fast.
# --------------------------------------------------------------------------------------

_Q = 0.0025
_tables: dict[tuple[float, float, int], tuple[dict, dict]] = {}


def _quant(x: float) -> float:
    return round(round(x / _Q) * _Q, 4)


def _build_tables(s: float, r: float, best_of: int) -> tuple[dict, dict]:
    """set_tab[(g1, g2, server)] = P(P1 wins set) at game start; match_tab[(s1, s2)] = P(P1 wins match)."""
    hold1 = p_win_game(0, 0, s)      # P1 wins own service game
    break1 = p_win_game(0, 0, r)     # P1 wins P2's service game
    set_tab: dict[tuple[int, int, int], float] = {}

    def pset(g1: int, g2: int, srv: int) -> float:
        key = (g1, g2, srv)
        if key in set_tab:
            return set_tab[key]
        hi, lo = max(g1, g2), min(g1, g2)
        if (hi >= 6 and hi - lo >= 2) or (hi == 7 and lo == 6):
            val = 1.0 if g1 > g2 else 0.0
        elif g1 >= 6 and g1 == g2:
            val = p_win_tb(0, 0, srv == 1, s, r)
        else:
            pg = hold1 if srv == 1 else break1
            val = pg * pset(g1 + 1, g2, 3 - srv) + (1 - pg) * pset(g1, g2 + 1, 3 - srv)
        set_tab[key] = val
        return val

    for g1 in range(0, 7):
        for g2 in range(0, 7):
            for srv in (1, 2):
                pset(g1, g2, srv)
    for g1, g2 in ((7, 5), (5, 7), (7, 6), (6, 7)):
        for srv in (1, 2):
            pset(g1, g2, srv)
    set0 = 0.5 * (set_tab[(0, 0, 1)] + set_tab[(0, 0, 2)])
    need = best_of // 2 + 1
    match_tab: dict[tuple[int, int], float] = {}

    def pm(s1: int, s2: int) -> float:
        if s1 >= need:
            return 1.0
        if s2 >= need:
            return 0.0
        key = (s1, s2)
        if key not in match_tab:
            match_tab[key] = set0 * pm(s1 + 1, s2) + (1 - set0) * pm(s1, s2 + 1)
        return match_tab[key]

    for s1 in range(need):
        for s2 in range(need):
            pm(s1, s2)
    match_tab[(need, 0)] = 1.0
    match_tab[(0, need)] = 0.0
    return set_tab, match_tab


def fast_match_prob(state: State, p_serve: float, p_return: float, best_of: int = 3) -> float:
    s1, s2, g1, g2, p1, p2, server, in_tb = state
    need = best_of // 2 + 1
    if s1 >= need:
        return 1.0
    if s2 >= need:
        return 0.0
    s, r = _quant(p_serve), _quant(p_return)
    key = (s, r, best_of)
    tabs = _tables.get(key)
    if tabs is None:
        if len(_tables) > 20000:
            _tables.clear()
        tabs = _build_tables(s, r, best_of)
        _tables[key] = tabs
    set_tab, match_tab = tabs
    if in_tb:
        first = _tb_first_server(server, p1 + p2)
        p_set = p_win_tb(p1, p2, first == 1, s, r)
    else:
        p_point = s if server == 1 else r
        p_game = p_win_game(p1, p2, p_point)
        p_set = p_game * set_tab[(g1 + 1, g2, 3 - server)] + (1 - p_game) * set_tab[(g1, g2 + 1, 3 - server)]

    def m(a: int, b: int) -> float:
        if a >= need:
            return 1.0
        if b >= need:
            return 0.0
        return match_tab[(a, b)]

    return p_set * m(s1 + 1, s2) + (1 - p_set) * m(s1, s2 + 1)


# --------------------------------------------------------------------------------------
# Dynamic serve probabilities (Kovalchik & Reid 2019 empirical-Bayes update)
# --------------------------------------------------------------------------------------


def update_serve(prior: float, n: int, won: int, kappa: float) -> float:
    """Shrink the observed in-match serve point win rate toward the prior.

    prior: pre-match P(server wins a point on serve); n/won: serve points played/won
    so far in this match; kappa: pseudo-count (large = trust the prior).
    """
    if kappa <= 0:
        return won / n if n else prior
    return (won + kappa * prior) / (n + kappa)


def kappa_schedule(progress: float, early: float = 640.0, mid: float = 160.0, late: float = 40.0) -> float:
    """Phase-dependent pseudo-count, log-linear between 25%, 50%, 75% of the match.

    Defaults are the values selected in the 2026 Trace study; the replay harness
    refits them per tour.
    """
    import math

    progress = min(max(progress, 0.0), 1.0)
    if progress <= 0.25:
        return early
    if progress >= 0.75:
        return late
    if progress <= 0.5:
        t = (progress - 0.25) / 0.25
        return math.exp(math.log(early) + t * (math.log(mid) - math.log(early)))
    t = (progress - 0.5) / 0.25
    return math.exp(math.log(mid) + t * (math.log(late) - math.log(mid)))


# --------------------------------------------------------------------------------------
# Elo-derived serve/return probabilities: invert the chain so the pre-match match
# probability equals the Elo (or market) probability.
# --------------------------------------------------------------------------------------


@lru_cache(maxsize=4096)
def serve_return_from_match_prob(p_match: float, tour_serve: float = 0.64, best_of: int = 3) -> tuple[float, float]:
    """Find (p_serve, p_return) for player 1 with a common shift delta so that
    P(P1 wins from 0-0) == p_match. p_serve = tour_serve + d, p_return = (1 - tour_serve) + d.
    """
    target = min(max(p_match, 0.001), 0.999)
    need = best_of // 2 + 1

    def prob(d: float) -> float:
        s = min(max(tour_serve + d, 0.02), 0.98)
        r = min(max(1.0 - tour_serve + d, 0.02), 0.98)
        a = p_win_match(0, 0, 0, 0, 0, 0, True, s, r, need)
        b = p_win_match(0, 0, 0, 0, 0, 0, False, s, r, need)
        return 0.5 * (a + b)

    lo, hi = -0.30, 0.30
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if prob(mid) < target:
            lo = mid
        else:
            hi = mid
    d = 0.5 * (lo + hi)
    return min(max(tour_serve + d, 0.02), 0.98), min(max(1.0 - tour_serve + d, 0.02), 0.98)


def next_branches(
    sets_1: int,
    sets_2: int,
    games_1: int,
    games_2: int,
    p1_serves: bool,
    p_serve: float,
    p_return: float,
    best_of: int = 3,
) -> dict[str, float]:
    """If this game ends now: P(match | P1 holds/breaks) from 0-0 next game."""
    need = best_of // 2 + 1
    if p1_serves:
        hold = p_win_match(sets_1, sets_2, games_1 + 1, games_2, 0, 0, False, p_serve, p_return, need)
        broken = p_win_match(sets_1, sets_2, games_1, games_2 + 1, 0, 0, False, p_serve, p_return, need)
        return {"if_p1_wins_game": hold, "if_p1_loses_game": broken}
    broken = p_win_match(sets_1, sets_2, games_1 + 1, games_2, 0, 0, True, p_serve, p_return, need)
    hold = p_win_match(sets_1, sets_2, games_1, games_2 + 1, 0, 0, True, p_serve, p_return, need)
    return {"if_p1_wins_game": broken, "if_p1_loses_game": hold}
