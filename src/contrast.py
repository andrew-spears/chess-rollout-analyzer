"""Contrast layer: two candidate moves A, B -> how soon each downstream move
gets played after each of them.

For each rollout a move C gets the ordinal of the mover's turn on which it first
appears (1 = the very next own move). Rollouts where C never showed up are
CENSORED at M+1, one past the last own move the horizon allows. Then

    score(C) = 1 / (mean ordinal over all n rollouts)

The average is taken first, then inverted. Score is in (0, 1]: 1.0 = played
immediately in every rollout, 1/(M+1) = never played at all -- a floor, not
zero. It deliberately blends "how soon" with "how often", since a rollout that
never plays C drags the mean toward the censor.

    diff = score_A - score_B   >0 => played sooner after A;  <0 => after B.

This says nothing about *why* a move follows; it only measures when it shows up.
"""

import random
from collections import Counter

from .engine import DEPTH, MULTIPV, TEMPERATURE, move_token
from .rollout import HORIZON, rollout_times

N_ROLLOUTS = 70


def censor_for(horizon):
    """One past the last own move the horizon allows: M+1, where M=(horizon+1)//2.

    Ordinals count the mover's OWN turns, so the censor lives on that scale too,
    not on the ply scale of `horizon`.
    """
    return (horizon + 1) // 2 + 1


def score_from_ordinals(ordinal_sum, occ, n, censor):
    """1 / mean-ordinal, where the (n - occ) rollouts that never played the move
    contribute `censor` to the mean. Average first, then invert.

    Pure -- no engine, no randomness. 1.0 = played immediately every time,
    1/censor = never played.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if occ > n:
        raise ValueError("occ cannot exceed n")
    mean_ordinal = (ordinal_sum + (n - occ) * censor) / n
    return 1.0 / mean_ordinal


def _aggregate(engine, board, first_move, side, n, horizon, censor,
               squares=None, **kw):
    """score + appearance frequency per token, over `n` rollouts."""
    total, occ = Counter(), Counter()
    for _ in range(n):
        for tok, ordn in rollout_times(engine, board, first_move, side,
                                       horizon, squares=squares, **kw).items():
            total[tok] += ordn
            occ[tok] += 1
    scores = {t: score_from_ordinals(total[t], occ[t], n, censor) for t in total}
    freq = {t: occ[t] / n for t in total}
    return scores, freq


def contrast(engine, board, move_a, move_b, n=N_ROLLOUTS, horizon=HORIZON,
             depth=DEPTH, multipv=MULTIPV, temp=TEMPERATURE, rng=random):
    """Contrast two candidate moves from the same position. Returns rows.

    Each row is a downstream move with its score after A and after B, and
    diff = score_A - score_B.
    """
    for label, m in (("A", move_a), ("B", move_b)):
        if m not in board.legal_moves:
            raise ValueError(
                f"move {label}: {m.uci()} is not legal in position {board.fen()}"
            )
    if move_a == move_b:
        raise ValueError("A and B are the same move")

    side = board.turn
    censor = censor_for(horizon)
    floor = 1.0 / censor  # score of a move that never appeared on that side
    kw = dict(depth=depth, multipv=multipv, temp=temp, rng=rng)
    # squares are tracked per side: an arrow drawn on A's board must come from
    # a move actually seen in A's rollouts, not B's.
    sq_a, sq_b = {}, {}
    SA, FA = _aggregate(engine, board, move_a, side, n, horizon, censor, sq_a, **kw)
    SB, FB = _aggregate(engine, board, move_b, side, n, horizon, censor, sq_b, **kw)

    forced = {move_token(board, move_a), move_token(board, move_b)}
    rows = []
    for t in (set(SA) | set(SB)) - forced:
        sa, sb = SA.get(t, floor), SB.get(t, floor)
        rows.append({
            "move": t, "score_A": sa, "score_B": sb,
            "P_A": FA.get(t, 0.0), "P_B": FB.get(t, 0.0),
            "diff": sa - sb,  # >0 => played sooner after A
            "_sq_A": sq_a.get(t), "_sq_B": sq_b.get(t),
        })
    return sorted(rows, key=lambda r: r["diff"], reverse=True)
