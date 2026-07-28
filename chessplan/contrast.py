"""Contrast layer: two candidate moves A, B -> how central each downstream move
is to each one's plan.

For each rollout a move C gets the ordinal of the mover's turn on which it first
appears (1 = the very next own move). Averaging that raw ordinal needs a
censoring convention for rollouts where C never showed up, which makes the
number hard to read. Instead we report

    relevance(C) = mean over rollouts of  1/ordinal,  scoring 0 when C
                   never appeared within the horizon.

So relevance is in [0, 1]: 1.0 = played immediately in every rollout, 0 = never
played. Higher = more central to the plan. It still blends "how soon" with "how
often" -- that is the intent, and it is fine for ranking.

    delta = relevance_A - relevance_B   >0 => A's plan;  <0 => B's plan.
"""

import random
from collections import Counter

from .engine import DEPTH, MULTIPV, TEMPERATURE, move_token
from .rollout import HORIZON, rollout_times

N_ROLLOUTS = 70


def relevance(recip_sum, n):
    """Mean of 1/ordinal over ALL n rollouts; absences contribute 0.

    Pure -- no engine, no randomness. 1.0 = played immediately every time,
    0.0 = never played.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    return recip_sum / n


def _aggregate(engine, board, first_move, side, n, horizon, squares=None, **kw):
    """relevance + appearance frequency per token, over `n` rollouts."""
    recip, occ = Counter(), Counter()
    for _ in range(n):
        for tok, ordn in rollout_times(engine, board, first_move, side,
                                       horizon, squares=squares, **kw).items():
            recip[tok] += 1.0 / ordn
            occ[tok] += 1
    R = {t: relevance(recip[t], n) for t in recip}
    freq = {t: occ[t] / n for t in recip}
    return R, freq


def contrast(engine, board, move_a, move_b, n=N_ROLLOUTS, horizon=HORIZON,
             depth=DEPTH, multipv=MULTIPV, temp=TEMPERATURE, rng=random):
    """Contrast two candidate moves from the same position. Returns rows.

    Each row is a downstream move with its relevance under A and under B, and
    delta = relevance_A - relevance_B.
    """
    for label, m in (("A", move_a), ("B", move_b)):
        if m not in board.legal_moves:
            raise ValueError(
                f"move {label}: {m.uci()} is not legal in position {board.fen()}"
            )
    if move_a == move_b:
        raise ValueError("A and B are the same move")

    side = board.turn
    kw = dict(depth=depth, multipv=multipv, temp=temp, rng=rng)
    # squares are tracked per side: an arrow drawn on A's board must come from
    # a move actually seen in A's rollouts, not B's.
    sq_a, sq_b = {}, {}
    RA, FA = _aggregate(engine, board, move_a, side, n, horizon, sq_a, **kw)
    RB, FB = _aggregate(engine, board, move_b, side, n, horizon, sq_b, **kw)

    forced = {move_token(board, move_a), move_token(board, move_b)}
    rows = []
    for t in (set(RA) | set(RB)) - forced:
        ra, rb = RA.get(t, 0.0), RB.get(t, 0.0)
        rows.append({
            "move": t, "rel_A": ra, "rel_B": rb,
            "P_A": FA.get(t, 0.0), "P_B": FB.get(t, 0.0),
            "delta": ra - rb,  # >0 => more central to A's plan
            "_sq_A": sq_a.get(t), "_sq_B": sq_b.get(t),
        })
    return sorted(rows, key=lambda r: r["delta"], reverse=True)
