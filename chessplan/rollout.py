"""Rollout layer: position + horizon -> {token: first own-move ordinal it appeared}."""

import random

from .engine import DEPTH, MULTIPV, TEMPERATURE, move_token, softmax_sample, top_moves

HORIZON = 14  # plies; the mover gets (HORIZON+1)//2 own moves in that window


def rollout_times(engine, board, first_move, side, horizon=HORIZON,
                  depth=DEPTH, multipv=MULTIPV, temp=TEMPERATURE, rng=random,
                  squares=None):
    """Return {token: ordinal of the mover's turn on which it FIRST appeared}.

    Ordinal 1 = the forced first move, 2 = the mover's next own move, etc.
    If `squares` is a dict it is filled with {token: (from_square, to_square)}
    from each token's first appearance, for drawing arrows later.
    """
    b = board.copy()
    times = {}
    ordinal = 0

    def note(mv, ordn):
        tok = move_token(b, mv)
        if tok not in times:
            times[tok] = ordn
            if squares is not None:
                squares.setdefault(tok, (mv.from_square, mv.to_square))

    if b.turn == side:
        ordinal += 1
        note(first_move, ordinal)
    b.push(first_move)
    for _ in range(horizon - 1):
        if b.is_game_over():
            break
        mv = softmax_sample(top_moves(engine, b, multipv, depth), temp, rng)
        if b.turn == side:
            ordinal += 1
            note(mv, ordinal)
        b.push(mv)
    return times
