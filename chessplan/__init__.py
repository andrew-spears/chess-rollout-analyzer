"""chessplan — extract the plan behind a chess move by contrasting rollouts.

    from chessplan import SetupBoard, analyze, show, to_frame
    b = SetupBoard(); b.play("e4", "e5", "Nf3", "Nc6")   # click pieces, or play()
    rows, A, B = analyze(b.board, "c3", "Nc3", n=30)
    show(b.board, rows, A, B)
    to_frame(rows)
"""

import random

import chess

from .board import SetupBoard
from .contrast import N_ROLLOUTS, contrast, relevance
from .engine import (DEPTH, MULTIPV, TEMPERATURE, find_engine, move_token,
                     open_engine, rest_of, score_moves, softmax_sample, top_moves)
from .rollout import HORIZON, rollout_times
from .viz import buckets, gap_chart, plan_boards, show, to_frame

__all__ = [
    "SetupBoard", "analyze", "show", "to_frame", "plan_boards", "gap_chart",
    "buckets", "contrast", "relevance", "rollout_times", "open_engine",
    "find_engine", "top_moves", "score_moves", "rest_of", "softmax_sample",
    "move_token", "DEPTH", "MULTIPV", "TEMPERATURE", "HORIZON", "N_ROLLOUTS",
]


def parse_move(board, move):
    """'c3' (SAN) or 'c2c3' (UCI) or a chess.Move -> chess.Move."""
    if isinstance(move, chess.Move):
        return move
    text = str(move).strip()
    for parse in (board.parse_san, board.parse_uci):
        try:
            return parse(text)
        except ValueError:
            continue
    raise ValueError(f"illegal or unparseable move in this position: {text!r}")


def analyze(board, a, b, n=N_ROLLOUTS, horizon=HORIZON, depth=DEPTH,
            multipv=MULTIPV, temp=TEMPERATURE, engine_path=None, seed=None):
    """Contrast two candidate moves in `board`. Returns (rows, move_a, move_b).

    `a` and `b` are single moves, SAN or UCI. Each row scores a downstream move
    by `relevance` under each side: 1.0 = played immediately in every rollout,
    0.0 = never played within the horizon. `delta` = rel_A - rel_B.

    `temp` (pawns) and `horizon` change what the numbers mean; `n`, `depth` and
    `multipv` trade cost against noise.
    """
    rng = random.Random(seed) if seed is not None else random
    move_a, move_b = parse_move(board, a), parse_move(board, b)
    with open_engine(engine_path) as engine:
        rows = contrast(engine, board, move_a, move_b, n=n, horizon=horizon,
                        depth=depth, multipv=multipv, temp=temp, rng=rng)
    return rows, move_a, move_b
