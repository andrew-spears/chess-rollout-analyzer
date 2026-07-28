"""Engine lifecycle and policy: position -> a sampled move (MultiPV + softmax)."""

import math
import os
import random
import shutil
from contextlib import contextmanager

import chess
import chess.engine

# Defaults. `temp` (pawns) and `horizon` change what the numbers MEAN;
# `depth`, `multipv` and `n` only change cost/noise. See README.
DEPTH = 8
MULTIPV = 3
TEMPERATURE = 0.6
MATE_CP = 100000


def find_engine(path=None):
    """--engine arg > $STOCKFISH_PATH > `stockfish` on PATH."""
    found = path or os.environ.get("STOCKFISH_PATH") or shutil.which("stockfish")
    if not found:
        raise FileNotFoundError(
            "Stockfish not found. Install it (`brew install stockfish` or "
            "`apt install stockfish`), set $STOCKFISH_PATH, or pass a path."
        )
    return found


@contextmanager
def open_engine(path=None):
    engine = chess.engine.SimpleEngine.popen_uci(find_engine(path))
    try:
        yield engine
    finally:
        engine.quit()


def top_moves(engine, board, k=MULTIPV, depth=DEPTH):
    infos = engine.analyse(board, chess.engine.Limit(depth=depth), multipv=k)
    return [(i["pv"][0], i["score"].pov(board.turn).score(mate_score=MATE_CP))
            for i in infos]


def score_moves(engine, board, moves, depth=DEPTH):
    moves = list(moves)
    infos = engine.analyse(board, chess.engine.Limit(depth=depth),
                           multipv=len(moves), root_moves=moves)
    return [(i["pv"][0], i["score"].pov(board.turn).score(mate_score=MATE_CP))
            for i in infos]


def rest_of(engine, board, exclude, k=MULTIPV, depth=DEPTH):
    """Policy top-k minus the excluded moves: realistic next-best play."""
    ex = set(exclude)
    pool = top_moves(engine, board, k + len(ex), depth)
    return [(m, cp) for m, cp in pool if m not in ex][:k]


def softmax_sample(scored, temperature=TEMPERATURE, rng=random):
    """Sample from [(move, cp), ...]; `temperature` is in pawns."""
    best = max(cp for _, cp in scored)
    scale = 100.0 * max(temperature, 1e-6)
    weights = [math.exp((cp - best) / scale) for _, cp in scored]
    r = rng.random() * sum(weights)
    acc = 0.0
    for (move, _), w in zip(scored, weights):
        acc += w
        if r <= acc:
            return move
    return scored[-1][0]


def move_token(board, move):
    """Move identity = (piece, destination), e.g. `Bg7`, `d4`. Coarse on purpose."""
    piece = board.piece_at(move.from_square)
    dest = chess.square_name(move.to_square)
    letter = piece.symbol().upper()
    return dest if letter == "P" else letter + dest
