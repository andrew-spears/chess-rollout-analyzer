"""Tests run WITHOUT a Stockfish binary: pure pieces directly, pipeline via a stub."""

import random
import shutil

import chess
import chess.engine
import pytest

from chessplan import (contrast, move_token, relevance, rollout_times,
                       softmax_sample)


# ---------------------------------------------------------------- pure pieces
def test_move_token():
    b = chess.Board()
    assert move_token(b, chess.Move.from_uci("e2e4")) == "e4"
    assert move_token(b, chess.Move.from_uci("g1f3")) == "Nf3"
    assert move_token(b, chess.Move.from_uci("b1c3")) == "Nc3"


def test_softmax_sample_singleton():
    only = chess.Move.from_uci("e2e4")
    assert softmax_sample([(only, -30)]) is only


def test_softmax_sample_skews_to_better_at_low_temperature():
    good, bad = chess.Move.from_uci("e2e4"), chess.Move.from_uci("a2a3")
    scored = [(good, 60), (bad, -60)]
    rng = random.Random(0)
    picks = [softmax_sample(scored, temperature=0.1, rng=rng) for _ in range(200)]
    assert picks.count(good) > 190


def test_relevance_edges():
    # never appeared -> 0
    assert relevance(recip_sum=0.0, n=10) == 0.0
    # played immediately (ordinal 1) in every rollout -> 1.0
    assert relevance(recip_sum=10.0, n=10) == 1.0
    # always played on the mover's 2nd move -> 0.5
    assert relevance(recip_sum=10 * 0.5, n=10) == 0.5
    # played at once in half the rollouts, never in the rest -> 0.5
    assert relevance(recip_sum=5.0, n=10) == 0.5


# ---------------------------------------------------------------- stub engine
class StubEngine:
    """Deterministic canned policy: prefers d2d4/d7d5 when available, else the
    first legal move in python-chess's ordering."""

    def analyse(self, board, limit, multipv=1, root_moves=None):
        moves = list(root_moves) if root_moves else list(board.legal_moves)
        preferred = [m for m in moves if m.uci() in ("d2d4", "d7d5")]
        ordered = preferred + [m for m in moves if m not in preferred]
        return [{"pv": [m], "score": chess.engine.PovScore(
                    chess.engine.Cp(50 - 10 * i), board.turn)}
                for i, m in enumerate(ordered[:max(multipv, 1)])]


def test_rollout_forced_move_gets_ordinal_one():
    times = rollout_times(StubEngine(), chess.Board(),
                          chess.Move.from_uci("e2e4"), chess.WHITE,
                          horizon=6, rng=random.Random(0))
    assert times["e4"] == 1


def test_rollout_never_played_move_is_absent():
    times = rollout_times(StubEngine(), chess.Board(),
                          chess.Move.from_uci("e2e4"), chess.WHITE,
                          horizon=6, rng=random.Random(0))
    assert "Rh6" not in times  # a rook move the stub will never produce


def test_rollout_records_from_and_to_squares():
    squares = {}
    times = rollout_times(StubEngine(), chess.Board(),
                          chess.Move.from_uci("e2e4"), chess.WHITE, horizon=6,
                          rng=random.Random(0), squares=squares)
    assert squares["e4"] == (chess.E2, chess.E4)
    assert set(squares) == set(times)


def test_contrast_relevance_is_bounded_and_delta_consistent():
    rows = contrast(StubEngine(), chess.Board(),
                    chess.Move.from_uci("e2e4"), chess.Move.from_uci("c2c4"),
                    n=3, horizon=6, rng=random.Random(0))
    assert rows, "expected some downstream moves"
    for r in rows:
        assert 0.0 <= r["rel_A"] <= 1.0 and 0.0 <= r["rel_B"] <= 1.0
        assert r["delta"] == pytest.approx(r["rel_A"] - r["rel_B"])
    # rows come back sorted strongest-A-plan first
    assert rows == sorted(rows, key=lambda r: r["delta"], reverse=True)
    # a move never played under one side scores exactly 0 there
    assert any(r["rel_A"] == 0.0 or r["rel_B"] == 0.0 for r in rows)


def test_contrast_arrow_squares_are_per_side():
    rows = contrast(StubEngine(), chess.Board(),
                    chess.Move.from_uci("e2e4"), chess.Move.from_uci("c2c4"),
                    n=2, horizon=8, rng=random.Random(0))
    for r in rows:
        # a side with no sightings must carry no squares for that side
        if r["rel_A"] == 0.0:
            assert r["_sq_A"] is None
        if r["rel_B"] == 0.0:
            assert r["_sq_B"] is None
        for key in ("_sq_A", "_sq_B"):
            if r[key] is not None:
                frm, to = r[key]
                assert 0 <= frm < 64 and 0 <= to < 64


def test_contrast_excludes_the_two_forced_moves():
    rows = contrast(StubEngine(), chess.Board(),
                    chess.Move.from_uci("e2e4"), chess.Move.from_uci("c2c4"),
                    n=2, horizon=6, rng=random.Random(0))
    assert {"e4", "c4"}.isdisjoint({r["move"] for r in rows})


def test_contrast_rejects_illegal_and_identical_moves():
    board = chess.Board()
    with pytest.raises(ValueError, match="not legal"):
        contrast(StubEngine(), board, chess.Move.from_uci("e2e5"),
                 chess.Move.from_uci("c2c4"), n=1, horizon=4)
    with pytest.raises(ValueError, match="same move"):
        contrast(StubEngine(), board, chess.Move.from_uci("e2e4"),
                 chess.Move.from_uci("e2e4"), n=1, horizon=4)


def test_buckets_split_by_band():
    from chessplan import buckets
    rows = [{"move": "d4", "rel_A": .9, "rel_B": .1, "delta": .8},
            {"move": "Nf3", "rel_A": .5, "rel_B": .5, "delta": .0},
            {"move": "Bb5", "rel_A": .1, "rel_B": .7, "delta": -.6}]
    a, common, b = buckets(rows, band=0.05)
    assert [r["move"] for r in a] == ["d4"]
    assert [r["move"] for r in common] == ["Nf3"]
    assert [r["move"] for r in b] == ["Bb5"]


# ---------------------------------------------------------------- widget
def test_setup_board_moves_undo_reset():
    from chessplan import SetupBoard
    b = SetupBoard()
    b.attempt = "e2e4:1"
    b.attempt = "e7e5:2"
    assert [m.uci() for m in b.board.move_stack] == ["e2e4", "e7e5"]
    b.action = "undo:1"
    assert [m.uci() for m in b.board.move_stack] == ["e2e4"]
    b.action = "reset:2"
    assert b.board.move_stack == []
    b.action = "undo:3"  # undo on an empty board must not raise
    assert b.board.move_stack == []
    b.attempt = "a1a8:4"  # illegal -> rejected, board unchanged
    assert b.board.move_stack == []
    assert "not legal" in b.status


def test_setup_board_auto_queens_promotions():
    from chessplan import SetupBoard
    b = SetupBoard(chess.Board("8/P6k/8/8/8/8/8/7K w - - 0 1"))
    b.attempt = "a7a8:1"
    assert b.board.move_stack[-1].uci() == "a7a8q"


# ---------------------------------------------------------------- integration
@pytest.mark.skipif(shutil.which("stockfish") is None,
                    reason="needs a real Stockfish binary")
def test_ponziani_integration_tiny():
    from chessplan import analyze, to_frame
    board = chess.Board()
    for mv in ["e2e4", "e7e5", "g1f3", "b8c6"]:
        board.push_uci(mv)
    rows, a, b = analyze(board, "c3", "Nc3", n=2, horizon=6, depth=4, seed=0)
    assert rows and a.uci() == "c2c3" and b.uci() == "b1c3"
    df = to_frame(rows)
    assert {"move", "rel_A", "rel_B", "delta"} <= set(df.columns)
    assert not any(c.startswith("_") for c in df.columns)
