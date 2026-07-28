"""Tests run WITHOUT a Stockfish binary: pure pieces directly, pipeline via a stub."""

import random
import shutil

import chess
import chess.engine
import pytest

from src import (contrast, move_token, score_from_ordinals, rollout_times,
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


def test_censor_is_on_the_own_move_scale():
    from src.contrast import censor_for
    # horizon is in plies; the mover gets (horizon+1)//2 own moves, censor is 1 past
    assert censor_for(14) == 8
    assert censor_for(6) == 4
    assert censor_for(1) == 2


def test_score_edges():
    # 1/mean-ordinal: average first, then invert
    # played immediately in every rollout -> 1/1 = 1.0
    assert score_from_ordinals(ordinal_sum=10, occ=10, n=10, censor=8) == 1.0
    # always played on the mover's 2nd move -> 1/2 = 0.5
    assert score_from_ordinals(ordinal_sum=20, occ=10, n=10, censor=8) == 0.5
    # never played -> mean is the censor -> 1/8, the floor (NOT zero)
    assert score_from_ordinals(ordinal_sum=0, occ=0, n=10, censor=8) == 1 / 8
    # half played at once, half never -> mean (5*1 + 5*8)/10 = 4.5
    assert score_from_ordinals(ordinal_sum=5, occ=5, n=10, censor=8) == 1 / 4.5


def test_score_is_bounded_by_one_and_the_floor():
    for occ in range(0, 11):
        s = score_from_ordinals(ordinal_sum=occ, occ=occ, n=10, censor=8)
        assert 1 / 8 <= s <= 1.0


def test_score_rejects_bad_inputs():
    with pytest.raises(ValueError):
        score_from_ordinals(ordinal_sum=1, occ=1, n=0, censor=8)
    with pytest.raises(ValueError):
        score_from_ordinals(ordinal_sum=1, occ=11, n=10, censor=8)


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


def test_contrast_scores_bounded_and_diff_consistent():
    rows = contrast(StubEngine(), chess.Board(),
                    chess.Move.from_uci("e2e4"), chess.Move.from_uci("c2c4"),
                    n=3, horizon=6, rng=random.Random(0))
    floor = 1 / ((6 + 1) // 2 + 1)  # censor_for(horizon=6)
    assert rows, "expected some downstream moves"
    for r in rows:
        assert floor <= r["score_A"] <= 1.0 and floor <= r["score_B"] <= 1.0
        assert r["diff"] == pytest.approx(r["score_A"] - r["score_B"])
    # rows come back sorted by diff, largest first
    assert rows == sorted(rows, key=lambda r: r["diff"], reverse=True)
    # a move never played under one side sits exactly at the floor there
    assert any(r["score_A"] == floor or r["score_B"] == floor for r in rows)


def test_contrast_arrow_squares_are_per_side():
    rows = contrast(StubEngine(), chess.Board(),
                    chess.Move.from_uci("e2e4"), chess.Move.from_uci("c2c4"),
                    n=2, horizon=8, rng=random.Random(0))
    floor = 1 / ((8 + 1) // 2 + 1)  # censor_for(horizon=8)
    for r in rows:
        # a side with no sightings sits at the floor and carries no squares
        if r["score_A"] == floor:
            assert r["_sq_A"] is None
        if r["score_B"] == floor:
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


def test_buckets_rank_by_diff_and_by_min():
    from src import buckets
    rows = [{"move": "onlyA", "score_A": .9, "score_B": .1, "diff": .8},
            {"move": "bothHigh", "score_A": .7, "score_B": .6, "diff": .1},
            {"move": "bothLow", "score_A": .2, "score_B": .2, "diff": .0},
            {"move": "onlyB", "score_A": .1, "score_B": .8, "diff": -.7}]
    a, common, b = buckets(rows)
    # A ranked by score_A - score_B, B by the reverse
    assert [r["move"] for r in a][:2] == ["onlyA", "bothHigh"]
    assert [r["move"] for r in b][:1] == ["onlyB"]
    # common ranked by min(score_A, score_B) -- a lopsided move must not win
    assert [r["move"] for r in common][:2] == ["bothHigh", "bothLow"]
    assert common[0]["move"] != "onlyA"


def test_buckets_are_rankings_not_a_partition():
    from src import buckets
    # every row appears in all three rankings; they are top-N views, not buckets
    rows = [{"move": "x", "score_A": .5, "score_B": .4, "diff": .1},
            {"move": "y", "score_A": .3, "score_B": .9, "diff": -.6}]
    a, common, b = buckets(rows)
    assert len(a) == len(common) == len(b) == len(rows)


def test_panels_top_controls_how_many_moves_are_drawn():
    from src import panels
    board = chess.Board()
    rows = contrast(StubEngine(), board, chess.Move.from_uci("e2e4"),
                    chess.Move.from_uci("c2c4"), n=2, horizon=8,
                    rng=random.Random(0))
    a = chess.Move.from_uci("e2e4")
    b = chess.Move.from_uci("c2c4")
    # 2 bars per move x 3 panels
    for n in (1, 3):
        assert panels(board, rows, a, b, top=n).data.count('rx="2"') == 6 * n


def test_show_interactive_slider_rerenders():
    from src import show_interactive
    board = chess.Board()
    rows = contrast(StubEngine(), board, chess.Move.from_uci("e2e4"),
                    chess.Move.from_uci("c2c4"), n=2, horizon=8,
                    rng=random.Random(0))
    box = show_interactive(board, rows, chess.Move.from_uci("e2e4"),
                           chess.Move.from_uci("c2c4"), top=2)
    controls, view = box.children
    slider, picker = controls.children
    assert view.value.count('rx="2"') == 12
    slider.value = 4          # simulate the user dragging it
    assert view.value.count('rx="2"') == 24

    # switching the metric re-renders too, and changes the numbers
    from src import METRICS
    speed_html = view.value
    picker.value = "frequency"
    assert view.value != speed_html
    assert METRICS["frequency"][1] in view.value   # its definition is shown
    picker.value = "speed"
    assert view.value == speed_html   # switching back is lossless


def test_show_interactive_slider_caps_at_max_top():
    from src import show_interactive
    board = chess.Board()
    rows = contrast(StubEngine(), board, chess.Move.from_uci("e2e4"),
                    chess.Move.from_uci("c2c4"), n=2, horizon=10,
                    rng=random.Random(0))
    a, b = chess.Move.from_uci("e2e4"), chess.Move.from_uci("c2c4")
    assert len(rows) > 3, "need more rows than the cap for this to mean anything"
    slider = show_interactive(board, rows, a, b, max_top=3).children[0].children[0]
    assert slider.max == 3          # capped, not raised to len(rows)
    # but never above the number of moves actually available
    slider = show_interactive(board, rows[:2], a, b, max_top=9).children[0].children[0]
    assert slider.max == 2


def test_scale_grows_the_rendering():
    from src import panels
    board = chess.Board()
    rows = contrast(StubEngine(), board, chess.Move.from_uci("e2e4"),
                    chess.Move.from_uci("c2c4"), n=1, horizon=6,
                    rng=random.Random(0))
    a, b = chess.Move.from_uci("e2e4"), chess.Move.from_uci("c2c4")
    small = panels(board, rows, a, b, top=2, scale=1.0).data
    big = panels(board, rows, a, b, top=2, scale=2.0).data
    assert '<svg width="360"' in small and '<svg width="720"' in big
    assert "font:12px" in small and "font:24px" in big


def test_with_metric_reinterprets_without_touching_the_engine():
    from src import with_metric
    rows = [{"move": "d4", "score_A": .5, "score_B": .25,
             "P_A": 0.8, "P_B": 0.1, "diff": .25}]
    freq = with_metric(rows, "frequency")
    assert freq[0]["score_A"] == 0.8 and freq[0]["score_B"] == 0.1
    assert freq[0]["diff"] == pytest.approx(0.7)
    # the original rows are untouched, so switching back recovers them
    assert rows[0]["score_A"] == .5
    assert with_metric(rows, "speed")[0]["score_A"] == .5


def test_panels_rejects_unknown_metric():
    from src import panels
    board = chess.Board()
    rows = contrast(StubEngine(), board, chess.Move.from_uci("e2e4"),
                    chess.Move.from_uci("c2c4"), n=1, horizon=6,
                    rng=random.Random(0))
    with pytest.raises(ValueError, match="metric must be one of"):
        panels(board, rows, chess.Move.from_uci("e2e4"),
               chess.Move.from_uci("c2c4"), metric="nonsense")


# ---------------------------------------------------------------- widget
def test_setup_board_moves_undo_reset():
    from src import SetupBoard
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
    from src import SetupBoard
    b = SetupBoard(chess.Board("8/P6k/8/8/8/8/8/7K w - - 0 1"))
    b.attempt = "a7a8:1"
    assert b.board.move_stack[-1].uci() == "a7a8q"


# ---------------------------------------------------------------- integration
@pytest.mark.skipif(shutil.which("stockfish") is None,
                    reason="needs a real Stockfish binary")
def test_ponziani_integration_tiny():
    from src import analyze, to_frame
    board = chess.Board()
    for mv in ["e2e4", "e7e5", "g1f3", "b8c6"]:
        board.push_uci(mv)
    rows, a, b = analyze(board, "c3", "Nc3", n=2, horizon=6, depth=4, seed=0)
    assert rows and a.uci() == "c2c3" and b.uci() == "b1c3"
    df = to_frame(rows, "c3", "Nc3")
    assert list(df.columns) == [
        "move", "score after c3", "score after Nc3", "score diff (c3 - Nc3)",
        "% of rollouts played after c3", "% of rollouts played after Nc3"]
    assert not any(c.startswith("_") for c in df.columns)
