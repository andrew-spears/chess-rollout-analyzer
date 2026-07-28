# chess-rollout-analyzer

Compare two candidate chess moves by **how soon each downstream move gets played
from the resulting positions**.

Pick two legal moves, A and B, from the same position. Force each one, then let
a policy (e.g. Stockfish) play many rollouts from each resulting position. For every move that shows up downstream, measure the expected number of moves until that move is played.

A score is the inverse of this expected number of moves until played, so that higher scores mean "played sooner."

```
score = mean over rollouts of  1 / (moves until first played)
```

counting **0** for any rollout where the move was never played.

| score  | meaning                              |
| ------ | ------------------------------------ |
| `1.00` | played immediately, in every rollout |
| `0.50` | played on your 2nd move, on average  |
| `0.00` | never played within the horizon      |

Every move is scored twice, once after A and once after B. We then sort all downstream moves in 3 ways:

By score after A − score after B (most unique to A)
By min(score after A, score after B) (most common to both)
By score after B − score after A (most unique to B)

## Install

```bash
pip install -e .
brew install stockfish        # or: apt install stockfish
```

Stockfish is found via the `engine_path` argument, else `$STOCKFISH_PATH`, else
`stockfish` on your `PATH`.

## Use

Open [analysis.ipynb](analysis.ipynb) and run top to bottom. Set the position up
by **clicking pieces on the board**, name two moves, and you get three panels
plus a table.

```python
from src import SetupBoard, analyze, show, to_frame

b = SetupBoard()
b.play("e4", "e5", "Nf3", "Nc6")      # or click on the board
b                                      # displays the interactive board

rows, A, B = analyze(b.board, "c3", "Nc3", n=30, horizon=14, depth=6, seed=0)
show(b.board, rows, A, B)
to_frame(rows, b.board.san(A), b.board.san(B))
```

On the Ponziani position, `Qa4`, `d3` and `Nd2` come sooner after `c3`, while
`Bb5`, `Nd4` and `Nd5` come sooner after `Nc3`. Moves like `d4` and `d5` arrive
at about the same time in both.

## Reading the output

`show()` prints three panels:

1. **Played sooner after A** — diff greater than `band`
2. **Played about as soon after both** — |diff| within `band` (default `0.05`)
3. **Played sooner after B** — diff below `−band`

Each panel is a board beside its own bars. The board shows that candidate played
(green arrow) with coloured arrows to the moves listed next to it; arrows are
only drawn from moves actually seen in _that_ candidate's rollouts. Each bar's
length is the move's score **in that line**, with the score difference in
parentheses.

`to_frame()` returns the same data with spelled-out column names: `score after
c3`, `score diff (c3 - Nc3)`, `% of rollouts played after c3`, and so on. The
`%` columns are how often a move appeared **at all**, which is worth checking
against the score — a low score can mean "played late" or "rarely played," and
these columns tell you which.

## Knobs

Two change what the numbers **mean**; the rest trade cost against noise.

| knob      | default | effect                                                                                                                          |
| --------- | ------- | ------------------------------------------------------------------------------------------------------------------------------- |
| `temp`    | `0.6`   | **Meaning.** In _pawns_. How loosely the engine explores.                                                                       |
| `horizon` | `14`    | **Meaning.** Plies rolled out. Anything played later scores 0, so this sets how far "soon" reaches.                             |
| `n`       | `70`    | Cost. Rollouts per candidate; more is just less noise.                                                                          |
| `depth`   | `8`     | Cost. Engine search depth per move.                                                                                             |
| `multipv` | `3`     | Candidates the engine returns each step — the pool the sampler draws from. `1` makes rollouts deterministic and ignores `temp`. |
| `seed`    | `None`  | Reproducibility.                                                                                                                |

## Layout

```
src/
  engine.py     engine lifecycle + policy (top_moves, softmax_sample, move_token)
  rollout.py    rollout_times — one playout to {move: first own-move ordinal}
  contrast.py   contrast + the pure score_from_ordinals helper
  board.py      SetupBoard — the click-to-move position editor
  viz.py        the three panels and to_frame
analysis.ipynb  the front end
tests/          run without a Stockfish binary (stub engine)
```

## Tests

```bash
pytest -q
```

The pure pieces, the widget, and the whole rollout/contrast pipeline are tested
against a stub engine, so the suite runs with no Stockfish installed. One
integration test uses the real engine and skips when the binary is absent.

## Possible next step: human-game policy

Rollouts could sample from what humans actually play instead of what Stockfish
prefers, using the free [Lichess opening
explorer](https://explorer.lichess.ovh) — no API key, one HTTP request per
position, filterable by rating band and time control. Cached to disk it would be
faster than the engine. The catch is depth: past roughly 10–12 plies most
positions have too few games to sample from, so it needs either a fallback to
the engine or an early stop.

## License

MIT
