# chessplan

**What is the *plan* behind a chess move?** Pick two candidate moves from the
same position. Force each one, let Stockfish play out the rest many times, and
see which follow-up moves show up **sooner and more often** after A than after B.

Every downstream move gets a **relevance** score under each candidate:

> `relevance` = average of `1 / (own moves until you first play it)`,
> scoring **0** when you never play it within the horizon.

So `1.0` = played immediately in every rollout, `0` = never played. Higher =
more central to that plan. `delta = rel_A − rel_B` says which candidate owns it.

## Install

```bash
pip install -e .
brew install stockfish        # or: apt install stockfish
```

Stockfish is found via the `engine_path` argument, else `$STOCKFISH_PATH`, else
`stockfish` on your `PATH`.

## Use

Open [analysis.ipynb](analysis.ipynb) and run top to bottom. Set the position up
by **clicking pieces on the board**, pick two moves, get annotated boards, a
grouped bar chart, and a DataFrame.

```python
from chessplan import SetupBoard, analyze, show, to_frame

b = SetupBoard()
b.play("e4", "e5", "Nf3", "Nc6")      # or click on the board
b                                      # displays the interactive board

rows, A, B = analyze(b.board, "c3", "Nc3", n=30, horizon=14, depth=6, seed=0)
show(b.board, rows, A, B)
to_frame(rows)
```

On the Ponziani position, `c3` plays for `Qa4`, `d3`, `Nd2` (the c-pawn line
wants the a4 pin and a solid centre) while `Nc3` plays for `Nd4`, `Nd5`, `Bb5`
(pieces first). Moves like `d4` and `d5` come out **common to both**.

## Knobs

Two change what the numbers **mean**; the rest trade cost against noise.

| knob | default | effect |
|---|---|---|
| `temp` | `0.6` | **Meaning.** In *pawns*. How loosely the engine explores. |
| `horizon` | `14` | **Meaning.** Plies rolled out — how far a "plan" may reach. |
| `n` | `70` | Cost. Rollouts per candidate; more is just less noise. |
| `depth` | `8` | Cost. Engine search depth per move. |
| `multipv` | `3` | Candidates the engine considers each step — the pool the sampler draws from. `1` makes rollouts deterministic and ignores `temp`. |
| `seed` | `None` | Reproducibility. |

## Reading the output

- **Boards** — each candidate played (green arrow), with coloured arrows to that
  plan's strongest follow-ups. Arrows are drawn only from moves actually seen in
  *that* candidate's rollouts.
- **Bars** — three groups: A's plan, common to both, B's plan. Bar length is the
  relevance gap.
- **DataFrame** — `rel_A`/`rel_B` plus `P_A`/`P_B`, how often each move appeared
  at all.

`relevance` deliberately blends *how soon* with *how often*: a move played at
once in a quarter of rollouts and never otherwise scores the same as one always
played on the 4th own move. That's fine for ranking plans, but don't read it as
a pure "how many moves away" number.

## Layout

```
chessplan/
  engine.py     engine lifecycle + policy (top_moves, softmax_sample, move_token)
  rollout.py    rollout_times — one playout to {move: first own-move ordinal}
  contrast.py   contrast + the pure relevance helper
  board.py      SetupBoard — the click-to-move position editor
  viz.py        annotated boards, grouped bar chart, to_frame
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

Instead of Stockfish's evaluation, rollouts could sample from what humans
actually play, using the free [Lichess opening explorer]
(https://explorer.lichess.ovh/lichess?fen=...) — no API key, one HTTP request
per position, filterable by rating band and time control. Cache responses to
disk and it is faster than the engine. The catch is depth: past roughly 10–12
plies most positions have too few games to sample from, so it needs either a
fallback to the engine or an early stop.

## License

MIT
