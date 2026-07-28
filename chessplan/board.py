"""Drag-and-drop board widget for setting up the position to analyse.

    b = SetupBoard()          # displays; drag pieces to play moves
    b.board                   # live chess.Board, legality enforced Python-side
"""

import anywidget
import chess
import traitlets

_GLYPHS = {"P": "♙", "N": "♘", "B": "♗", "R": "♖",
           "Q": "♕", "K": "♔", "p": "♟", "n": "♞",
           "b": "♝", "r": "♜", "q": "♛", "k": "♚"}

_ESM = """
function render({ model, el }) {
  el.innerHTML = "";
  const wrap = document.createElement("div");
  wrap.style.cssText = "font-family:system-ui,sans-serif;display:inline-block";
  const grid = document.createElement("div");
  grid.style.cssText =
    "display:grid;grid-template-columns:repeat(8,54px);grid-template-rows:repeat(8,54px);" +
    "border:2px solid #3b3b46;border-radius:4px;overflow:hidden;user-select:none";
  const status = document.createElement("div");
  status.style.cssText = "margin-top:8px;font-size:13px;color:#8a8a99;min-height:18px";

  let seq = 0;   // monotonic, so every click is a distinct traitlet value
  let sel = null;  // selected source square index (0=a8, left-to-right, top-down)

  function send(name, value) {
    sel = null;
    model.set(name, value + ":" + (++seq));
    model.save_changes();
  }

  const bar = document.createElement("div");
  bar.style.cssText = "margin-top:8px;display:flex;gap:6px";
  for (const [label, action] of [["↶ Undo", "undo"], ["Reset", "reset"]]) {
    const btn = document.createElement("button");
    btn.textContent = label;
    btn.style.cssText =
      "padding:4px 10px;font-size:12px;border:1px solid #3b3b46;border-radius:4px;" +
      "background:#26262e;color:#d8d8e0;cursor:pointer";
    btn.onclick = () => send("action", action);
    bar.appendChild(btn);
  }

  function squareName(i) {
    return "abcdefgh"[i % 8] + (8 - Math.floor(i / 8));
  }

  function draw() {
    const pieces = model.get("squares");   // 64 glyph-or-empty strings, a8-first
    const legal = model.get("legal_from"); // square names with a legal move
    grid.innerHTML = "";
    for (let i = 0; i < 64; i++) {
      const name = squareName(i);
      const dark = (Math.floor(i / 8) + i) % 2 === 1;
      const sq = document.createElement("div");
      sq.style.cssText =
        "display:flex;align-items:center;justify-content:center;font-size:38px;line-height:1;" +
        "cursor:pointer;position:relative;background:" + (dark ? "#6d7f95" : "#c3cddb");
      if (sel === i) sq.style.background = "#b6c25c";
      sq.textContent = pieces[i] || "";
      sq.style.color = pieces[i] && pieces[i].codePointAt(0) < 0x265a ? "#fff" : "#111";
      sq.style.textShadow = "0 1px 1px rgba(0,0,0,.35)";
      if (!pieces[i] && !sel) sq.style.cursor = "default";

      sq.onclick = () => {
        if (sel === null) {
          if (legal.includes(name)) { sel = i; draw(); }
        } else if (sel === i) {
          sel = null; draw();
        } else {
          send("attempt", squareName(sel) + name);
        }
      };
      grid.appendChild(sq);
    }
    status.textContent = model.get("status");
  }

  // `version` bumps on EVERY Python-side change, so the repaint never gets
  // skipped just because the piece layout happens to be unchanged.
  model.on("change:version", draw);
  wrap.appendChild(grid); wrap.appendChild(status); wrap.appendChild(bar);
  el.appendChild(wrap);
  draw();
}
export default { render };
"""


class SetupBoard(anywidget.AnyWidget):
    """Click a piece then its destination. Only legal moves are accepted."""

    _esm = _ESM
    squares = traitlets.List(traitlets.Unicode(), default_value=[""] * 64).tag(sync=True)
    legal_from = traitlets.List(traitlets.Unicode()).tag(sync=True)
    status = traitlets.Unicode("").tag(sync=True)
    attempt = traitlets.Unicode("").tag(sync=True)
    action = traitlets.Unicode("").tag(sync=True)
    version = traitlets.Int(0).tag(sync=True)

    def __init__(self, board=None, **kw):
        super().__init__(**kw)
        self.board = board.copy() if board is not None else chess.Board()
        self._start = self.board.copy()
        self._sync()

    def _sync(self):
        sq = []
        for rank in range(7, -1, -1):
            for file in range(8):
                piece = self.board.piece_at(chess.square(file, rank))
                sq.append(_GLYPHS[piece.symbol()] if piece else "")
        self.squares = sq
        self.legal_from = sorted({chess.square_name(m.from_square)
                                  for m in self.board.legal_moves})
        turn = "White" if self.board.turn else "Black"
        line = self._san_line()
        self.status = (f"{turn} to move" + (f"  —  {line}" if line else "")
                       if not self.board.is_game_over()
                       else f"Game over: {self.board.result()}")
        self.version += 1  # always changes -> the frontend always repaints

    def _san_line(self):
        b = self._start.copy()
        out = []
        for i, m in enumerate(self.board.move_stack):
            if b.turn == chess.WHITE:
                out.append(f"{b.fullmove_number}.")
            out.append(b.san(m))
            b.push(m)
        return " ".join(out)

    @traitlets.observe("attempt")
    def _on_attempt(self, change):
        uci = change["new"].split(":")[0]
        if not uci:
            return
        for cand in (uci, uci + "q"):  # auto-queen promotions
            try:
                move = chess.Move.from_uci(cand)
            except ValueError:
                continue
            if move in self.board.legal_moves:
                self.board.push(move)
                self._sync()
                return
        self.status = f"{uci} is not legal"

    @traitlets.observe("action")
    def _on_action(self, change):
        what = change["new"].split(":")[0]
        if what == "undo" and self.board.move_stack:
            self.board.pop()
        elif what == "reset":
            self.board = self._start.copy()
        self._sync()

    @property
    def fen(self):
        return self.board.fen()

    def play(self, *moves):
        """Apply SAN or UCI moves programmatically: b.play('e4', 'e5')."""
        for m in moves:
            for parse in (self.board.parse_san, self.board.parse_uci):
                try:
                    self.board.push(parse(m))
                    break
                except ValueError:
                    continue
            else:
                raise ValueError(f"illegal or unparseable move: {m}")
        self._sync()
        return self
