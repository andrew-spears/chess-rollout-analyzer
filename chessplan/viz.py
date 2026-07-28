"""Result views: annotated boards (where each plan goes) + a diverging bar chart."""

import chess
import chess.svg
from IPython.display import HTML, display

A_COLOR = "#4c9be8"   # A's plan
B_COLOR = "#e8734c"   # B's plan
SHARED = "#9aa0ad"    # played about equally by both

_BOARD_STYLE = (
    ".square.light {fill:#e8ecf2} .square.dark {fill:#8b9bb0}"
    ".coord {fill:#8a8a99; font-size:10px}"
)


def buckets(rows, band=0.05):
    """Split rows into (A's plan, common, B's plan).

    `band` is the |delta| below which a move counts as common to both. delta is
    a difference of relevances, each in [0, 1], so 0.05 is a small gap.
    """
    a = sorted((r for r in rows if r["delta"] > band),
               key=lambda r: r["delta"], reverse=True)
    b = sorted((r for r in rows if r["delta"] < -band),
               key=lambda r: r["delta"])
    # sorted by delta too, so bar length decreases down the group like the others
    common = sorted((r for r in rows if abs(r["delta"]) <= band),
                    key=lambda r: r["delta"], reverse=True)
    return a, common, b


def _annotated_board(board, played, rows, key, sq_key, color, top):
    """Board after `played`, with arrows for that side's top plan-moves."""
    b = board.copy()
    b.push(played)
    rows = [r for r in rows if r[key] > 0][:top]

    arrows, fill = [], {}
    for i, r in enumerate(rows):
        frm, to = r.get(sq_key) or (None, None)
        if to is None:
            continue
        # opacity by rank among the shown moves
        weight = 1.0 - i / max(len(rows) - 1, 1)
        fill[to] = color + f"{int(60 + 150 * weight):02x}"
        if frm is not None and frm != to:
            arrows.append(chess.svg.Arrow(frm, to, color=color + "cc"))

    # the move that was actually played, in green, so the board is self-labelling
    arrows.append(chess.svg.Arrow(played.from_square, played.to_square,
                                  color="#3fb950dd"))
    return chess.svg.board(b, arrows=arrows, fill=fill, size=330,
                           coordinates=True, style=_BOARD_STYLE)


def plan_boards(board, rows, move_a, move_b, top=6, band=0.05):
    """Side-by-side boards: each shows its own move played, plus arrows to the
    moves that are most central to that move's plan."""
    a, _, b = buckets(rows, band)
    san_a, san_b = board.san(move_a), board.san(move_b)

    def panel(played, san, side_rows, key, sq_key, color):
        svg = _annotated_board(board, played, side_rows, key, sq_key, color, top)
        return (f'<div style="display:inline-block;margin-right:20px;'
                f'vertical-align:top;text-align:center">'
                f'<div style="font:600 14px system-ui;color:{color};'
                f'margin-bottom:6px">after {san}</div>{svg}</div>')

    return HTML(
        '<div style="white-space:nowrap;overflow-x:auto">'
        + panel(move_a, san_a, a, "rel_A", "_sq_A", A_COLOR)
        + panel(move_b, san_b, b, "rel_B", "_sq_B", B_COLOR)
        + '</div>'
        '<div style="font:11px system-ui;color:#8a8a99;margin-top:4px">'
        'green = the move played · coloured arrows = that plan\'s follow-ups, '
        'strongest first</div>')


def gap_chart(rows, move_a="A", move_b="B", top=16, band=0.05, width=720):
    """Diverging bars grouped into A's plan / common / B's plan."""
    rows = sorted(rows, key=lambda r: abs(r["delta"]), reverse=True)[:top]
    a, common, b = buckets(rows, band)
    ordered = (a + common + b)
    if not ordered:
        return HTML('<div style="font:13px system-ui;color:#8a8a99">no moves</div>')

    span = max(abs(r["delta"]) for r in ordered) or 1.0
    row_h, mid = 24, width / 2
    tok_gutter, val_gutter = 44, 34
    half = mid - tok_gutter - val_gutter
    head = 34
    # a header line for each non-empty group
    groups = [(g, lbl, col) for g, lbl, col in (
        (a, f"{move_a}'s plan", A_COLOR),
        (common, "common to both", SHARED),
        (b, f"{move_b}'s plan", B_COLOR)) if g]
    height = head + len(ordered) * row_h + len(groups) * 22 + 16

    p = [f'<svg width="{width}" height="{height}" font-family="system-ui" '
         f'style="max-width:100%">',
         f'<text x="{mid - 8}" y="14" fill="{B_COLOR}" font-size="11" '
         f'text-anchor="end">← more central to {move_b}</text>',
         f'<text x="{mid + 8}" y="14" fill="{A_COLOR}" font-size="11">'
         f'more central to {move_a} →</text>']

    y = head
    for group, label, gcolor in groups:
        p.append(f'<text x="4" y="{y + 4}" font-size="11" font-weight="600" '
                 f'fill="{gcolor}">{label}</text>')
        y += 18
        for r in group:
            length = abs(r["delta"]) / span * half
            right = r["delta"] >= 0
            if r["delta"] > band:
                x, color = mid, A_COLOR
            elif r["delta"] < -band:
                x, color = mid - length, B_COLOR
            else:
                length = max(length, 3)
                x, color = mid - length / 2, SHARED
            p.append(f'<rect x="{x:.1f}" y="{y}" width="{length:.1f}" '
                     f'height="14" rx="2" fill="{color}"/>')
            tok_x = (x + length + 7) if right else (x - 7)
            anchor = "start" if right else "end"
            val_x = mid + (half + tok_gutter + 6) * (1 if right else -1)
            p.append(f'<text x="{tok_x:.1f}" y="{y + 11}" font-size="12" '
                     f'fill="#e8e8f0" text-anchor="{anchor}">{r["move"]}</text>')
            p.append(f'<text x="{val_x:.1f}" y="{y + 11}" font-size="10" '
                     f'fill="#8a8a99" text-anchor="{anchor}">'
                     f'{r["delta"]:+.2f}</text>')
            y += row_h
        y += 4

    p.append(f'<text x="4" y="{height - 4}" font-size="10" fill="#8a8a99">'
             f'bar = gap in relevance (1 = played at once every rollout, '
             f'0 = never played)</text>')
    p.append("</svg>")
    return HTML("".join(p))


def to_frame(rows):
    """Rows as a pandas DataFrame, strongest A-plan first."""
    import pandas as pd

    df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")}
                       for r in rows])
    return df.sort_values("delta", ascending=False).reset_index(drop=True)


def show(board, rows, move_a, move_b, top=6, band=0.05):
    """The whole result: annotated boards above, grouped gap chart below."""
    display(plan_boards(board, rows, move_a, move_b, top, band))
    display(gap_chart(rows, board.san(move_a), board.san(move_b), band=band))
