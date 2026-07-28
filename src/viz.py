"""Result views: one panel per group — board on the left, its bars on the right."""

import chess
import chess.svg
from IPython.display import HTML, display

A_COLOR = "#4c9be8"   # played sooner after A
B_COLOR = "#e8734c"   # played sooner after B
SHARED = "#3fb950"    # played about equally soon after both

# selectable metrics: key -> (label, definition shown above the panels).
# Both are already present on every row, so switching costs no new rollouts.
METRICS = {
    "speed": ("1/E[moves until played]", 
              "score = 1 / E[moves until played]"),
    "frequency": ("Frequency in rollouts",
                  "score = proportion of rollouts where move was played"),
}
_METRIC_FIELDS = {"speed": ("score_A", "score_B"),
                  "frequency": ("P_A", "P_B")}


def with_metric(rows, metric="speed"):
    """Return rows with score_A/score_B/diff set from the chosen metric.

    The rollout data already carries both measures, so this is a pure
    reinterpretation -- no engine work is repeated.
    """
    key_a, key_b = _METRIC_FIELDS[metric]
    out = []
    for r in rows:
        sa, sb = r[key_a], r[key_b]
        out.append({**r, "score_A": sa, "score_B": sb, "diff": sa - sb})
    return out

_BOARD_STYLE = (
    ".square.light {fill:#e8ecf2} .square.dark {fill:#8b9bb0}"
    ".coord {fill:#8a8a99; font-size:10px}"
)


def buckets(rows, band=None):
    """Rank rows three ways: (unique to A, common to both, unique to B).

    These are rankings, not a partition -- a move can place highly in more than
    one list. `band` is accepted and ignored, for backwards compatibility.

        unique to A  : highest  score_A - score_B
        common       : highest  min(score_A, score_B)
        unique to B  : highest  score_B - score_A
    """
    a = sorted(rows, key=lambda r: r["score_A"] - r["score_B"], reverse=True)
    b = sorted(rows, key=lambda r: r["score_B"] - r["score_A"], reverse=True)
    common = sorted(rows, key=lambda r: min(r["score_A"], r["score_B"]),
                    reverse=True)
    return a, common, b


def _board_svg(board, played, rows, sq_key, color, top, size=260):
    """Board after `played` (or as-is if `played` is None), with square shading
    and arrows. Opacity normalized so highest score is fully opaque."""
    b = board.copy()
    if played is not None:
        b.push(played)
    shown = [r for r in rows if r.get(sq_key)][:top]

    score_key = "score_A" if sq_key == "_sq_A" else "score_B"
    if shown:
        max_score = max(r.get(score_key, 0) for r in shown)
    else:
        max_score = 1.0

    # arrows only -- opacity normalized so the top score is fully opaque
    arrows = []
    for r in shown:
        frm, to = r[sq_key]
        score = r.get(score_key, 0.5)
        alpha = max(40, int(255 * score / max(max_score, 0.01)))
        if frm != to:
            arrows.append(chess.svg.Arrow(frm, to, color=color + f"{alpha:02x}"))

    # the move that was played is marked by shading its squares, not an arrow
    fill = {}
    if played is not None:
        fill[played.to_square] = "#f5d76eaa"
    return chess.svg.board(b, arrows=arrows, fill=fill, size=size,
                           coordinates=True, style=_BOARD_STYLE)


def _bars(rows, san_a, san_b, span, floor=0.0, width=360, scale=1.0):
    """Paired bars per move: one for the score after A, one after B.

    Bars are drawn from `floor` (the never-played score) rather than from zero,
    since every score sits above that floor and drawing from 0 would leave all
    the bars nearly the same length. `span` and `floor` come from the caller and
    are shared across panels, so lengths are comparable between the boards.
    """
    if not rows:
        return (f'<div style="font:{12 * scale:.0f}px system-ui;color:#8a8a99">'
                f'(none)</div>')
    reach = max(span - floor, 1e-9)
    width = width * scale
    pair_h, bar_h = 32 * scale, 11 * scale
    label_w, val_w = 48 * scale, 46 * scale
    f_move, f_val = 12 * scale, 10 * scale
    bar_max = width - label_w - val_w
    height = len(rows) * pair_h + 6

    p = [f'<svg width="{width:.0f}" height="{height:.0f}" '
         f'font-family="system-ui" style="max-width:100%">']
    for i, r in enumerate(rows):
        y = i * pair_h
        p.append(f'<text x="{label_w - 8 * scale:.1f}" y="{y + 17 * scale:.1f}" '
                 f'font-size="{f_move:.1f}" '
                 f'fill="#e8e8f0" text-anchor="end">{r["move"]}</text>')
        for j, (key, color, san) in enumerate(
                (("score_A", A_COLOR, san_a), ("score_B", B_COLOR, san_b))):
            by = y + j * (bar_h + 2 * scale) + scale
            length = max((r[key] - floor) / reach * bar_max, 1.5)
            p.append(f'<rect x="{label_w:.1f}" y="{by:.1f}" '
                     f'width="{length:.1f}" height="{bar_h:.1f}" '
                     f'rx="2" fill="{color}"/>')
            p.append(f'<text x="{label_w + length + 5 * scale:.1f}" '
                     f'y="{by + 9 * scale:.1f}" '
                     f'font-size="{f_val:.1f}" fill="#8a8a99">{r[key]:.2f} '
                     f'<tspan fill="#6f7684">{san}</tspan></text>')
    p.append("</svg>")
    return "".join(p)


def _panel(title, subtitle, color, board_svg, bars_html, scale=1.0):
    return (
        f'<div style="margin-bottom:{26 * scale:.0f}px">'
        f'<div style="font:600 {14 * scale:.0f}px system-ui;color:{color}">'
        f'{title}</div>'
        f'<div style="font:{11 * scale:.0f}px system-ui;color:#8a8a99;'
        f'margin:2px 0 8px">{subtitle}</div>'
        f'<div style="display:flex;gap:{18 * scale:.0f}px;'
        f'align-items:flex-start;flex-wrap:wrap">'
        f'<div>{board_svg}</div><div style="padding-top:2px">{bars_html}</div>'
        f'</div></div>')


def panels(board, rows, move_a, move_b, top=6, band=None, metric="speed",
           scale=1.0):
    """Build the three stacked panels: unique to A, common to both, unique to B.

    `top` sets how many moves each list shows. `metric` picks what the bars
    measure -- see METRICS. `scale` grows or shrinks the whole rendering
    (1.5 = half again as large). Each panel is a board (with arrows to those
    moves) beside paired bars giving both values; all three panels share one bar
    scale. Returns an HTML object rather than displaying, so callers can
    re-render it (see `show_interactive`).
    """
    if metric not in METRICS:
        raise ValueError(f"metric must be one of {sorted(METRICS)}, got {metric!r}")
    rows = with_metric(rows, metric)
    a, common, b = buckets(rows)
    san_a, san_b = board.san(move_a), board.san(move_b)
    a, common, b = a[:top], common[:top], b[:top]

    # one scale across all three panels, so bars are comparable between boards.
    # bars start at the lowest value shown, so small differences stay visible.
    shown = a + common + b
    span = max((max(r["score_A"], r["score_B"]) for r in shown), default=1.0)
    floor = min((min(r["score_A"], r["score_B"]) for r in shown), default=0.0)

    label, definition = METRICS[metric]
    board_px = round(260 * scale)
    html = [
        f'<div style="font:{12 * scale:.0f}px system-ui;color:#c8c8d4;'
        f'margin-bottom:{14 * scale:.0f}px;line-height:1.5">'
        f'<div style="text-align:center;font-style:italic;margin:8px 0;'
        f'padding:8px;background:rgba(255,255,255,0.04);border-radius:4px">'
        f'<code style="color:#e8e8f0">{definition}</code></div><br>'
    ]

    html.append(_panel(
        f"Unique to {san_a}",
        f"highest  score after {san_a} − score after {san_b}",
        A_COLOR,
        _board_svg(board, move_a, a, "_sq_A", A_COLOR, top, board_px),
        _bars(a, san_a, san_b, span, floor, scale=scale), scale))

    # shared moves are shown from the starting position, before either candidate
    html.append(_panel(
        "Common to both",
        f"highest  min(score after {san_a}, score after {san_b})",
        SHARED,
        _board_svg(board, None, common, "_sq_A", SHARED, top, board_px),
        _bars(common, san_a, san_b, span, floor, scale=scale), scale))

    html.append(_panel(
        f"Unique to {san_b}",
        f"highest  score after {san_b} − score after {san_a}",
        B_COLOR,
        _board_svg(board, move_b, b, "_sq_B", B_COLOR, top, board_px),
        _bars(b, san_a, san_b, span, floor, scale=scale), scale))

    # the dark background is set here rather than inherited: an ipywidgets.HTML
    # renders on the notebook's own (usually light) background, which would
    # leave this pale-on-white and unreadable.
    return HTML(f'<div style="background:#1b1b21;'
                f'padding:{18 * scale:.0f}px;border-radius:6px">'
                f'{"".join(html)}</div>')


def show(board, rows, move_a, move_b, top=6, band=None, metric="speed",
         scale=1.0):
    """Display the three panels. See `panels` for what they mean."""
    display(panels(board, rows, move_a, move_b, top, band, metric, scale))


def show_interactive(board, rows, move_a, move_b, top=6, max_top=10,
                     metric="speed", scale=1.4, png_path="panels.png"):
    """Same three panels, with controls for how many moves and which metric.

    `scale` sizes the whole output; the slider caps at `max_top` moves. The
    Save PNG button writes the current view to `png_path` (needs Chrome).
    """
    import ipywidgets as widgets

    # cap the slider at max_top, but never above the number of moves available
    limit = max(min(len(rows), max_top), 1)
    slider = widgets.IntSlider(
        value=min(top, limit), min=1, max=limit, step=1,
        description="moves shown:", continuous_update=False,
        style={"description_width": "initial"},
        layout=widgets.Layout(width="380px"))
    picker = widgets.Dropdown(
        options=[(METRICS[k][0], k) for k in METRICS], value=metric,
        description="score:", style={"description_width": "initial"},
        layout=widgets.Layout(width="380px"))
    # an HTML widget rather than Output(): setting .value always re-renders,
    # whereas Output only captures display() once it is attached to a frontend.
    view = widgets.HTML()

    def render(*_):
        view.value = panels(board, rows, move_a, move_b, top=slider.value,
                            metric=picker.value, scale=scale).data

    # save whatever is currently on screen, at the current slider/metric
    name = widgets.Text(value=png_path, description="png:",
                        style={"description_width": "initial"},
                        layout=widgets.Layout(width="300px"))
    save = widgets.Button(description="Save PNG", icon="download",
                          layout=widgets.Layout(width="120px"))
    note = widgets.HTML()

    def on_save(_):
        note.value = '<span style="font:12px system-ui;color:#8a8a99">saving…</span>'
        try:
            written = save_png(view.value, name.value)
            msg, color = f"saved {written}", "#3fb950"
        except Exception as exc:                      # surface it in the notebook
            msg, color = str(exc), "#e8734c"
        note.value = f'<span style="font:12px system-ui;color:{color}">{msg}</span>'

    save.on_click(on_save)

    slider.observe(render, names="value")
    picker.observe(render, names="value")
    render()
    # the VBox must be told to grow, or it clips the scaled-up panels
    return widgets.VBox([widgets.HBox([slider, picker]),
                         widgets.HBox([name, save, note]), view],
                        layout=widgets.Layout(width="100%"))


_CHROME_CANDIDATES = (
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "google-chrome", "chromium", "chromium-browser",
)


def _find_chrome(path=None):
    """A headless-capable Chrome/Chromium, or None."""
    import os
    import shutil

    for cand in ([path] if path else []) + list(_CHROME_CANDIDATES):
        if cand and (os.path.isfile(cand) or shutil.which(cand)):
            return cand
    return None


def save_png(html, path, width=None, height=None, chrome=None):
    """Render `html` (str or IPython HTML) to a PNG at `path`.

    Uses headless Chrome, so no extra Python packages are needed. `width`
    defaults to wide enough that panels keep their side-by-side layout instead
    of wrapping; `height` defaults to a tall canvas cropped down to the content.
    Returns the path written.
    """
    import subprocess
    import tempfile
    from pathlib import Path

    exe = _find_chrome(chrome)
    if exe is None:
        raise RuntimeError(
            "No Chrome/Chromium found for PNG export. Install Google Chrome, "
            "or pass chrome='/path/to/chrome'."
        )
    markup = getattr(html, "data", html)
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    if width is None:
        width = _natural_width(markup)

    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "page.html"
        # nowrap on the panel rows keeps the board and its bars side by side,
        # the way they appear in a wide notebook cell
        src.write_text(
            '<meta charset="utf-8">'
            '<style>div[style*="display:flex"]{flex-wrap:nowrap!important}</style>'
            '<body style="margin:0;background:#1b1b21">'
            f'{markup}</body>', encoding="utf-8")
        cmd = [exe, "--headless", "--disable-gpu", "--no-sandbox",
               "--hide-scrollbars", f"--screenshot={path}",
               f"--window-size={round(width)},{height or 4000}",
               src.as_uri()]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if not path.exists():
        raise RuntimeError(f"Chrome failed to write {path}:\n{proc.stderr[-500:]}")

    _crop(path, crop_height=height is None)
    return str(path)


def _natural_width(markup):
    """Width that fits the widest board+bars row without wrapping.

    The panels are built from SVGs whose widths are in the markup, so measure
    those rather than guessing: widest board + widest bars + gap + padding.
    """
    import re

    widths = [float(w) for w in re.findall(r'<svg width="([\d.]+)"', markup)]
    if not widths:
        return 1200
    # boards and bar charts alternate; boards are square (they carry a viewBox)
    boards = [float(w) for w in re.findall(
        r'<svg[^>]*width="([\d.]+)"[^>]*viewBox', markup)]
    bars = [w for w in widths if w not in boards] or [360.0]
    # the bar SVG's declared width excludes the value text drawn past the bar,
    # so leave generous slack or the longest labels get clipped
    return max(boards or [260.0]) + max(bars) + 200


def _crop(path, crop_height=True):
    """Trim uniform background from the right and bottom, if Pillow is present."""
    try:
        from PIL import Image
    except ImportError:
        return  # leave the fixed-size image; still perfectly usable
    im = Image.open(path).convert("RGB")
    bg = im.getpixel((im.width - 1, im.height - 1))

    def last_used(size, line):
        for i in range(size - 1, 0, -1):
            if any(px != bg for px in line(i).getdata()):
                return i
        return size - 1

    right = last_used(im.width, lambda x: im.crop((x, 0, x + 1, im.height)))
    bottom = (last_used(im.height, lambda y: im.crop((0, y, im.width, y + 1)))
              if crop_height else im.height - 24)
    # pad past the last drawn pixel so text sitting at the edge is not clipped
    im.crop((0, 0, min(right + 40, im.width),
             min(bottom + 30, im.height))).save(path)


def to_frame(rows, move_a="A", move_b="B"):
    """Rows as a DataFrame with self-describing column names."""
    import pandas as pd

    df = pd.DataFrame([{
        "move": r["move"],
        f"score after {move_a}": r["score_A"],
        f"score after {move_b}": r["score_B"],
        f"score diff ({move_a} - {move_b})": r["diff"],
        f"% of rollouts played after {move_a}": 100 * r["P_A"],
        f"% of rollouts played after {move_b}": 100 * r["P_B"],
    } for r in rows])
    return (df.sort_values(f"score diff ({move_a} - {move_b})", ascending=False)
              .reset_index(drop=True))
