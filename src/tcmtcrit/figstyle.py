"""Figure style and layout verification.

Every figure is a single panel. Axis labels, tick labels and legends sit outside the
plotting box, which holds only data. check_layout renders the figure, collects the
bounding boxes of all text artists and reports overlaps, clipping at the canvas edge,
text inside the axes rectangle and labels whose underscores would print literally
(subscripts written outside math mode).
"""
from __future__ import annotations

import os
import re

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, NullFormatter

CM = 1.0 / 2.54

# Okabe-Ito colour-blind-safe palette.
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000"]
GREY = "#4D4D4D"
BAND = "#BFBFBF"
MARKERS = ["o", "s", "^", "D", "v", "P", "X"]

STYLE = {
    "figure.dpi": 110,
    "savefig.dpi": 600,
    "font.family": "DejaVu Sans",
    "mathtext.fontset": "dejavusans",
    "font.size": 13,
    "axes.labelsize": 15,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 12.5,
    "axes.linewidth": 1.1,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 5,
    "ytick.major.size": 5,
    "xtick.minor.size": 3,
    "ytick.minor.size": 3,
    "lines.linewidth": 2.0,
    "lines.markersize": 7,
    "legend.frameon": False,
    "legend.handlelength": 2.2,
    "legend.labelspacing": 0.6,
    "axes.unicode_minus": True,
}


def new_figure(width_cm=17.5, height_cm=12.0):
    plt.rcParams.update(STYLE)
    fig, ax = plt.subplots(figsize=(width_cm * CM, height_cm * CM), layout="constrained")
    fig.get_layout_engine().set(w_pad=0.10, h_pad=0.10, wspace=0.02, hspace=0.02)
    return fig, ax


def _plain(value, _pos=None):
    return f"{value:g}" if value > 0 else ""


def _decade(value, _pos=None):
    return rf"$10^{{{int(round(np.log10(value)))}}}$" if value > 0 else ""


def log_axis(ax, which, ticks=None):
    """Logarithmic axis with plainly written tick labels and unlabelled minor ticks.

    ticks lists the labelled positions, written as plain numbers; by default every
    decade is labelled as a power of ten.
    """
    (ax.set_xscale if which == "x" else ax.set_yscale)("log")
    axis = ax.xaxis if which == "x" else ax.yaxis
    if ticks is None:
        axis.set_major_locator(LogLocator(base=10.0, numticks=12))
        axis.set_major_formatter(FuncFormatter(_decade))
    else:
        axis.set_major_locator(FixedLocator(ticks))
        axis.set_major_formatter(FuncFormatter(_plain))
    axis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10), numticks=12))
    axis.set_minor_formatter(NullFormatter())


def legend_outside(fig, ax, where="bottom", ncol=2):
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return None
    loc = {"right": "outside right upper", "bottom": "outside lower center"}[where]
    return fig.legend(handles, labels, loc=loc, ncol=ncol, frameon=False)


_MATH = re.compile(r"\$[^$]*\$")


def _literal_underscore(text: str) -> bool:
    return "_" in _MATH.sub("", text)


def check_layout(fig, ax):
    """List of layout problems; an empty list means the figure is clean."""
    fig.canvas.draw()
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    box = ax.get_window_extent(renderer=r)
    items = []
    for kind, art in (("x label", ax.xaxis.label), ("y label", ax.yaxis.label),
                      ("title", ax.title)):
        if art.get_text():
            items.append((kind, art.get_text(), art.get_window_extent(renderer=r)))
    for ticks, labels, lim in ((ax.get_xticks(), ax.get_xticklabels(), ax.get_xlim()),
                               (ax.get_yticks(), ax.get_yticklabels(), ax.get_ylim())):
        lo, hi = min(lim), max(lim)
        for pos, t in zip(ticks, labels):
            if t.get_text() and t.get_visible() and lo <= pos <= hi:
                items.append(("tick", t.get_text(), t.get_window_extent(renderer=r)))
    for t in ax.texts:
        items.append(("annotation", t.get_text(), t.get_window_extent(renderer=r)))
    for lg in list(fig.legends) + ([ax.get_legend()] if ax.get_legend() else []):
        items.append(("legend", "", lg.get_window_extent(renderer=r)))
        for t in lg.get_texts():
            if _literal_underscore(t.get_text()):
                items.append(("legend entry", t.get_text(), None))

    problems = []
    for kind, text, bb in items:
        if text and _literal_underscore(text):
            problems.append(f"{kind} '{text}' contains a literal underscore")
        if bb is None:
            continue
        if kind in ("annotation", "legend"):
            if bb.x0 < box.x1 and bb.x1 > box.x0 and bb.y0 < box.y1 and bb.y1 > box.y0:
                problems.append(f"{kind} intrudes into the plot area")
        if bb.x0 < -1 or bb.y0 < -1 or bb.x1 > fig.bbox.x1 + 1 or bb.y1 > fig.bbox.y1 + 1:
            problems.append(f"{kind} clipped at the canvas edge")
    boxed = [(k, b) for k, _, b in items if b is not None]
    for i in range(len(boxed)):
        for j in range(i + 1, len(boxed)):
            (k1, b1), (k2, b2) = boxed[i], boxed[j]
            if min(b1.x1, b2.x1) - max(b1.x0, b2.x0) > 1 and \
               min(b1.y1, b2.y1) - max(b1.y0, b2.y0) > 1:
                problems.append(f"{k1} overlaps {k2}")
    return sorted(set(problems))


def save(fig, ax, stem, folder="figures"):
    """Verify the layout, then write 600 dpi PNG and vector PDF. Raises on problems."""
    problems = check_layout(fig, ax)
    if problems:
        plt.close(fig)
        raise RuntimeError(f"{stem}: " + "; ".join(problems))
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, stem)
    fig.savefig(path + ".png", dpi=600, metadata={"Software": None})
    fig.savefig(path + ".pdf", metadata={"Creator": None, "Producer": None,
                                         "CreationDate": None})
    return path + ".png"
