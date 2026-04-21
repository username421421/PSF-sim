import matplotlib as mpl
import numpy as np
from cycler import cycler
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

NATURE_CYCLE = [
    "#1B3B6F",
    "#D55E00",
    "#007C73",
    "#B2479A",
    "#C99A00",
    "#4D4D4D",
    "#56B4E9",
    "#000000",
]

NEAR_BLACK = "#111111"
LIGHT_GUIDE = "#d8dde6"


def apply_publication_style():
    mpl.rcParams.update(
        {
            "figure.figsize": (3.5, 2.4),
            "figure.dpi": 150,
            "figure.constrained_layout.use": True,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "savefig.transparent": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 6,
            "axes.labelsize": 7,
            "axes.titlesize": 7,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 6,
            "mathtext.fontset": "dejavusans",
            "mathtext.default": "regular",
            "axes.linewidth": 0.6,
            "axes.labelpad": 2.0,
            "lines.linewidth": 1.2,
            "lines.markersize": 3.5,
            "lines.markeredgewidth": 0.6,
            "patch.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.minor.width": 0.45,
            "ytick.minor.width": 0.45,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "xtick.minor.size": 1.5,
            "ytick.minor.size": 1.5,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.top": False,
            "ytick.right": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "legend.handlelength": 1.6,
            "legend.handletextpad": 0.4,
            "legend.borderaxespad": 0.3,
            "legend.labelspacing": 0.25,
            "legend.columnspacing": 0.8,
            "axes.grid": False,
            "axes.facecolor": "#ffffff",
            "figure.facecolor": "#ffffff",
            "savefig.facecolor": "#ffffff",
            "text.color": NEAR_BLACK,
            "axes.labelcolor": NEAR_BLACK,
            "axes.edgecolor": NEAR_BLACK,
            "xtick.color": NEAR_BLACK,
            "ytick.color": NEAR_BLACK,
            "image.cmap": "viridis",
            "image.interpolation": "none",
            "axes.prop_cycle": cycler(color=NATURE_CYCLE),
        }
    )


def make_figure(figsize, dpi=150):
    return Figure(figsize=figsize, dpi=dpi, facecolor="#ffffff", constrained_layout=True)


def style_axis(ax, *, xbins=5, ybins=5):
    ax.tick_params(axis="both", which="major", pad=1.5)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=xbins, min_n_ticks=3))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=ybins, min_n_ticks=3))
    ax.title.set_color(NEAR_BLACK)
    ax.set_facecolor("#ffffff")


def style_image_axis(ax, *, xlabel, ylabel, title=None, xbins=5, ybins=5):
    if title:
        ax.set_title(title, pad=3.0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    style_axis(ax, xbins=xbins, ybins=ybins)


def style_profile_axis(ax, *, xlabel, ylabel, title=None):
    if title:
        ax.set_title(title, pad=3.0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    style_axis(ax, xbins=5, ybins=4)
    ax.grid(axis="y", color=LIGHT_GUIDE, linewidth=0.45)


def add_compact_colorbar(fig, mappable, ax, label, *, ticks=None):
    cbar = fig.colorbar(mappable, ax=ax, fraction=0.042, pad=0.025, shrink=0.98)
    if ticks is not None:
        cbar.set_ticks(ticks)
    cbar.set_label(label, labelpad=3)
    cbar.ax.tick_params(width=0.6, length=2.0, pad=1.2, labelsize=mpl.rcParams["ytick.labelsize"])
    cbar.outline.set_linewidth(0.6)
    if ticks is None:
        cbar.ax.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
    return cbar


def add_panel_labels(axes, labels=None, *, x=-0.14, y=1.04):
    if labels is None:
        labels = "abcdefghijklmnopqrstuvwxyz"
    flat_axes = np.asarray(axes, dtype=object).ravel()
    for label, ax in zip(labels, flat_axes):
        ax.text(
            x,
            y,
            str(label),
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontweight="bold",
            clip_on=False,
        )


def intensity_norm(data, *, vmin=0.0):
    arr = np.asarray(data, dtype=float)
    if arr.size == 0 or not np.any(np.isfinite(arr)):
        vmax = 1.0
    else:
        vmax = float(np.nanmax(arr))
    if vmax <= vmin:
        vmax = vmin + 1.0
    return Normalize(vmin=vmin, vmax=vmax)


def signed_phase_norm(limit):
    bound = abs(float(limit))
    return TwoSlopeNorm(vmin=-bound, vcenter=0.0, vmax=bound)
