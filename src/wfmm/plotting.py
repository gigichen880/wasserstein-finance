from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG_DPI = 150

plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 10,
    "legend.fontsize": 7,
    "figure.dpi": FIG_DPI,
})

SHOCK_COLOR = "#f4a261"


def panel_label(ax, label: str) -> None:
    ax.text(-0.10, 1.04, label, transform=ax.transAxes, fontsize=11, fontweight="bold",
            va="bottom", ha="left")


def legend_below(ax, ncol: int = 2, fontsize: float = 6.5, handles=None) -> None:
    ax.legend(
        handles=handles, frameon=False, loc="upper center",
        bbox_to_anchor=(0.5, -0.32), ncol=ncol, fontsize=fontsize,
    )


def save_fig(fig, path, bottom_pad: float = 0.0) -> None:
    if bottom_pad:
        fig.tight_layout(rect=[0, bottom_pad, 1, 1])
    else:
        fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=FIG_DPI)
    plt.close(fig)
