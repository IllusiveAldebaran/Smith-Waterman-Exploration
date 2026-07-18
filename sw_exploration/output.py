"""Output helpers: console printing, file writing, and matplotlib matrix figure.

build_matrix_figure visualises the actual H matrix for each pair as an imshow.
Overlays are added for cells passing --min-score (red tint) and cells where
Farrar's lazy-F correction fired (lime scatter points).

write_output writes structured JSON: metadata block plus one entry per pair
containing per-pair stats, the H matrix (when computed), and lazy-F trigger
coordinates.

Path helpers auto-increment a numeric suffix so successive runs never overwrite.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from math import ceil
from pathlib import Path


# ---------------------------------------------------------------------------
# Console helpers
# ---------------------------------------------------------------------------

# Since events are sometimes stored as cell events these reconstruct into a matrix
# Ex: if farrar's LazyF loop triggers in cell [2,3] and [43,5] and that's all the info stored
#     then this reconstructs it so its actual stats in a matrix
def matrix_from_cell_events(
    events: list[tuple[int, int, int | None]], rows: int, cols: int, default: int = 0
) -> list[list[int]]:
    """Reconstruct a dense matrix by placing each (row, col, value) event's
    value at its cell. Cells with no matching event keep `default`.

    Works for any cell-event list recorded via Recorder.add_cell_event —
    e.g. an "h_matrix" event list (one entry per cell) — not specific to
    H-matrix scores.
    """
    mat = [[default] * cols for _ in range(rows)]
    for row, col, value in events:
        mat[row][col] = value
    return mat


def format_matrix(matrix: list[list[int]], query: str, reference: str) -> str:
    header = "      " + " ".join(f"{ch:>3}" for ch in "-" + reference)
    lines = [header]
    for i, row in enumerate(matrix):
        label = "-" if i == 0 else query[i - 1]
        lines.append(f"{label:>3}   " + " ".join(f"{cell:>3}" for cell in row))
    return "\n".join(lines)


def print_counts(title: str, counts: Counter[str], prefix: str | None = None) -> None:
    print(f"\n{title}:")
    for key in sorted(counts):
        if prefix is None or key.startswith(prefix):
            print(f"  {key}: {counts[key]}")


def print_times(times: dict[str, float]) -> None:
    total = sum(times.values())
    print("\nTiming:")
    for key in sorted(times):
        seconds = times[key]
        pct = (seconds / total * 100.0) if total else 0.0
        print(f"  {key}: {seconds:.6f}s ({pct:.2f}% of recorded time)")


def smith_waterman_time(times: dict[str, float]) -> float:
    return times.get("smith_waterman.dp_fill", 0.0) + times.get(
        "smith_waterman.traceback", 0.0
    )


def farrar_time(times: dict[str, float]) -> float:
    return (
        times.get("farrar.profile_build", 0.0)
        + times.get("farrar.main_striped_pass", 0.0)
        + times.get("farrar.lazy_f_correction", 0.0)
    )


# ---------------------------------------------------------------------------
# Overlay registry
# ---------------------------------------------------------------------------

def _overlay_lazy_f(ax, p, mat, **_):
    import numpy as np
    from matplotlib.lines import Line2D
    triggers = p.get("triggers", {}).get("farrar.lazy_f_trigger", [])
    if not triggers:
        return None
    # Triggers stored as (query_row, ref_col, value); after transpose x=query, y=ref.
    xs = [r for r, c, _ in triggers]
    ys = [c for r, c, _ in triggers]
    ax.scatter(xs, ys, c="lime", s=4, alpha=0.8, marker=".")
    return Line2D([0], [0], marker=".", color="w", markerfacecolor="lime",
                  markersize=6, label=f"lazy-F ({len(triggers)})")


def _overlay_match(ax, p, mat, **_):
    import numpy as np
    from matplotlib.patches import Patch
    query, ref = p["query_seq"], p["reference_seq"]
    q = np.array(list(query))
    r = np.array(list(ref))
    # match_grid[ri, ci] == True where ref[ri] == query[ci]
    match_grid = r[:, None] == q[None, :]
    mask = np.full(mat.shape, np.nan)
    mask[1:, 1:] = np.where(match_grid, 1.0, np.nan)
    ax.imshow(mask, aspect="auto", cmap="Greens", alpha=0.35,
              origin="upper", vmin=0, vmax=1)
    return Patch(facecolor="mediumseagreen", alpha=0.5, label="match")


def _overlay_mismatch(ax, p, mat, **_):
    import numpy as np
    from matplotlib.patches import Patch
    query, ref = p["query_seq"], p["reference_seq"]
    q = np.array(list(query))
    r = np.array(list(ref))
    mismatch_grid = r[:, None] != q[None, :]
    mask = np.full(mat.shape, np.nan)
    mask[1:, 1:] = np.where(mismatch_grid, 1.0, np.nan)
    ax.imshow(mask, aspect="auto", cmap="Oranges", alpha=0.35,
              origin="upper", vmin=0, vmax=1)
    return Patch(facecolor="darkorange", alpha=0.5, label="mismatch")


OVERLAY_REGISTRY: dict[str, callable] = {
    "lazy_f": _overlay_lazy_f,
    "match": _overlay_match,
    "mismatch": _overlay_mismatch,
}


# ---------------------------------------------------------------------------
# Matplotlib figure
# ---------------------------------------------------------------------------

def build_matrix_figure(
    pairs: list[dict],
    overlays: list[str] | None = None,
    metric: str = "SWaG max score",
    annotate: bool = False,
):
    """Build a figure showing each pair's H matrix.

    Layout: Query positions on the X axis (top), Reference positions on Y (left).
    The H matrix is transposed so columns = query, rows = reference.

    overlays selects from OVERLAY_REGISTRY; each renderer fires only when its
    data is present (e.g. min_score overlay does nothing without --min-score).

    Returns a matplotlib Figure; the caller decides whether to show or save it.
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        raise ImportError(
            "matplotlib and numpy are required for matrix visualisation. "
            "Install with: pip install matplotlib numpy"
        )

    overlays = overlays or []
    n = len(pairs)
    cols = ceil(n ** 0.5)
    rows = ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.2), squeeze=False)
    fig.suptitle(metric, fontsize=10)

    for idx, p in enumerate(pairs):
        ax = axes[idx // cols][idx % cols]
        query_len = p["query_len"]
        ref_len = p["reference_len"]
        legend_handles = []

        # Transpose: original matrix is (query+1) rows × (ref+1) cols.
        # After transpose: rows = ref positions, cols = query positions.
        mat = np.array(p["h_matrix"], dtype=float).T

        im = ax.imshow(mat, aspect="auto", cmap="Blues", origin="upper")
        plt.colorbar(im, ax=ax)

        for key in overlays:
            handle = OVERLAY_REGISTRY[key](ax, p, mat)
            if handle is not None:
                legend_handles.append(handle)

        if legend_handles:
            ax.legend(handles=legend_handles, fontsize=6, loc="upper left",
                      bbox_to_anchor=(1.05, 1.1), borderaxespad=0)

        # Numeric position ticks. X (query) on top; Y (reference) on left.
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position("top")
        ax.tick_params(bottom=False, right=False)

        x_step = max(1, query_len // 8)
        y_step = max(1, ref_len // 8)
        x_ticks = list(range(0, query_len + 1, x_step))
        y_ticks = list(range(0, ref_len + 1, y_step))
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_ticks, fontsize=6)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_ticks, fontsize=6)

        ax.set_xlabel(f"Query (len={query_len})", fontsize=7)
        ax.set_ylabel(f"Reference (len={ref_len})", fontsize=7)

        if annotate:
            vmin_val = mat.min()
            span = (mat.max() - vmin_val) or 1.0
            # Scale font so it shrinks gracefully for large matrices; floor at 1pt.
            fontsize = max(1.0, min(5.0, 400.0 / max(query_len + 1, ref_len + 1)))
            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    val = mat[i, j]
                    norm = (val - vmin_val) / span
                    color = "white" if norm > 0.5 else "black"
                    ax.text(j, i, str(int(val)), ha="center", va="center",
                            fontsize=fontsize, color=color)

        title = f"{p['query_name']} × {p['reference_name']}\nmax score={p['score']}"
        ax.set_title(title, fontsize=8, pad=14)

    for idx in range(n, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    fig.tight_layout()
    return fig


def build_summary_figure(pairs: list[dict]):
    """Aggregate and average all numeric per-pair stats across the run.

    Shows mean ± std for alignment metrics and (when available) timing.
    Does not require H matrices.

    Returns a matplotlib Figure.
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        raise ImportError(
            "matplotlib and numpy are required for visualisation. "
            "Install with: pip install matplotlib numpy"
        )

    n = len(pairs)

    scores   = np.array([p["score"]               for p in pairs], dtype=float)
    lazy_f   = np.array([p["lazy_f_corrections"]   for p in pairs], dtype=float)
    farrar_t = np.array([p["farrar_time_s"]        for p in pairs], dtype=float)
    sw_t     = np.array([p["smith_waterman_time_s"] for p in pairs], dtype=float)

    has_timing = farrar_t.any() or sw_t.any()
    ncols = 2 if has_timing else 1
    fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols + 2, 4))
    if ncols == 1:
        axes = [axes]
    fig.suptitle(f"Averaged summary — {n} pair{'s' if n != 1 else ''}", fontsize=10)

    def _bar(ax, labels, means, stds, colors, title, ylabel):
        x = np.arange(len(labels))
        bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, width=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=8)
        # Annotate mean ± std above each bar.
        for xi, (m, s) in enumerate(zip(means, stds)):
            ax.text(xi, m + s + max(means) * 0.03 if max(means) else s,
                    f"{m:.3g}±{s:.3g}", ha="center", va="bottom", fontsize=7)

    _bar(
        axes[0],
        labels=["score", "lazy-F corrections"],
        means=[scores.mean(), lazy_f.mean()],
        stds=[scores.std(), lazy_f.std()],
        colors=["steelblue", "darkorange"],
        title="Alignment metrics (mean ± std)",
        ylabel="value",
    )

    if has_timing:
        _bar(
            axes[1],
            labels=["farrar", "scalar DP"],
            means=[farrar_t.mean(), sw_t.mean()],
            stds=[farrar_t.std(), sw_t.std()],
            colors=["steelblue", "darkorange"],
            title="Timing (mean ± std)",
            ylabel="seconds",
        )

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# File output
# ---------------------------------------------------------------------------

def write_output(path: str, pairs_data: list[dict], pen, args: object) -> None:
    """Write alignment results to path as JSON."""
    pen_list = list(pen)
    meta = {
        "implementation": getattr(args, "implementation", "farrar"),
        "penalties": {
            "match":    pen_list[0],
            "mismatch": pen_list[1],
            "del_open": pen_list[2],
            "del_ext":  pen_list[3],
            "ins_open": pen_list[4],
            "ins_ext":  pen_list[5],
        },
        "lanes": getattr(args, "lanes", 8),
    }
    with Path(path).expanduser().open("w", encoding="utf-8") as f:
        json.dump({"metadata": meta, "pairs": pairs_data}, f, indent=2)


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def unnumbered_stem(path: Path) -> str:
    stem = path.stem
    while stem and stem[-1].isdigit():
        stem = stem[:-1]
    return stem or path.stem


def numbered_path(path: str, run_number: int) -> str:
    expanded = Path(path).expanduser()
    stem = unnumbered_stem(expanded)
    return str(expanded.with_name(f"{stem}{run_number}{expanded.suffix}"))


def next_output_path(path_template: str) -> str:
    run_number = 1
    while True:
        candidate = numbered_path(path_template, run_number)
        if not Path(candidate).exists():
            return candidate
        run_number += 1
