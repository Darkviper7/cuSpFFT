#!/usr/bin/env python3
"""Generate report-ready figures from logs/results.csv.

Reads the sweep CSV produced by scripts/run_sweep.sh + scripts/parse_log.awk
and writes a coherent figure suite into figures/. All five plots share a
fixed method palette and ordering so figures are visually comparable.

Usage:
    .venv/bin/python scripts/plot_results.py            # all five
    .venv/bin/python scripts/plot_results.py --plot runtime_overview
    .venv/bin/python scripts/plot_results.py --csv logs/results.csv --out figures/

Plots:
    runtime_overview.png    — every matrix, log-scale runtime, methods grouped
    memory_overview.png     — every matrix, log-scale peak memory
    headline_buckets.png    — 2x2: small / medium / large / power-of-2 panels
    failure_grid.png        — matrix x method status heatmap
    speedup_vs_size.png     — log-log scatter of sparse/dense speedup vs size
    density_scaling.png     — per-cell runtime vs density, log-log, per method
"""

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

# ---- Stable method ordering / palette ---------------------------------------

METHOD_ORDER = [
    "CPU FFT reference (FFTW3)",
    "Dense cuFFT (baseline)",
    "SpFFT (baseline)",
    "Sparse CSC (mixed-radix)",
    "Sparse CSC (cufft smooth)",
]
PALETTE = {
    "CPU FFT reference (FFTW3)": "#7f7f7f",
    "Dense cuFFT (baseline)":    "#4878d0",  # muted blue
    "SpFFT (baseline)":          "#ee854a",  # muted orange
    "Sparse CSC (mixed-radix)":  "#6acc64",  # muted green
    "Sparse CSC (cufft smooth)": "#d65f5f",  # muted red
}
SHORT = {
    "CPU FFT reference (FFTW3)": "CPU FFTW3",
    "Dense cuFFT (baseline)":    "Dense cuFFT",
    "SpFFT (baseline)":          "SpFFT",
    "Sparse CSC (mixed-radix)":  "CSC mixed-radix",
    "Sparse CSC (cufft smooth)": "CSC cuFFT-smooth",
}

# Status palette for the failure grid.
STATUS_COLORS = {
    "ok":       "#4daf4a",
    "OOM":      "#ff7f00",
    "PLAN":     "#e41a1c",
    "HOST OOM": "#984ea3",
    "ERR":      "#a65628",
    "N/A":      "#dddddd",
}
STATUS_ORDER = ["ok", "OOM", "PLAN", "HOST OOM", "ERR", "N/A"]


# ---- Loader -----------------------------------------------------------------

def is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def classify_error(err: str) -> str:
    if not isinstance(err, str) or not err:
        return "ok"
    s = err.lower()
    if "code: 5" in s or "cufft_internal_error" in s:
        return "PLAN"
    if "host" in s and "alloc" in s:
        return "HOST OOM"
    if "out of memory" in s or "oom" in s or "bad_alloc" in s:
        return "OOM"
    return "ERR"


def load(csv_path: Path, small_max: int, medium_max: int) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    for col in ("rows", "cols", "nnz", "median_ms", "min_ms", "max_ms",
                "n_iters", "mem_mb", "preprocess_ms", "max_abs", "max_rel", "rms_abs"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["density"] = df["nnz"] / (df["rows"] * df["cols"])
    df["area"] = df["rows"] * df["cols"]
    df["pow2"] = df["cols"].apply(lambda c: is_power_of_two(int(c)) if pd.notna(c) else False)
    df["bucket"] = pd.cut(
        df["rows"],
        bins=[-1, small_max, medium_max, np.inf],
        labels=["small", "medium", "large"],
    )
    df["status"] = df["error"].apply(classify_error)
    return df


def methods_present(df: pd.DataFrame) -> list[str]:
    """Return methods in stable order, skipping any not in the data."""
    seen = set(df["method"].unique())
    return [m for m in METHOD_ORDER if m in seen]


# ---- Plot 1: per-matrix runtime overview ------------------------------------

def plot_runtime_overview(df: pd.DataFrame, out: Path):
    methods = methods_present(df)
    matrices = (df.drop_duplicates("matrix")
                  .sort_values("area")["matrix"].tolist())

    fig_h = max(6, 0.30 * len(matrices))
    fig, ax = plt.subplots(figsize=(13, fig_h))

    n = len(methods)
    bar_h = 0.85 / n
    y_centers = np.arange(len(matrices))

    for i, method in enumerate(methods):
        sub = (df[df["method"] == method]
               .set_index("matrix")
               .reindex(matrices))
        offsets = y_centers - 0.425 + (i + 0.5) * bar_h
        ms = sub["median_ms"].to_numpy()
        statuses = sub["status"].fillna("N/A").to_numpy()

        ok = (statuses == "ok") & np.isfinite(ms) & (ms > 0)
        ax.barh(offsets[ok], ms[ok], bar_h,
                color=PALETTE[method], label=SHORT[method])

        for off, st in zip(offsets[~ok], statuses[~ok]):
            if st in ("N/A",):
                continue
            ax.text(0.05, off, st, va="center", ha="left",
                    fontsize=6.5, color=PALETTE[method], fontweight="bold")

    ax.set_yticks(y_centers)
    ax.set_yticklabels(matrices, fontsize=7)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlim(left=0.05)
    ax.set_xlabel("median runtime (ms, log scale)")
    ax.set_title("Per-matrix runtime — sorted by matrix area "
                 "(failures shown as labels at left edge)")
    ax.grid(axis="x", which="both", alpha=0.25)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---- Plot 2: per-matrix memory overview -------------------------------------

def plot_memory_overview(df: pd.DataFrame, out: Path):
    methods = methods_present(df)
    matrices = (df.drop_duplicates("matrix")
                  .sort_values("area")["matrix"].tolist())

    fig_h = max(6, 0.30 * len(matrices))
    fig, ax = plt.subplots(figsize=(13, fig_h))

    n = len(methods)
    bar_h = 0.85 / n
    y_centers = np.arange(len(matrices))

    for i, method in enumerate(methods):
        sub = (df[df["method"] == method]
               .set_index("matrix")
               .reindex(matrices))
        offsets = y_centers - 0.425 + (i + 0.5) * bar_h
        mb = sub["mem_mb"].to_numpy()
        statuses = sub["status"].fillna("N/A").to_numpy()

        ok = (statuses == "ok") & np.isfinite(mb) & (mb > 0)
        ax.barh(offsets[ok], mb[ok], bar_h,
                color=PALETTE[method], label=SHORT[method])

        for off, st in zip(offsets[~ok], statuses[~ok]):
            if st in ("N/A",):
                continue
            ax.text(0.5, off, st, va="center", ha="left",
                    fontsize=6.5, color=PALETTE[method], fontweight="bold")

    ax.set_yticks(y_centers)
    ax.set_yticklabels(matrices, fontsize=7)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlim(left=1.0)
    ax.set_xlabel("peak device memory (MB, log scale)")
    ax.set_title("Per-matrix peak memory — sorted by matrix area")
    ax.grid(axis="x", which="both", alpha=0.25)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---- Plot 3: 2x2 headline buckets -------------------------------------------

def _pick_bucket_matrices(df: pd.DataFrame, *, max_per_panel: int | None = None):
    """Group matrices by bucket. With max_per_panel set, trim each list to
    that many evenly-spaced representatives; otherwise include every matrix
    in the bucket."""
    by_matrix = df.drop_duplicates("matrix").set_index("matrix")
    buckets = {
        "small":       sorted(by_matrix[by_matrix["bucket"] == "small"].index,
                              key=lambda m: by_matrix.at[m, "area"]),
        "medium":      sorted(by_matrix[by_matrix["bucket"] == "medium"].index,
                              key=lambda m: by_matrix.at[m, "area"]),
        "large":       sorted(by_matrix[by_matrix["bucket"] == "large"].index,
                              key=lambda m: by_matrix.at[m, "area"]),
        "power-of-2":  sorted(by_matrix[by_matrix["pow2"]].index,
                              key=lambda m: by_matrix.at[m, "area"]),
    }
    if max_per_panel is None:
        return buckets
    out = {}
    for key, ms in buckets.items():
        if len(ms) <= max_per_panel:
            out[key] = ms
        else:
            idx = np.linspace(0, len(ms) - 1, max_per_panel).round().astype(int)
            out[key] = [ms[i] for i in idx]
    return out


def _fmt_ms(v: float) -> str:
    """Compact ms label: more precision at small magnitudes, none at large."""
    if v < 1:    return f"{v:.3f}"
    if v < 10:   return f"{v:.2f}"
    if v < 100:  return f"{v:.1f}"
    if v < 10_000: return f"{v:.0f}"
    return f"{v/1000:.1f}k"


def _fmt_mb(v: float) -> str:
    """Compact MB label: thousands as 'GB' for readability."""
    if v < 10:    return f"{v:.2f}"
    if v < 100:   return f"{v:.1f}"
    if v < 1000:  return f"{v:.0f}"
    return f"{v/1000:.1f}G"


def _draw_bucket_panel(ax, df: pd.DataFrame, methods, matrices, title: str,
                       *, value_col: str = "median_ms",
                       ylabel: str = "execution time (ms)",
                       fmt_fn=_fmt_ms):
    """Render one bucket's grouped-bar panel onto `ax`. Shared by every
    bucket plot, both runtime (`value_col='median_ms'`) and memory
    (`value_col='mem_mb'`).

    Failed/not-attempted (matrix, method) cells get a small marker plus
    short caption near the x-axis baseline (no stub bar)."""
    if not matrices:
        ax.set_title(f"{title}\n(no matrices in this bucket)")
        ax.set_xticks([])
        ax.set_yticks([])
        return

    x = np.arange(len(matrices))
    n = len(methods)
    bar_w = 0.85 / n

    fail_records = []   # (offset, marker, color)
    matrix_fail_reasons = {}  # x_idx -> dominant reason among failed methods
    for i, method in enumerate(methods):
        sub = (df[df["method"] == method]
               .set_index("matrix")
               .reindex(matrices))
        offsets = x - 0.425 + (i + 0.5) * bar_w
        vals = sub[value_col].to_numpy()
        statuses = sub["status"].fillna("N/A").to_numpy()

        ok = (statuses == "ok") & np.isfinite(vals) & (vals > 0)
        ax.bar(offsets[ok], vals[ok], bar_w,
               color=PALETTE[method], label=SHORT[method],
               edgecolor="white", linewidth=0.4, zorder=2)
        for off, v in zip(offsets[ok], vals[ok]):
            ax.text(off, v * 1.06, fmt_fn(v),
                    ha="center", va="bottom",
                    rotation=90, fontsize=7,
                    color="#444444", zorder=3)

        for j, (off, st) in enumerate(zip(offsets[~ok], statuses[~ok])):
            if st == "N/A":
                fail_records.append((off, "·", PALETTE[method]))
            else:
                fail_records.append((off, "×", PALETTE[method]))
                # Track per-matrix reason for the matrix-level annotation
                xi = np.where(~ok)[0][j]
                matrix_fail_reasons.setdefault(xi, st)

    ax.set_xticks(x)
    ax.set_xticklabels(matrices, rotation=30, ha="right", fontsize=9)
    ax.set_yscale("log")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=12, pad=10)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", which="major", color="#dddddd",
            linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)

    # Headroom at top so the rotated value labels above tall bars don't
    # clip and the legend (placed above the panel by the caller) doesn't
    # overlap. Log scale, so the multiplier compounds.
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax * 2.5)

    # Render failure glyphs at axes-fraction y near the x-axis. Use a
    # blended transform so x is in data coords (matches bars) but y is
    # in axes-fraction (stays put when log limits change). Per-method
    # colored ×/· markers; reason text is per-matrix (centered under
    # the group), shown only when at least one method failed there.
    if fail_records:
        from matplotlib.transforms import blended_transform_factory
        trans = blended_transform_factory(ax.transData, ax.transAxes)
        for off, marker, color in fail_records:
            ax.text(off, 0.06, marker, transform=trans,
                    ha="center", va="center", color=color,
                    fontsize=10, fontweight="bold", zorder=3)
        for xi, reason in matrix_fail_reasons.items():
            ax.text(xi, 0.015, reason, transform=trans,
                    ha="center", va="bottom", color="#666666",
                    fontsize=7, zorder=3)


_BUCKET_TITLES = {
    "small":      "Small matrices (rows ≤ 5,000)",
    "medium":     "Medium matrices (5,000 < rows ≤ 20,000)",
    "large":      "Large matrices (rows > 20,000)",
    "power-of-2": "Power-of-two column dimensions",
}


def plot_headline_buckets(df: pd.DataFrame, out: Path,
                          max_per_panel: int | None = None):
    methods = methods_present(df)
    picks = _pick_bucket_matrices(df, max_per_panel=max_per_panel)

    largest = max((len(v) for v in picks.values()), default=0)
    panel_w = max(7.0, 0.55 * largest)
    fig_w = panel_w * 2
    fig_h = max(10, 0.32 * largest + 7)

    with plt.rc_context(_PAPER_RC):
        fig, axs = plt.subplots(2, 2, figsize=(fig_w, fig_h))
        layout = [
            ("small",      axs[0, 0]),
            ("medium",     axs[0, 1]),
            ("large",      axs[1, 0]),
            ("power-of-2", axs[1, 1]),
        ]
        for key, ax in layout:
            _draw_bucket_panel(ax, df, methods, picks[key],
                               _BUCKET_TITLES[key])

        handles = [mpatches.Patch(color=PALETTE[m], label=SHORT[m])
                   for m in methods]
        fig.legend(handles=handles, loc="upper center",
                   ncol=len(methods), bbox_to_anchor=(0.5, 1.005),
                   fontsize=9, frameon=False)
        fig.suptitle("Median execution time across matrix size and shape regimes",
                     y=1.03)
        fig.tight_layout()
        fig.savefig(out, dpi=220, bbox_inches="tight")
        plt.close(fig)


_PAPER_RC = {
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.edgecolor":    "#444444",
    "axes.labelcolor":   "#222222",
    "axes.titlesize":    12,
    "axes.titleweight":  "regular",
    "axes.labelsize":    10.5,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "xtick.color":       "#444444",
    "ytick.color":       "#444444",
    "grid.color":        "#dddddd",
    "grid.linewidth":    0.6,
    "legend.frameon":    False,
    "legend.fontsize":   9,
    "font.size":         10,
}


def _plot_single_bucket(df: pd.DataFrame, out: Path, key: str,
                        *, value_col: str = "median_ms",
                        ylabel: str = "execution time (ms)",
                        fmt_fn=_fmt_ms):
    """Render one bucket as its own paper-ready PNG."""
    methods = methods_present(df)
    picks = _pick_bucket_matrices(df, max_per_panel=None)
    matrices = picks[key]

    fig_w = max(7.0, 0.55 * max(len(matrices), 1))
    fig_h = 5.5
    with plt.rc_context(_PAPER_RC):
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        _draw_bucket_panel(ax, df, methods, matrices, _BUCKET_TITLES[key],
                           value_col=value_col, ylabel=ylabel, fmt_fn=fmt_fn)

        handles = [mpatches.Patch(color=PALETTE[m], label=SHORT[m])
                   for m in methods]
        fig.legend(handles=handles, loc="upper center",
                   ncol=len(methods), bbox_to_anchor=(0.5, 1.04),
                   fontsize=9, frameon=False, handlelength=1.2,
                   columnspacing=1.4)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(out, dpi=220, bbox_inches="tight")
        plt.close(fig)


def plot_bucket_small(df: pd.DataFrame, out: Path):
    _plot_single_bucket(df, out, "small")


def plot_bucket_medium(df: pd.DataFrame, out: Path):
    _plot_single_bucket(df, out, "medium")


def plot_bucket_large(df: pd.DataFrame, out: Path):
    _plot_single_bucket(df, out, "large")


def plot_bucket_pow2(df: pd.DataFrame, out: Path):
    _plot_single_bucket(df, out, "power-of-2")


def plot_bucket_small_mem(df: pd.DataFrame, out: Path):
    _plot_single_bucket(df, out, "small",
                        value_col="mem_mb", ylabel="peak memory (MB)",
                        fmt_fn=_fmt_mb)


def plot_bucket_medium_mem(df: pd.DataFrame, out: Path):
    _plot_single_bucket(df, out, "medium",
                        value_col="mem_mb", ylabel="peak memory (MB)",
                        fmt_fn=_fmt_mb)


def plot_bucket_large_mem(df: pd.DataFrame, out: Path):
    _plot_single_bucket(df, out, "large",
                        value_col="mem_mb", ylabel="peak memory (MB)",
                        fmt_fn=_fmt_mb)


def plot_bucket_pow2_mem(df: pd.DataFrame, out: Path):
    _plot_single_bucket(df, out, "power-of-2",
                        value_col="mem_mb", ylabel="peak memory (MB)",
                        fmt_fn=_fmt_mb)


# ---- Plot 4: failure grid ---------------------------------------------------

def plot_failure_grid(df: pd.DataFrame, out: Path):
    methods = methods_present(df)
    matrices = (df.drop_duplicates("matrix")
                  .sort_values("area")["matrix"].tolist())

    grid = pd.DataFrame("N/A", index=matrices, columns=methods)
    for _, row in df.iterrows():
        grid.at[row["matrix"], row["method"]] = row["status"]

    color_index = {s: i for i, s in enumerate(STATUS_ORDER)}
    Z = np.array([[color_index[grid.at[m, k]] for k in methods]
                  for m in matrices], dtype=int)

    cmap = ListedColormap([STATUS_COLORS[s] for s in STATUS_ORDER])

    fig, ax = plt.subplots(figsize=(8, max(6, 0.28 * len(matrices))))
    ax.imshow(Z, aspect="auto", cmap=cmap,
              vmin=-0.5, vmax=len(STATUS_ORDER) - 0.5,
              interpolation="nearest")

    for i, m in enumerate(matrices):
        for j, k in enumerate(methods):
            st = grid.at[m, k]
            if st not in ("ok", "N/A"):
                ax.text(j, i, st, ha="center", va="center",
                        fontsize=6.5, color="white", fontweight="bold")

    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([SHORT[k] for k in methods], rotation=20, ha="right")
    ax.set_yticks(range(len(matrices)))
    ax.set_yticklabels(matrices, fontsize=7)
    ax.set_title("Per-(matrix, method) outcome — green = success")

    legend_handles = [mpatches.Patch(color=STATUS_COLORS[s], label=s)
                      for s in STATUS_ORDER]
    ax.legend(handles=legend_handles, loc="upper left",
              bbox_to_anchor=(1.02, 1.0), fontsize=8, frameon=False)

    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---- Plot 5: speedup vs size ------------------------------------------------

def plot_speedup_vs_size(df: pd.DataFrame, out: Path):
    sparse_label = "Sparse CSC (cufft smooth)"
    dense_label  = "Dense cuFFT (baseline)"

    if sparse_label not in df["method"].unique():
        # Fall back to mixed-radix if cuFFT-smooth isn't in the sweep.
        sparse_label = "Sparse CSC (mixed-radix)"

    sparse = (df[df["method"] == sparse_label]
              .set_index("matrix")[["median_ms", "status", "area", "pow2"]])
    dense  = (df[df["method"] == dense_label]
              .set_index("matrix")[["median_ms", "status"]]
              .rename(columns={"median_ms": "dense_ms",
                               "status":    "dense_status"}))
    j = sparse.join(dense, how="outer")
    j["speedup"] = j["dense_ms"] / j["median_ms"]   # > 1 means sparse wins

    fig, ax = plt.subplots(figsize=(11, 7))

    both_ok = (j["status"] == "ok") & (j["dense_status"] == "ok")
    pow2 = j["pow2"].fillna(False)

    ax.scatter(j.loc[both_ok & ~pow2, "area"],
               j.loc[both_ok & ~pow2, "speedup"],
               s=55, c="#1f77b4", edgecolor="black",
               linewidth=0.6, label="general shape", zorder=3)
    ax.scatter(j.loc[both_ok & pow2, "area"],
               j.loc[both_ok & pow2, "speedup"],
               s=55, c="#d62728", edgecolor="black",
               linewidth=0.6, label="power-of-2 cols", zorder=3)

    # Annotate dense-failure cases at the top edge of the plot
    dense_failed = (j["status"] == "ok") & (j["dense_status"] != "ok")
    if dense_failed.any():
        ymax = j.loc[both_ok, "speedup"].max() * 3
        sub = j[dense_failed]
        ax.scatter(sub["area"], np.full(len(sub), ymax),
                   marker="^", s=80, c="#2ca02c", edgecolor="black",
                   linewidth=0.6, label="dense failed → sparse-only", zorder=3)
        for matrix, row in sub.iterrows():
            ax.annotate(matrix, (row["area"], ymax),
                        xytext=(3, 3), textcoords="offset points", fontsize=7)

    ax.axhline(1.0, color="black", linestyle="--", linewidth=1, zorder=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("matrix area (rows × cols, log scale)")
    ax.set_ylabel(f"speedup: {SHORT[dense_label]} / {SHORT[sparse_label]}")
    ax.set_title("Sparse vs. dense speedup as a function of matrix size and shape")
    ax.grid(which="both", alpha=0.25)
    ax.legend(loc="best", fontsize=9)

    # Annotate notable matrices
    for matrix in ("sstmodel", "benzene", "delaunay_n14", "pct20stif"):
        if matrix in j.index and not pd.isna(j.at[matrix, "speedup"]):
            ax.annotate(matrix,
                        (j.at[matrix, "area"], j.at[matrix, "speedup"]),
                        xytext=(5, 5), textcoords="offset points",
                        fontsize=7.5, alpha=0.85)

    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---- Plot 6: density scaling -------------------------------------------------

BUCKET_MARKERS = {"small": "o", "medium": "s", "large": "^"}


def plot_density_scaling(df: pd.DataFrame, out: Path):
    """Per-cell runtime vs input density, log-log, one series per method.

    Y is `median_ms / area * 1e6` (nanoseconds per output cell). The per-method
    log-log slope is fitted by least-squares and reported in the legend so the
    empirical density-dependence is a single readable number rather than a
    visual guess.
    """
    from matplotlib.lines import Line2D

    work = df[df["status"] == "ok"].copy()
    work = work.dropna(subset=["median_ms", "rows", "cols", "nnz"])
    work["ns_per_cell"] = work["median_ms"] * 1e6 / work["area"]
    methods = methods_present(work)

    plt.rcParams.update({
        "axes.labelsize":  12,
        "axes.titlesize":  13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
    })

    fig, ax = plt.subplots(figsize=(10, 6.5))

    method_legend = []
    for method in methods:
        sub = work[work["method"] == method]
        if sub.empty:
            continue
        # Scatter, marker shape per size bucket
        for bucket, marker in BUCKET_MARKERS.items():
            bsub = sub[sub["bucket"] == bucket]
            if bsub.empty:
                continue
            ax.scatter(bsub["density"], bsub["ns_per_cell"],
                       c=PALETTE[method], marker=marker,
                       s=70, alpha=0.85,
                       edgecolor="black", linewidth=0.5,
                       zorder=3)

        # Log-log linear fit
        valid = sub.dropna(subset=["density", "ns_per_cell"])
        valid = valid[(valid["density"] > 0) & (valid["ns_per_cell"] > 0)]
        slope_str = ""
        if len(valid) >= 3:
            slope, intercept = np.polyfit(np.log10(valid["density"]),
                                          np.log10(valid["ns_per_cell"]), 1)
            xs = np.logspace(np.log10(valid["density"].min()),
                             np.log10(valid["density"].max()), 60)
            ys = 10 ** (slope * np.log10(xs) + intercept)
            ax.plot(xs, ys, color=PALETTE[method],
                    linestyle="--", linewidth=1.5, alpha=0.85, zorder=2)
            slope_str = f"  (slope = {slope:+.2f})"

        method_legend.append(
            mpatches.Patch(color=PALETTE[method],
                           label=f"{SHORT[method]}{slope_str}"))

    bucket_legend = [
        Line2D([0], [0], marker=BUCKET_MARKERS[b], color="black",
               linestyle="", markerfacecolor="white",
               markersize=9, markeredgewidth=0.8, label=f"{b} matrix")
        for b in ("small", "medium", "large")
    ]

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("input density  (nnz / rows·cols)")
    ax.set_ylabel("per-cell runtime  (ns / output cell)")
    ax.set_title("Per-cell runtime vs. input density")
    ax.grid(which="both", alpha=0.25)

    # Annotate marquee matrices so the abstract scatter grounds in known runs.
    marquee = ("sstmodel", "benzene", "delaunay_n14", "pct20stif",
               "kron_g500-logn16", "amazon0302")
    for matrix in marquee:
        sub = work[work["matrix"] == matrix]
        if sub.empty:
            continue
        # Use one point per matrix (whichever method is present) for the label
        row = sub.iloc[0]
        ax.annotate(matrix,
                    (row["density"], row["ns_per_cell"]),
                    xytext=(6, 6), textcoords="offset points",
                    fontsize=8, alpha=0.85,
                    arrowprops=dict(arrowstyle="-", lw=0.5, alpha=0.4))

    leg1 = ax.legend(handles=method_legend, loc="upper left",
                     title="method", frameon=True, framealpha=0.95)
    ax.add_artist(leg1)
    ax.legend(handles=bucket_legend, loc="lower left",
              title="size bucket", frameon=True, framealpha=0.95)

    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    plt.rcdefaults()


# ---- CLI --------------------------------------------------------------------

PLOTS = {
    "runtime_overview":   plot_runtime_overview,
    "memory_overview":    plot_memory_overview,
    "headline_buckets":   plot_headline_buckets,
    "bucket_small":       plot_bucket_small,
    "bucket_medium":      plot_bucket_medium,
    "bucket_large":       plot_bucket_large,
    "bucket_pow2":        plot_bucket_pow2,
    "bucket_small_mem":   plot_bucket_small_mem,
    "bucket_medium_mem":  plot_bucket_medium_mem,
    "bucket_large_mem":   plot_bucket_large_mem,
    "bucket_pow2_mem":    plot_bucket_pow2_mem,
    "failure_grid":       plot_failure_grid,
    "speedup_vs_size":    plot_speedup_vs_size,
    "density_scaling":    plot_density_scaling,
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", default="logs/results.csv", type=Path)
    p.add_argument("--out", default="figures", type=Path)
    p.add_argument("--plot", choices=list(PLOTS), action="append",
                   help="Render only the named plot (repeatable). Default: all.")
    p.add_argument("--small-max",  type=int, default=5_000)
    p.add_argument("--medium-max", type=int, default=20_000)
    p.add_argument("--headline-max-per-panel", type=int, default=None,
                   help="Cap the number of matrices per panel in headline_buckets "
                        "(evenly spaced through each bucket). Default: no cap.")
    args = p.parse_args()

    if not args.csv.exists():
        raise SystemExit(f"error: {args.csv} not found. "
                         f"Run scripts/run_sweep.sh or backfill the CSV first.")
    args.out.mkdir(parents=True, exist_ok=True)

    df = load(args.csv, args.small_max, args.medium_max)
    print(f"Loaded {len(df)} rows / {df['matrix'].nunique()} matrices "
          f"/ {df['method'].nunique()} methods from {args.csv}")

    targets = args.plot or list(PLOTS)
    for name in targets:
        path = args.out / f"{name}.png"
        if name == "headline_buckets":
            PLOTS[name](df, path, max_per_panel=args.headline_max_per_panel)
        else:
            PLOTS[name](df, path)
        print(f"  wrote {path}")


if __name__ == "__main__":
    main()
