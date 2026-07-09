#!/usr/bin/env python3
"""
plot_breakdown.py — preprocessing-vs-GPU-kernel breakdown figures from the
sampled-100 sheet.csv (produced by make_sheet.py).

Two panels:
  (left)  grouped stacked bars (prep + kernel) for dense / cufft-smooth /
          mixed-radix on the matrices where ALL THREE complete -> shows dense's
          preprocessing (dense-grid scatter + cufftPlan2d) dominating its total.
  (right) prep fraction of total vs matrix size for cufft-smooth (all completing).

Usage: python3 scripts/plot_breakdown.py results_sampled/sheet.csv figures_sampled/breakdown.png
"""
import csv, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

def f(x):
    try: return float(x)
    except: return None

def main():
    sheet = sys.argv[1] if len(sys.argv) > 1 else "results_sampled/sheet.csv"
    out   = sys.argv[2] if len(sys.argv) > 2 else "figures_sampled/breakdown.png"
    rows = list(csv.DictReader(open(sheet)))

    M = [("dense", "Dense cuFFT", "#4c78a8"),
         ("cufftsmooth", "CSC cuFFT-smooth", "#f58518"),
         ("mixedradix", "CSC mixed-radix", "#54a24b")]

    tri = [r for r in rows if all(r[f"{k}_status"] == "OK" for k, _, _ in M)]
    tri.sort(key=lambda r: int(r["rows"]))

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 6))

    # ---- left: grouped stacked prep+kernel bars ----
    x = np.arange(len(tri)); w = 0.26
    for i, (k, lab, c) in enumerate(M):
        prep = np.array([f(r[f"{k}_prep_ms"]) or 0 for r in tri])
        kern = np.array([f(r[f"{k}_kernel_ms"]) or 0 for r in tri])
        off = (i - 1) * w
        axL.bar(x + off, prep, w, color=c, alpha=0.45,
                label=f"{lab} — preprocessing" if i == 0 else None)
        axL.bar(x + off, kern, w, bottom=prep, color=c,
                label=f"{lab} — kernel" if i == 0 else lab)
    # custom legend: color = method, alpha = prep vs kernel
    from matplotlib.patches import Patch
    leg = [Patch(fc=c, label=lab) for _, lab, c in M] + \
          [Patch(fc="#777", alpha=0.45, label="preprocessing (lower)"),
           Patch(fc="#777", label="kernel (upper)")]
    axL.legend(handles=leg, fontsize=8, loc="upper left")
    axL.set_xticks(x)
    axL.set_xticklabels([f"{r['matrix'][:14]}\n{r['rows']}" for r in tri],
                        rotation=90, fontsize=6.5)
    axL.set_ylabel("time (ms)")
    axL.set_title(f"Prep + kernel, matrices where all 3 complete (n={len(tri)})")
    axL.grid(axis="y", alpha=0.3)

    # ---- right: prep fraction vs size for cufft-smooth ----
    cs = [r for r in rows if r["cufftsmooth_status"] == "OK"]
    size = np.array([int(r["rows"]) for r in cs])
    prep = np.array([f(r["cufftsmooth_prep_ms"]) or 0 for r in cs])
    kern = np.array([f(r["cufftsmooth_kernel_ms"]) or 1 for r in cs])
    frac = prep / (prep + kern)
    axR.scatter(size, frac, s=28, color="#f58518", edgecolor="k", linewidth=0.3)
    axR.axhline(np.median(frac), ls="--", color="#555",
                label=f"median {np.median(frac):.2f}")
    axR.set_xscale("log")
    axR.set_xlabel("matrix dimension (rows)")
    axR.set_ylabel("preprocessing / total")
    axR.set_ylim(0, 1)
    axR.set_title(f"CSC cuFFT-smooth: preprocessing fraction (n={len(cs)})")
    axR.legend(fontsize=9); axR.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")

if __name__ == "__main__":
    main()
