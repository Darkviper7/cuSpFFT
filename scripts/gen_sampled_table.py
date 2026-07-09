#!/usr/bin/env python3
"""
gen_sampled_table.py — emit the LaTeX longtable body (100 rows) for the
sampled-100 per-matrix appendix table in draftPaper/2DcuSpFFT/main.tex.

Reads results_sampled_v2/sheet.csv (per-method prep/kernel/total/status) and
sampled_matrices.json (SuiteSparse group/name). Sorted by min(rows,cols).

Columns emitted per row (10):
  \textsf{group/name} & R$\times$C & nnz & dens% &
    dense(k/p) & mixed(k/p) & smooth(k/p) & sym(k/p) &
    smooth-vs-dense(kernel) & sym-vs-dense(kernel)
where k/p = kernel/preprocessing time (ms). Runtimes use integer ms >= 100 and
one decimal otherwise to keep the portrait table narrow. Failed cells show
OOM/PLAN; speedups show '--' when dense does not complete.

Usage: python3 scripts/gen_sampled_table.py > /tmp/sampled_rows.tex
"""
import csv, json, sys, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sheet CSV path: first CLI arg, else the default 4090 sheet.
SHEET = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    ROOT, "results_sampled_v2", "sheet.csv")
JSON = "/home/smanthe/Documents/Binary_Sparse_FFT_GPU/sampled_matrices.json"


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def tex_escape(s):
    return s.replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")


def fmt(x):
    if x is None:
        return ""
    return f"{x:.0f}" if x >= 100 else f"{x:.1f}"


def group_map():
    d = json.load(open(JSON))
    return {m["name"]: m["group"] for b in d["buckets"] for m in b["matrices"]}


# kernel/prep cell for a method: "k/p" if OK, else the failure token
def cell(r, key):
    st = r[f"{key}_status"]
    if st == "OK":
        return f"{fmt(f(r[f'{key}_kernel_ms']))}/{fmt(f(r[f'{key}_prep_ms']))}"
    if st in ("OOM", "PLAN", "TIMEOUT", "FAIL", "DNF"):
        return r"\textsf{" + st + "}"
    return "--"


# peak-memory cell (MB, integer) for a method; blank when it didn't complete
# (the kernel column already flags OOM/PLAN). Reads mem_mb from the raw per-method
# sheet columns if present, else "".
def mem_cell(r, key):
    if r[f"{key}_status"] != "OK":
        return ""
    m = f(r.get(f"{key}_mem_mb"))
    return f"{m:,.0f}" if m is not None else ""


# kernel-time speedup of `key` over dense, '--' when dense (or key) failed
def speedup_vs_dense(r, key):
    if r["dense_status"] == "OK" and r[f"{key}_status"] == "OK":
        dk, xk = f(r["dense_kernel_ms"]), f(r[f"{key}_kernel_ms"])
        if dk and xk:
            return f"{dk/xk:.2f}$\\times$"
    return "--"


def main():
    grp = group_map()
    rows = list(csv.DictReader(open(SHEET)))
    rows.sort(key=lambda r: min(int(r["rows"]), int(r["cols"])))
    n = 0
    for r in rows:
        name = r["matrix"]
        full = f"{grp.get(name, '')}/{name}" if grp.get(name) else name
        shape = f"{int(r['rows']):,}$\\times${int(r['cols']):,}"
        nnz = f"{int(r['nnz']):,}"
        dens = f(r["density"])
        densf = f"{dens*100:.3g}" if dens else ""
        cols = [
            r"\textsf{" + tex_escape(full) + "}",
            shape, nnz, densf,
            cell(r, "dense"), cell(r, "mixedradix"),
            cell(r, "cufftsmooth"), cell(r, "cufftsmoothsym"),
            mem_cell(r, "dense"), mem_cell(r, "mixedradix"),
            mem_cell(r, "cufftsmooth"), mem_cell(r, "cufftsmoothsym"),
            speedup_vs_dense(r, "cufftsmooth"),
            speedup_vs_dense(r, "cufftsmoothsym"),
        ]
        print(" & ".join(cols) + r" \\")
        n += 1
    print(f"% emitted {n} rows", file=sys.stderr)


if __name__ == "__main__":
    main()
