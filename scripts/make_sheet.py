#!/usr/bin/env python3
"""
make_sheet.py — pivot the sampled-100 sweep CSV into a per-matrix comparison
sheet: one row per matrix, grouped {preprocess, kernel, total, status} columns
for dense cuFFT, CSC cuFFT-smooth, and CSC mixed-radix.

Input : the merged results.csv produced by run_sampled_sweep.sh
        (16-col schema incl. preprocess_ms), plus an optional env.txt GPU stamp.
Output: <out>.csv  (Excel/Sheets-openable) and <out>.html (color-coded table).

Usage:
    python3 scripts/make_sheet.py <results.csv> <out_prefix> [env.txt]
Example:
    python3 scripts/make_sheet.py logs_sampled/results.csv results_sampled/sheet
"""
import csv
import html
import os
import sys

# method label (as printed by the binary / stored in CSV) -> short column key
METHOD_KEY = {
    "Dense cuFFT (baseline)": "dense",
    "Sparse CSC (cufft smooth)": "cufftsmooth",
    "Sparse CSC (cufft smooth sym)": "cufftsmoothsym",
    "Sparse CSC (mixed-radix)": "mixedradix",
}
METHOD_ORDER = ["dense", "cufftsmooth", "cufftsmoothsym", "mixedradix"]
METHOD_TITLE = {
    "dense": "Dense cuFFT",
    "cufftsmooth": "CSC cuFFT-smooth",
    "cufftsmoothsym": "CSC cuFFT-smooth-sym",
    "mixedradix": "CSC mixed-radix",
}


def fnum(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def classify(err):
    """Map the CSV `error` field to a short status token."""
    if not err:
        return "OK"
    e = err.lower()
    if "out of memory" in e or "oom" in e:
        return "OOM"
    if "internal_error" in e or "cufft error code: 5" in e or "plan" in e:
        return "PLAN"
    if "invalid" in e:
        return "FAIL"
    return "FAIL"


def main():
    if len(sys.argv) < 3:
        sys.exit("usage: make_sheet.py <results.csv> <out_prefix> [env.txt]")
    csv_path, out_prefix = sys.argv[1], sys.argv[2]
    env_path = sys.argv[3] if len(sys.argv) > 3 else os.path.join(
        os.path.dirname(csv_path), "env.txt")

    gpu_stamp = ""
    if os.path.isfile(env_path):
        with open(env_path) as f:
            gpu_stamp = " | ".join(l.strip() for l in f if l.strip())

    # matrix -> {rows, cols, nnz, methods: {key: {...}}}
    mats = {}
    seen_matrices = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            name = r["matrix"]
            if name not in mats:
                mats[name] = {"rows": fnum(r["rows"]), "cols": fnum(r["cols"]),
                              "nnz": fnum(r["nnz"]), "m": {}}
                seen_matrices.append(name)
            key = METHOD_KEY.get(r["method"])
            if key is None:
                continue
            prep = fnum(r.get("preprocess_ms"))
            kern = fnum(r.get("median_ms"))
            status = classify(r.get("error", ""))
            total = (prep + kern) if (prep is not None and kern is not None) else None
            mats[name]["m"][key] = {"prep": prep, "kernel": kern,
                                    "total": total, "status": status,
                                    "mem": fnum(r.get("mem_mb"))}

    # A method with NO row for a present matrix = process died before it printed
    # (timeout or crash) -> DNF.
    for name in mats:
        for key in METHOD_ORDER:
            if key not in mats[name]["m"]:
                mats[name]["m"][key] = {"prep": None, "kernel": None,
                                        "total": None, "status": "DNF", "mem": None}

    order = sorted(seen_matrices,
                   key=lambda n: (mats[n]["rows"] or 0) or (mats[n]["cols"] or 0))

    # ---- CSV out ----
    cols = ["matrix", "rows", "cols", "nnz", "density"]
    for key in METHOD_ORDER:
        cols += [f"{key}_prep_ms", f"{key}_kernel_ms", f"{key}_total_ms",
                 f"{key}_mem_mb", f"{key}_status"]
    csv_out = out_prefix + ".csv"
    os.makedirs(os.path.dirname(csv_out) or ".", exist_ok=True)
    with open(csv_out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for name in order:
            d = mats[name]
            dens = (d["nnz"] / (d["rows"] * d["cols"])
                    if d["rows"] and d["cols"] and d["nnz"] else None)
            row = [name, fmt_int(d["rows"]), fmt_int(d["cols"]), fmt_int(d["nnz"]),
                   f"{dens:.3e}" if dens else ""]
            for key in METHOD_ORDER:
                m = d["m"][key]
                row += [fmt(m["prep"]), fmt(m["kernel"]), fmt(m["total"]),
                        fmt(m["mem"]), m["status"]]
            w.writerow(row)

    # ---- HTML out ----
    html_out = out_prefix + ".html"
    with open(html_out, "w") as f:
        f.write(render_html(order, mats, gpu_stamp))

    n_ok = {k: sum(1 for n in order if mats[n]["m"][k]["status"] == "OK")
            for k in METHOD_ORDER}
    print(f"matrices: {len(order)}")
    print("completed (OK): " + ", ".join(f"{METHOD_TITLE[k]}={n_ok[k]}"
                                         for k in METHOD_ORDER))
    print(f"CSV : {csv_out}")
    print(f"HTML: {html_out}")


def fmt(x):
    return f"{x:.3f}" if isinstance(x, float) else ""


def fmt_int(x):
    return str(int(x)) if isinstance(x, float) else ""


STATUS_COLOR = {"OK": "", "OOM": "#8a5a00", "PLAN": "#7a3b00",
                "FAIL": "#7a0000", "DNF": "#555"}


def render_html(order, mats, gpu_stamp):
    def td(x, cls=""):
        c = f' class="{cls}"' if cls else ""
        return f"<td{c}>{x}</td>"

    head_methods = "".join(
        f'<th colspan="4" class="grp">{html.escape(METHOD_TITLE[k])}</th>'
        for k in METHOD_ORDER)
    sub = "".join("<th>prep<br>ms</th><th>kernel<br>ms</th><th>total<br>ms</th><th>st</th>"
                  for _ in METHOD_ORDER)

    rows_html = []
    for name in order:
        d = mats[name]
        dens = (d["nnz"] / (d["rows"] * d["cols"])
                if d["rows"] and d["cols"] and d["nnz"] else None)
        cells = [td(html.escape(name), "mtx"),
                 td(fmt_int(d["rows"])), td(fmt_int(d["cols"])),
                 td(fmt_int(d["nnz"])),
                 td(f"{dens:.2e}" if dens else "")]
        for key in METHOD_ORDER:
            m = d["m"][key]
            st = m["status"]
            color = STATUS_COLOR.get(st, "")
            stcell = (f'<td class="st" style="color:#fff;background:{color}">{st}</td>'
                      if color else f'<td class="st ok">{st}</td>')
            cells += [td(fmt(m["prep"])), td(fmt(m["kernel"])),
                      td(fmt(m["total"]), "tot"), stcell]
        rows_html.append("<tr>" + "".join(cells) + "</tr>")

    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>Sampled-100 benchmark: dense vs CSC methods</title>
<style>
 body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:1.2rem;color:#111}}
 h1{{font-size:1.15rem;margin:0 0 .2rem}}
 .stamp{{color:#666;font-size:.8rem;margin-bottom:.8rem}}
 table{{border-collapse:collapse;font-size:.78rem;font-variant-numeric:tabular-nums}}
 th,td{{border:1px solid #ddd;padding:2px 6px;text-align:right}}
 th{{background:#f4f4f4}}
 td.mtx{{text-align:left;font-family:ui-monospace,monospace}}
 th.grp{{background:#e8eef6}}
 td.tot{{font-weight:600}}
 td.st{{text-align:center;font-weight:600}}
 td.st.ok{{color:#0a0}}
 tr:nth-child(even) td{{background:#fafafa}}
</style></head><body>
<h1>Sampled-100 benchmark — dense cuFFT vs CSC cuFFT-smooth vs CSC mixed-radix</h1>
<div class="stamp">{html.escape(gpu_stamp)}<br>
preprocessing = host CSC build + H2D + plan/precompute · kernel = GPU FFT (median of 10) ·
total = prep+kernel · st: OK / OOM / PLAN(cufft plan fail) / DNF(timeout|crash)</div>
<table><thead>
<tr><th rowspan="2">matrix</th><th rowspan="2">rows</th><th rowspan="2">cols</th>
<th rowspan="2">nnz</th><th rowspan="2">density</th>{head_methods}</tr>
<tr>{sub}</tr></thead><tbody>
{chr(10).join(rows_html)}
</tbody></table></body></html>"""


if __name__ == "__main__":
    main()
