# Convert one cuSpFFT stdout log into CSV rows for plotting / reporting.
#
# Invoked from scripts/run_sweep.sh as:
#   awk -v MATRIX="$name" -v TS="$STAMP" -f scripts/parse_log.awk <log>
#
# Output schema (one row per (matrix, method) seen in the log):
#   timestamp,matrix,rows,cols,nnz,method,median_ms,min_ms,max_ms,n_iters,mem_mb,preprocess_ms,max_abs,max_rel,rms_abs,error
#
# Four regex blocks match the four stable printf formats in src/main.cu:
#   1. Repeat-mode summary (main.cu):
#      "%-26s  %10.3f  %12.2f  [min %.3f, max %.3f]  n=%d  prep %.3f\n"
#   2. Single-run summary (main.cu and the CPU baseline):
#      "%-26s  %10.3f  %12.2f  prep %.3f\n"   (trailing trivia after %.2f is tolerated)
#   preprocess_ms (host setup: build+H2D+plan) is captured from the " prep X"
#   token when present (0 for methods not yet instrumented; empty for CPU ref).
#   3. Correctness check (main.cu:164,213):
#      "  check %-19s  max_abs %.3e  max_rel %.3e  rms_abs %.3e\n"
#   4. Failure / OOM (catch blocks, e.g. main.cu:365-366):
#      "%-26s  %10s  %12s  [%s]\n"   (literal "OOM" tokens, real exception in [])
#
# Timing and check labels do not match verbatim — main.cu prints
# "Dense cuFFT (baseline)" / "SpFFT (baseline)" / "Sparse CSC (mixed-radix)"
# in print_bench, but "Dense cuFFT" / "SpFFT" / "Sparse CSC mixed-radix" in
# print_check. Both forms are normalized via canon() to a common key so
# correctness rows merge onto the right timing row.

BEGIN {
    OFS = ","
    rows = ""
    cols = ""
    nnz = ""
    order_n = 0
}

# Strip a method label (right-trim only; %-26s and %-19s are space-padded right).
function trim_right(s,    out) {
    out = s
    sub(/[ \t]+$/, "", out)
    return out
}

# RFC-4180-ish CSV quoting: wrap in double quotes if the field contains
# commas, quotes, or newlines, doubling any internal quote.
function csv_escape(s) {
    if (s ~ /[",\n]/) {
        gsub(/"/, "\"\"", s)
        return "\"" s "\""
    }
    return s
}

# Map a method label (timing or check form) to a canonical key.
#   "Dense cuFFT (baseline)"   -> "Dense cuFFT"
#   "Dense cuFFT"              -> "Dense cuFFT"
#   "SpFFT (baseline)"         -> "SpFFT"
#   "SpFFT"                    -> "SpFFT"
#   "Sparse CSC (mixed-radix)" -> "Sparse CSC mixed-radix"
#   "Sparse CSC mixed-radix"   -> "Sparse CSC mixed-radix"
function canon(s,    out) {
    out = s
    gsub(/ \(baseline\)/, "", out)
    gsub(/[()]/, "", out)
    gsub(/[ \t]+/, " ", out)
    sub(/^ /, "", out)
    sub(/ $/, "", out)
    return out
}

# Record a method by canonical key while preserving the original (timing-form)
# label for display. Subsequent calls for the same key are ignored, so the
# first-seen label wins — the timing label (with its parens) is the form that
# reaches the CSV.
function record(method,    key) {
    key = canon(method)
    if (!(key in seen)) {
        seen[key] = 1
        order[++order_n] = method
    }
}

# --- Header metadata ----------------------------------------------------------

/^Size:/ { rows = $2; cols = $4 }
/^nnz:/  { nnz  = $2 }

# --- Failure (catch block: literal "OOM" tokens, real cause in brackets) -----
# "Variant Label              OOM           OOM  [CUDA error: out of memory]"

match($0, /^(.{26})  +OOM +OOM +\[(.*)\]$/, m) {
    method = trim_right(m[1])
    key = canon(method)
    timing[key] = "" OFS "" OFS "" OFS "" OFS ""
    errors[key] = m[2]
    record(method)
    next
}

# --- Repeat-mode summary ------------------------------------------------------
# "Variant Label             123.456     789.01  [min 120.000, max 130.000]  n=10"

match($0, /^(.{26})  +([0-9.]+) +([0-9.]+) +\[min ([0-9.]+), max ([0-9.]+)\] +n=([0-9]+)( +prep ([0-9.]+))?/, m) {
    method = trim_right(m[1])
    key = canon(method)
    timing[key] = m[2] OFS m[4] OFS m[5] OFS m[6] OFS m[3]
    preproc[key] = m[8]
    record(method)
    next
}

# --- Single-run summary -------------------------------------------------------
# "Variant Label             123.456     789.01"   (and tolerate trailing
# annotations like "  [CPU, FFTW3 single-thread]")

match($0, /^(.{26})  +([0-9.]+) +([0-9.]+)(  +prep ([0-9.]+))?([[:space:]].*)?$/, m) {
    method = trim_right(m[1])
    key = canon(method)
    if (!(key in timing)) {
        timing[key] = m[2] OFS "" OFS "" OFS "1" OFS m[3]
        preproc[key] = m[5]
        record(method)
    }
    next
}

# --- Correctness check --------------------------------------------------------
# "  check method label    max_abs 1.234e-05  max_rel 5.678e-04  rms_abs 9.876e-06"
# Method label width is variable (longer labels exceed the %-19s minimum and
# print as-is, with two literal spaces before "max_abs").

match($0, /^  check (.+)  +max_abs +([0-9.eE+-]+) +max_rel +([0-9.eE+-]+) +rms_abs +([0-9.eE+-]+)/, m) {
    key = canon(trim_right(m[1]))
    correct[key] = m[2] OFS m[3] OFS m[4]
    next
}

END {
    for (i = 1; i <= order_n; i++) {
        method = order[i]
        key = canon(method)
        c = (key in correct) ? correct[key] : ",,"
        e = (key in errors) ? csv_escape(errors[key]) : ""
        p = (key in preproc) ? preproc[key] : ""
        print TS, MATRIX, rows, cols, nnz, method, timing[key], p, c, e
    }
}
