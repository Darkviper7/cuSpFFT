#!/usr/bin/env bash
# Download benchmark matrices from SuiteSparse into data/
set -e

DATA_DIR="$(dirname "$0")/../data"
mkdir -p "$DATA_DIR"

download_mtx() {
    local group="$1"
    local name="$2"
    local url="https://suitesparse-collection-website.herokuapp.com/MM/${group}/${name}.tar.gz"
    local tarball="${DATA_DIR}/${name}.tar.gz"

    if [ -f "${DATA_DIR}/${name}/${name}.mtx" ]; then
        echo "[skip] ${name}.mtx already exists"
        return
    fi

    echo "[download] ${group}/${name} ..."
    curl -L -o "$tarball" "$url"
    tar -xzf "$tarball" -C "$DATA_DIR"
    rm "$tarball"
    echo "[done] ${DATA_DIR}/${name}/${name}.mtx"
}

# Dev/debug (~1-5k nodes)
download_mtx HB sstmodel

# Benchmark vs dense cuFFT (~8-10k nodes)
download_mtx PARSEC benzene

# Large-scale OOM test (>50k nodes)
download_mtx Boeing pct20stif
