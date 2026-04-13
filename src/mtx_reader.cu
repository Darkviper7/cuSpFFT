#include "mtx_reader.h"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>

COOMatrix read_mtx(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open())
        throw std::runtime_error("Cannot open file: " + path);

    std::string line;

    // Parse header line: %%MatrixMarket matrix coordinate <type> <symmetry>
    if (!std::getline(f, line))
        throw std::runtime_error("Empty file: " + path);
    if (line.rfind("%%MatrixMarket", 0) != 0)
        throw std::runtime_error("Not a Matrix Market file: " + path);

    bool symmetric = (line.find("symmetric") != std::string::npos ||
                      line.find("Symmetric") != std::string::npos);

    // Skip comment lines
    while (std::getline(f, line)) {
        if (!line.empty() && line[0] != '%') break;
    }

    // Parse dimensions and stored entry count
    int rows, cols, stored_nnz;
    {
        std::istringstream ss(line);
        if (!(ss >> rows >> cols >> stored_nnz))
            throw std::runtime_error("Bad dimension line in: " + path);
    }

    COOMatrix coo;
    coo.rows = rows;
    coo.cols = cols;
    coo.row_idx.reserve(stored_nnz * (symmetric ? 2 : 1));
    coo.col_idx.reserve(stored_nnz * (symmetric ? 2 : 1));

    for (int i = 0; i < stored_nnz; i++) {
        if (!std::getline(f, line))
            throw std::runtime_error("Unexpected EOF in: " + path);
        std::istringstream ss(line);
        int r, c;
        if (!(ss >> r >> c))
            throw std::runtime_error("Bad entry line in: " + path);
        r--; c--;  // convert 1-based to 0-based

        coo.row_idx.push_back(r);
        coo.col_idx.push_back(c);

        if (symmetric && r != c) {
            coo.row_idx.push_back(c);
            coo.col_idx.push_back(r);
        }
    }

    // Deduplicate (some MTX files list duplicates)
    int n = static_cast<int>(coo.row_idx.size());
    std::vector<std::pair<int,int>> pairs(n);
    for (int i = 0; i < n; i++)
        pairs[i] = {coo.row_idx[i], coo.col_idx[i]};
    std::sort(pairs.begin(), pairs.end());
    pairs.erase(std::unique(pairs.begin(), pairs.end()), pairs.end());

    coo.row_idx.resize(pairs.size());
    coo.col_idx.resize(pairs.size());
    for (int i = 0; i < (int)pairs.size(); i++) {
        coo.row_idx[i] = pairs[i].first;
        coo.col_idx[i] = pairs[i].second;
    }
    coo.nnz = static_cast<int>(coo.row_idx.size());

    return coo;
}

CSRMatrix coo_to_csr(const COOMatrix& coo) {
    CSRMatrix csr;
    csr.rows    = coo.rows;
    csr.cols    = coo.cols;
    csr.nnz     = coo.nnz;
    csr.col_idx = coo.col_idx;          // column indices are already in row order
    csr.row_ptr.assign(coo.rows + 1, 0);

    // Count nonzeros per row
    for (int i = 0; i < coo.nnz; i++)
        csr.row_ptr[coo.row_idx[i] + 1]++;

    // Prefix sum to get row pointers
    for (int r = 0; r < coo.rows; r++)
        csr.row_ptr[r + 1] += csr.row_ptr[r];

    return csr;
}
