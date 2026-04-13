#pragma once
#include <vector>
#include <string>

// COO representation of a binary sparse matrix.
// All stored values are implicitly 1 (binary); only (row, col) pairs are kept.
struct COOMatrix {
    int rows;       // number of rows
    int cols;       // number of columns
    int nnz;        // number of nonzeros
    std::vector<int> row_idx;
    std::vector<int> col_idx;
};

// Read a Matrix Market .mtx file and return a binary COO matrix.
// All nonzero values are binarized to 1.
// Throws std::runtime_error on parse failure.
COOMatrix read_mtx(const std::string& path);

// CSR representation of a binary sparse matrix.
// row_ptr[r]..row_ptr[r+1] gives the range of nonzeros in row r.
struct CSRMatrix {
    int rows, cols, nnz;
    std::vector<int> row_ptr; // size rows+1
    std::vector<int> col_idx; // size nnz
};

// Convert a COO matrix (must be row-sorted) to CSR.
// read_mtx() already returns a row-sorted COO, so no re-sort is needed.
CSRMatrix coo_to_csr(const COOMatrix& coo);
