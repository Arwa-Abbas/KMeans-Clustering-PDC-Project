#pragma once

#include "kmeans/kmeans.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <random>
#include <string>
#include <vector>

namespace kmeans::test {

// Deterministic, well-separated 28-feature blobs for correctness tests.
inline Matrix make_blobs(std::uint64_t rows, int centers, std::uint64_t seed) {
    Matrix matrix;
    matrix.rows = rows;
    matrix.dimensions = 28;
    matrix.values.resize(rows * 28);
    std::mt19937_64 engine(seed);
    std::uniform_real_distribution<double> noise(-0.5, 0.5);
    for (std::uint64_t row = 0; row < rows; ++row) {
        const auto group = static_cast<int>(row % static_cast<std::uint64_t>(centers));
        for (int d = 0; d < 28; ++d) {
            const double center = static_cast<double>((group * 7 + d * 3) % 11) - 5.0;
            matrix.values[row * 28 + d] =
                static_cast<float>(center + noise(engine) + noise(engine));
        }
    }
    return matrix;
}

inline Matrix first_rows(const Matrix& matrix, std::uint64_t rows) {
    Matrix prefix;
    prefix.rows = rows;
    prefix.dimensions = matrix.dimensions;
    prefix.values.assign(matrix.values.begin(),
                         matrix.values.begin() + static_cast<std::ptrdiff_t>(rows * 28));
    return prefix;
}

// Writes the binary and the metadata fields that load_matrix and run_mpi check.
inline void write_matrix(const std::string& binary, const std::string& metadata,
                         const Matrix& matrix) {
    std::ofstream(binary, std::ios::binary)
        .write(reinterpret_cast<const char*>(matrix.values.data()),
               static_cast<std::streamsize>(matrix.values.size() * sizeof(float)));
    std::ofstream(metadata) << "{\n  \"rows\": " << matrix.rows << ",\n  \"features\": "
                            << matrix.dimensions << ",\n  \"data_type\": \"float32\",\n"
                            << "  \"byte_order\": \"little-endian\"\n}\n";
}

inline double max_difference(const std::vector<double>& a, const std::vector<double>& b) {
    double worst = a.size() == b.size() ? 0.0 : INFINITY;
    for (std::size_t i = 0; i < std::min(a.size(), b.size()); ++i)
        worst = std::max(worst, std::fabs(a[i] - b[i]));
    return worst;
}

// Same clustering up to floating-point summation order.
inline bool same_clustering(const Result& a, const Result& b) {
    return a.iterations == b.iterations && a.changed == b.changed && a.counts == b.counts &&
           a.initial_centroids == b.initial_centroids &&
           max_difference(a.centroids, b.centroids) <= 1e-9 &&
           std::fabs(a.sse - b.sse) <= 1e-9 * std::max(1.0, std::fabs(b.sse));
}

}  // namespace kmeans::test
