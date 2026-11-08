#include "kmeans/kmeans.hpp"
#include "synthetic.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {

using namespace kmeans;

bool hand_checked_clusters() {
    // Two 28-dimensional pairs: {0, 2} and {10, 12}; each point is 28 units^2 from its
    // centroid, so the final SSE is 4 * 28 = 112.
    Matrix matrix;
    matrix.rows = 4;
    matrix.dimensions = 28;
    for (const float value : {0.0F, 2.0F, 10.0F, 12.0F}) matrix.values.insert(matrix.values.end(), 28, value);
    Config config;
    config.clusters = 2;
    config.max_iterations = 20;
    config.sample_size = 4;
    config.print_progress = false;
    const Result a = run_sequential(matrix, config);
    const Result b = run_sequential(matrix, config);
    return std::fabs(a.sse - 112.0) < 1e-8 && a.iterations == b.iterations && a.sse == b.sse &&
           a.centroids == b.centroids;
}

bool row_prefix_loading() {
    const Matrix data = test::make_blobs(50, 3, 5);
    test::write_matrix("build/test_rows.bin", "build/test_rows.metadata.json", data);
    const Matrix prefix = load_matrix("build/test_rows.bin", "build/test_rows.metadata.json", 7);
    if (prefix.rows != 7 || prefix.values != test::first_rows(data, 7).values) return false;
    const Matrix all = load_matrix("build/test_rows.bin", "build/test_rows.metadata.json");
    if (all.rows != 50 || all.values != data.values) return false;
    try {
        load_matrix("build/test_rows.bin", "build/test_rows.metadata.json", 51);
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}

}  // namespace

int main() {
    if (!hand_checked_clusters() || !row_prefix_loading()) {
        std::cerr << "sequential test failed\n";
        return 1;
    }
    std::cout << "sequential test passed\n";
    return 0;
}
