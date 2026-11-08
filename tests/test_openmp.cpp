#include "kmeans/kmeans.hpp"
#include "kmeans/openmp.hpp"
#include "synthetic.hpp"

#include <iostream>

int main() {
    using namespace kmeans;
    // 10,007 rows: not divisible by any tested thread count, so blocks are uneven.
    const Matrix data = test::make_blobs(10007, 7, 11);
    Config config;
    config.clusters = 7;
    config.sample_size = 3000;
    config.max_iterations = 50;
    config.print_progress = false;

    const Result reference = run_sequential(data, config);
    bool good = reference.iterations > 1;
    for (int threads = 1; threads <= 5; ++threads) {
        const Result parallel = run_openmp(data, config, threads);
        good = good && test::same_clustering(parallel, reference);
        // One thread runs the identical kernel over the identical rows: bit-for-bit equal.
        if (threads == 1) good = good && parallel.centroids == reference.centroids && parallel.sse == reference.sse;
    }
    if (!good) {
        std::cerr << "OpenMP comparison failed\n";
        return 1;
    }
    std::cout << "OpenMP comparison passed\n";
    return 0;
}
