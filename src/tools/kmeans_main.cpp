// Entry point for the sequential build, and for the OpenMP build when compiled with
// KMEANS_OPENMP_CLI.
#include "kmeans/cli.hpp"
#ifdef KMEANS_OPENMP_CLI
#include "kmeans/openmp.hpp"
#endif

#include <cinttypes>
#include <cstdio>
#include <exception>
#include <iostream>

namespace {
#ifdef KMEANS_OPENMP_CLI
constexpr bool kAcceptsThreads = true;
constexpr const char* kVariant = "openmp";
#else
constexpr bool kAcceptsThreads = false;
constexpr const char* kVariant = "sequential";
#endif
}  // namespace

int main(int argc, char** argv) {
    using namespace kmeans;
    RunOptions options;
    try {
        options = parse_run_options(argc, argv, kAcceptsThreads);
    } catch (const UsageError& error) {
        std::cerr << "error: " << error.what() << '\n' << run_usage(argv[0], kAcceptsThreads);
        return 2;
    }
    try {
        const Stopwatch load_clock;
        const Matrix matrix = load_matrix(options.data, options.metadata, options.rows);
        const double loading_seconds = load_clock.seconds();
#ifdef KMEANS_OPENMP_CLI
        const Result result = run_openmp(matrix, options.config, options.threads);
#else
        const Result result = run_sequential(matrix, options.config);
#endif
        std::printf("rows=%" PRIu64 " clusters=%d iterations=%d final_sse=%.12g "
                    "loading_seconds=%.3f initialization_seconds=%.3f computation_seconds=%.3f\n",
                    matrix.rows, options.config.clusters, result.iterations, result.sse,
                    loading_seconds, result.initialization_seconds, result.computation_seconds);
        if (!options.output.empty()) {
            write_result(options.output, matrix.rows, matrix.dimensions, options.config, result,
                         loading_seconds, kVariant, 1, options.threads);
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
