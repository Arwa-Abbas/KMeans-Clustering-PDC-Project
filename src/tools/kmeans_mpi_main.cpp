// Entry point for the MPI build, and for the hybrid MPI + OpenMP build when compiled with
// KMEANS_HYBRID_CLI.
#include "kmeans/cli.hpp"
#include "kmeans/kernel.hpp"
#include "kmeans/mpi_kmeans.hpp"
#ifdef KMEANS_HYBRID_CLI
#include "kmeans/openmp.hpp"
#endif

#include <cinttypes>
#include <cstdio>
#include <exception>
#include <iostream>
#include <memory>

namespace {
#ifdef KMEANS_HYBRID_CLI
constexpr bool kAcceptsThreads = true;
constexpr const char* kVariant = "hybrid";
#else
constexpr bool kAcceptsThreads = false;
constexpr const char* kVariant = "mpi";
#endif

int run_program(int argc, char** argv, int rank, int processes) {
    using namespace kmeans;
    RunOptions options;
    try {
        options = parse_run_options(argc, argv, kAcceptsThreads);
    } catch (const UsageError& error) {
        if (rank == 0) std::cerr << "error: " << error.what() << '\n' << run_usage(argv[0], kAcceptsThreads);
        return 2;
    }

#ifdef KMEANS_HYBRID_CLI
    const int threads = options.threads;
    const AssignerFactory make_assigner = [threads](const float* points, std::uint64_t rows,
                                                    int dimensions, int clusters) {
        return std::unique_ptr<Assigner>(
            std::make_unique<OpenMPAssigner>(points, rows, dimensions, clusters, threads));
    };
#else
    const AssignerFactory make_assigner = [](const float* points, std::uint64_t rows,
                                             int dimensions, int clusters) {
        return std::unique_ptr<Assigner>(
            std::make_unique<SerialAssigner>(points, rows, dimensions, clusters));
    };
#endif

    MpiRun run;
    if (!run_mpi(options.data, options.metadata, options.rows, options.config, make_assigner,
                 MPI_COMM_WORLD, run)) {
        if (rank == 0) std::cerr << "error: MPI K-Means failed\n";
        return 1;
    }
    if (rank != 0) return 0;

    const Result& result = run.result;
    std::printf("rows=%" PRIu64 " processes=%d threads_per_process=%d clusters=%d iterations=%d "
                "final_sse=%.12g loading_seconds=%.3f initialization_seconds=%.3f "
                "computation_seconds=%.3f\n",
                run.rows, processes, options.threads, options.config.clusters, result.iterations,
                result.sse, run.loading_seconds, result.initialization_seconds,
                result.computation_seconds);
    if (!options.output.empty()) {
        try {
            write_result(options.output, run.rows, run.dimensions, options.config, result,
                         run.loading_seconds, kVariant, processes, options.threads);
        } catch (const std::exception& error) {
            std::cerr << "error: " << error.what() << '\n';
            return 1;
        }
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0;
    int processes = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &processes);
    if (provided < MPI_THREAD_FUNNELED) MPI_Abort(MPI_COMM_WORLD, 2);
    const int status = run_program(argc, argv, rank, processes);
    MPI_Finalize();
    return status;
}
