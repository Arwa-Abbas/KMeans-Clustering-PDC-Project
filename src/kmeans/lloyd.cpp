#include "kmeans/kernel.hpp"

#include <cinttypes>
#include <cstdio>

namespace kmeans {

void run_lloyd(Assigner& assigner, const Config& config, int dimensions,
               std::uint64_t global_rows, const Reducer& reduce, bool print_progress,
               Result& result) {
    const int clusters = config.clusters;
    const TotalsLayout layout{clusters, dimensions};
    std::vector<double> totals(layout.size());
    result.counts.assign(static_cast<std::size_t>(clusters), 0);

    const Stopwatch clock;
    for (int iteration = 0; iteration < config.max_iterations; ++iteration) {
        assigner.assign(result.centroids.data(), totals.data());
        if (reduce) reduce(totals.data(), totals.size());  // the one MPI_Allreduce per iteration

        int empty = 0;
        for (int c = 0; c < clusters; ++c) {
            const double count = totals[layout.counts() + c];
            result.counts[c] = static_cast<std::uint64_t>(count);
            if (count == 0) {  // an empty cluster keeps its previous centroid
                ++empty;
                continue;
            }
            const double* sum = totals.data() + layout.sums() + static_cast<std::size_t>(c) * dimensions;
            double* centroid = result.centroids.data() + static_cast<std::size_t>(c) * dimensions;
            for (int d = 0; d < dimensions; ++d) centroid[d] = sum[d] / count;
        }
        result.iterations = iteration + 1;
        result.changed = static_cast<std::uint64_t>(totals[layout.changed()]);
        result.sse = totals[layout.sse()];
        const double fraction =
            static_cast<double>(result.changed) / static_cast<double>(global_rows);
        if (print_progress) {
            std::printf("iteration=%d changed=%" PRIu64 " fraction=%.8f assignment_sse=%.9g empty=%d\n",
                        result.iterations, result.changed, fraction, result.sse, empty);
        }
        if (fraction < config.tolerance) break;
    }
    result.computation_seconds = clock.seconds();

    // The SSE above used the centroids from before the last update; recompute it.
    double sse = assigner.sse(result.centroids.data());
    if (reduce) reduce(&sse, 1);
    result.sse = sse;
}

Result run_kmeans(const Matrix& matrix, const Config& config, Assigner& assigner) {
    validate_config(config, matrix.rows);
    Result result;
    const Stopwatch clock;
    result.centroids = initialize_centroids(matrix, config);
    result.initialization_seconds = clock.seconds();
    result.initial_centroids = result.centroids;
    run_lloyd(assigner, config, matrix.dimensions, matrix.rows, {}, config.print_progress, result);
    return result;
}

Result run_sequential(const Matrix& matrix, const Config& config) {
    SerialAssigner assigner(matrix.values.data(), matrix.rows, matrix.dimensions, config.clusters);
    return run_kmeans(matrix, config, assigner);
}

}  // namespace kmeans
