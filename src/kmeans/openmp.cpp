#include "kmeans/openmp.hpp"

#include "kmeans/kernel.hpp"

#include <algorithm>
#include <omp.h>
#include <stdexcept>

namespace kmeans {
namespace {

constexpr std::size_t kCacheLineDoubles = 64 / sizeof(double);

// Round up to whole cache lines and add one spare line, so neighbouring threads' buffers
// never share a line however the vector happens to be aligned.
std::size_t padded(std::size_t count) {
    return (count + kCacheLineDoubles - 1) / kCacheLineDoubles * kCacheLineDoubles +
           kCacheLineDoubles;
}

int checked_threads(int threads) {
    if (threads < 1) throw std::invalid_argument("the thread count must be positive");
    return threads;
}

}  // namespace

OpenMPAssigner::OpenMPAssigner(const float* points, std::uint64_t rows, int dimensions,
                               int clusters, int threads)
    : points_(points), rows_(rows), dimensions_(dimensions), clusters_(clusters),
      threads_(checked_threads(threads)),
      stride_(padded(TotalsLayout{clusters, dimensions}.size())),
      partials_(stride_ * static_cast<std::size_t>(threads_)), assignments_(rows, -1) {}

void OpenMPAssigner::assign(const double* centroids, double* totals) {
    const TotalsLayout layout{clusters_, dimensions_};
    int team = 1;
#pragma omp parallel num_threads(threads_)
    {
        const int thread = omp_get_thread_num();
        if (thread == 0) team = omp_get_num_threads();
        const RowRange range = split_rows(rows_, omp_get_num_threads(), thread);
        double* partial = partials_.data() + static_cast<std::size_t>(thread) * stride_;
        std::fill(partial, partial + layout.size(), 0.0);
        assign_block(points_ + range.begin * static_cast<std::uint64_t>(dimensions_), range.count,
                     dimensions_, centroids, clusters_, assignments_.data() + range.begin, partial);
    }
    std::fill(totals, totals + layout.size(), 0.0);
    for (int thread = 0; thread < team; ++thread) {
        const double* partial = partials_.data() + static_cast<std::size_t>(thread) * stride_;
        for (std::size_t i = 0; i < layout.size(); ++i) totals[i] += partial[i];
    }
}

double OpenMPAssigner::sse(const double* centroids) const {
    std::vector<double> partial(static_cast<std::size_t>(threads_), 0.0);
#pragma omp parallel num_threads(threads_)
    {
        const int thread = omp_get_thread_num();
        const RowRange range = split_rows(rows_, omp_get_num_threads(), thread);
        partial[thread] = block_sse(points_ + range.begin * static_cast<std::uint64_t>(dimensions_),
                                    range.count, dimensions_, centroids, clusters_);
    }
    double total = 0;
    for (const double value : partial) total += value;
    return total;
}

Result run_openmp(const Matrix& matrix, const Config& config, int threads) {
    OpenMPAssigner assigner(matrix.values.data(), matrix.rows, matrix.dimensions,
                            config.clusters, threads);
    return run_kmeans(matrix, config, assigner);
}

}  // namespace kmeans
