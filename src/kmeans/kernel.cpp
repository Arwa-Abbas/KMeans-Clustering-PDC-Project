// Built without OpenMP: this file holds the only compiled copy of the assignment loop.
#include "kmeans/kernel.hpp"

#include <algorithm>

namespace kmeans {

void assign_block(const float* points, std::uint64_t rows, int dimensions,
                  const double* centroids, int clusters, int* assignments, double* totals) {
    const TotalsLayout layout{clusters, dimensions};
    double* sums = totals + layout.sums();
    double* counts = totals + layout.counts();
    std::uint64_t changed = 0;
    double sse = 0;
    for (std::uint64_t row = 0; row < rows; ++row) {
        const float* point = points + row * static_cast<std::uint64_t>(dimensions);
        int nearest = 0;
        double best = squared_distance(point, centroids, dimensions);
        for (int cluster = 1; cluster < clusters; ++cluster) {
            const double candidate = squared_distance(
                point, centroids + static_cast<std::size_t>(cluster) * dimensions, dimensions);
            if (candidate < best) {
                best = candidate;
                nearest = cluster;
            }
        }
        sse += best;
        if (assignments[row] != nearest) {
            assignments[row] = nearest;
            ++changed;
        }
        counts[nearest] += 1.0;
        double* sum = sums + static_cast<std::size_t>(nearest) * dimensions;
        for (int d = 0; d < dimensions; ++d) sum[d] += point[d];
    }
    totals[layout.changed()] += static_cast<double>(changed);
    totals[layout.sse()] += sse;
}

double block_sse(const float* points, std::uint64_t rows, int dimensions,
                 const double* centroids, int clusters) {
    double sse = 0;
    for (std::uint64_t row = 0; row < rows; ++row) {
        const float* point = points + row * static_cast<std::uint64_t>(dimensions);
        double best = squared_distance(point, centroids, dimensions);
        for (int cluster = 1; cluster < clusters; ++cluster) {
            best = std::min(best, squared_distance(
                point, centroids + static_cast<std::size_t>(cluster) * dimensions, dimensions));
        }
        sse += best;
    }
    return sse;
}

SerialAssigner::SerialAssigner(const float* points, std::uint64_t rows, int dimensions,
                               int clusters)
    : points_(points), rows_(rows), dimensions_(dimensions), clusters_(clusters),
      assignments_(rows, -1) {}

void SerialAssigner::assign(const double* centroids, double* totals) {
    const TotalsLayout layout{clusters_, dimensions_};
    std::fill(totals, totals + layout.size(), 0.0);
    assign_block(points_, rows_, dimensions_, centroids, clusters_, assignments_.data(), totals);
}

double SerialAssigner::sse(const double* centroids) const {
    return block_sse(points_, rows_, dimensions_, centroids, clusters_);
}

}  // namespace kmeans
