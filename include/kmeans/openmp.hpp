#pragma once

#include "kmeans/kmeans.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace kmeans {

// OpenMP assignment step: each thread runs the shared kernel on a contiguous block of rows
// into its own cache-line-padded totals buffer; buffers are added in thread order after
// the parallel region, so no thread writes shared totals and MPI is never called inside it.
class OpenMPAssigner final : public Assigner {
public:
    OpenMPAssigner(const float* points, std::uint64_t rows, int dimensions, int clusters,
                   int threads);
    void assign(const double* centroids, double* totals) override;
    double sse(const double* centroids) const override;

private:
    const float* points_;
    std::uint64_t rows_;
    int dimensions_;
    int clusters_;
    int threads_;
    std::size_t stride_;
    std::vector<double> partials_;
    std::vector<int> assignments_;
};

Result run_openmp(const Matrix& matrix, const Config& config, int threads);

}  // namespace kmeans
