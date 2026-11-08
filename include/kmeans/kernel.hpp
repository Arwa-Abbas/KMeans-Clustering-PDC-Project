#pragma once

#include "kmeans/kmeans.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace kmeans {

// Per-iteration totals, laid out so one MPI_Allreduce of doubles combines everything:
// [clusters*dimensions sums | clusters counts | changed | sse].
// Counts stay exact as doubles up to 2^53, far above HIGGS's 11 million rows.
struct TotalsLayout {
    int clusters;
    int dimensions;

    std::size_t sums() const { return 0; }
    std::size_t counts() const { return static_cast<std::size_t>(clusters) * dimensions; }
    std::size_t changed() const { return counts() + clusters; }
    std::size_t sse() const { return changed() + 1; }
    std::size_t size() const { return sse() + 1; }
};

struct RowRange {
    std::uint64_t begin;
    std::uint64_t count;
};

// Block `part` of `parts` contiguous blocks; the first rows % parts blocks get one extra
// row. MPI ranks and OpenMP threads split rows the same way.
inline RowRange split_rows(std::uint64_t rows, int parts, int part) {
    const auto n = static_cast<std::uint64_t>(parts);
    const auto i = static_cast<std::uint64_t>(part);
    const std::uint64_t base = rows / n;
    const std::uint64_t extra = rows % n;
    return {i * base + std::min(i, extra), base + (i < extra ? 1 : 0)};
}

inline double squared_distance(const float* point, const double* centroid, int dimensions) {
    double total = 0;
    for (int d = 0; d < dimensions; ++d) {
        const double delta = static_cast<double>(point[d]) - centroid[d];
        total += delta * delta;
    }
    return total;
}

// The assignment kernel. It is compiled once, in kernel.cpp without OpenMP, and that one
// copy runs in the sequential program, in every OpenMP thread, and in every MPI rank, so
// one-worker timings of all four variants are directly comparable. Adds into `totals`.
void assign_block(const float* points, std::uint64_t rows, int dimensions,
                  const double* centroids, int clusters, int* assignments, double* totals);
double block_sse(const float* points, std::uint64_t rows, int dimensions,
                 const double* centroids, int clusters);

// Single-threaded assignment step used by the sequential and MPI-only builds.
class SerialAssigner final : public Assigner {
public:
    SerialAssigner(const float* points, std::uint64_t rows, int dimensions, int clusters);
    void assign(const double* centroids, double* totals) override;
    double sse(const double* centroids) const override;

private:
    const float* points_;
    std::uint64_t rows_;
    int dimensions_;
    int clusters_;
    std::vector<int> assignments_;
};

}  // namespace kmeans
