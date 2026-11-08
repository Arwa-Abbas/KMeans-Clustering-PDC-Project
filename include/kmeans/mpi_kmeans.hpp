#pragma once

#include "kmeans/kmeans.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <mpi.h>
#include <string>

namespace kmeans {

// Builds the per-rank assignment step: SerialAssigner for MPI, OpenMPAssigner for hybrid.
using AssignerFactory = std::function<std::unique_ptr<Assigner>(
    const float* points, std::uint64_t rows, int dimensions, int clusters)>;

struct MpiRun {
    Result result;
    std::uint64_t rows = 0;  // rows clustered across all ranks
    int dimensions = 0;
    double loading_seconds = 0;
};

// Distributed K-Means. Each rank reads one contiguous block of the first `row_limit` rows
// (every row when 0) with MPI-IO and holds a copy of the centroids; each iteration ends
// with a single MPI_Allreduce. Returns false on every rank if any rank fails.
bool run_mpi(const std::string& binary_path, const std::string& metadata_path,
             std::uint64_t row_limit, const Config& config,
             const AssignerFactory& make_assigner, MPI_Comm comm, MpiRun& run);

}  // namespace kmeans
