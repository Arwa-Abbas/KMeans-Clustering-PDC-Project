#include "kmeans/mpi_kmeans.hpp"

#include "kmeans/kernel.hpp"

#include <algorithm>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace kmeans {
namespace {

bool all_ranks(bool ok, MPI_Comm comm) {
    int local = ok ? 1 : 0;
    int global = 0;
    MPI_Allreduce(&local, &global, 1, MPI_INT, MPI_MIN, comm);
    return global == 1;
}

// Every rank makes the same number of collective reads, even with uneven blocks.
bool collective_read(MPI_File file, std::uint64_t first_row, std::uint64_t rows, int dims,
                     float* points, MPI_Comm comm) {
    const std::uint64_t values = rows * static_cast<std::uint64_t>(dims);
    std::uint64_t most = 0;
    MPI_Allreduce(&values, &most, 1, MPI_UINT64_T, MPI_MAX, comm);
    constexpr std::uint64_t chunk = 1U << 20;
    bool ok = true;
    for (std::uint64_t done = 0; done < most; done += chunk) {
        const std::uint64_t position = std::min(done, values);
        const int count = static_cast<int>(std::min(values - position, chunk));
        const auto offset = static_cast<MPI_Offset>(
            (first_row * static_cast<std::uint64_t>(dims) + position) * sizeof(float));
        if (MPI_File_read_at_all(file, offset, points + position, count, MPI_FLOAT,
                                 MPI_STATUS_IGNORE) != MPI_SUCCESS) {
            ok = false;  // keep looping so the collective calls stay matched
        }
    }
    return ok;
}

}  // namespace

bool run_mpi(const std::string& binary_path, const std::string& metadata_path,
             std::uint64_t row_limit, const Config& config,
             const AssignerFactory& make_assigner, MPI_Comm comm, MpiRun& run) {
    int rank = 0;
    int processes = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &processes);
    const auto report = [rank](const std::exception& error) {
        std::cerr << "error (rank " << rank << "): " << error.what() << '\n';
    };

    MatrixShape file;
    bool ok = true;
    try {
        file = read_metadata(metadata_path);
        if (row_limit > file.rows) {
            throw std::invalid_argument("--rows " + std::to_string(row_limit) + " exceeds the " +
                                        std::to_string(file.rows) + " rows in " + binary_path);
        }
        validate_config(config, row_limit != 0 ? row_limit : file.rows);
    } catch (const std::exception& error) {
        if (rank == 0) report(error);
        ok = false;
    }
    if (!all_ranks(ok, comm)) return false;

    run = MpiRun{};
    run.rows = row_limit != 0 ? row_limit : file.rows;
    run.dimensions = file.dimensions;
    const int dims = file.dimensions;
    const RowRange block = split_rows(run.rows, processes, rank);

    std::vector<float> points;
    std::unique_ptr<Assigner> assigner;
    try {
        points.resize(std::max<std::uint64_t>(block.count, 1) * static_cast<std::uint64_t>(dims));
        assigner = make_assigner(points.data(), block.count, dims, config.clusters);
    } catch (const std::exception& error) {
        report(error);
        ok = false;
    }
    // If any rank cannot allocate its block, every rank stops together.
    if (!all_ranks(ok, comm)) return false;

    MPI_File handle;
    if (MPI_File_open(comm, binary_path.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &handle) !=
        MPI_SUCCESS) {
        if (rank == 0) std::cerr << "error: cannot open " << binary_path << '\n';
        return false;
    }
    MPI_Offset bytes = 0;
    MPI_File_get_size(handle, &bytes);
    ok = static_cast<std::uint64_t>(bytes) ==
         file.rows * static_cast<std::uint64_t>(dims) * sizeof(float);
    if (!all_ranks(ok, comm)) {
        MPI_File_close(&handle);
        if (rank == 0) std::cerr << "error: " << binary_path << " does not match its metadata\n";
        return false;
    }
    const Stopwatch load_clock;
    ok = collective_read(handle, block.begin, block.count, dims, points.data(), comm);
    MPI_File_close(&handle);
    if (!all_ranks(ok, comm)) return false;
    run.loading_seconds = load_clock.seconds();

    // Rank 0 runs K-Means++ on the same first-rows sample as the sequential program.
    Result& result = run.result;
    result.centroids.assign(static_cast<std::size_t>(config.clusters) * dims, 0.0);
    const Stopwatch init_clock;
    int init_ok = 1;
    if (rank == 0) {
        try {
            const Matrix sample =
                load_matrix(binary_path, metadata_path, std::min(config.sample_size, run.rows));
            result.centroids = initialize_centroids(sample, config);
        } catch (const std::exception& error) {
            report(error);
            init_ok = 0;
        }
    }
    MPI_Bcast(&init_ok, 1, MPI_INT, 0, comm);
    if (!init_ok) return false;
    MPI_Bcast(result.centroids.data(), static_cast<int>(result.centroids.size()), MPI_DOUBLE, 0,
              comm);
    result.initial_centroids = result.centroids;
    result.initialization_seconds = init_clock.seconds();

    // Only the main thread calls MPI, after the OpenMP region has finished.
    const Reducer reduce = [comm](double* values, std::size_t count) {
        MPI_Allreduce(MPI_IN_PLACE, values, static_cast<int>(count), MPI_DOUBLE, MPI_SUM, comm);
    };
    run_lloyd(*assigner, config, dims, run.rows, reduce, config.print_progress && rank == 0,
              result);
    // The slowest rank determines the parallel time.
    MPI_Allreduce(MPI_IN_PLACE, &result.computation_seconds, 1, MPI_DOUBLE, MPI_MAX, comm);
    return true;
}

}  // namespace kmeans
