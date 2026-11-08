#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace kmeans {

// Row-major float32 points, as stored in the preprocessed binary.
struct Matrix {
    std::vector<float> values;
    std::uint64_t rows = 0;
    int dimensions = 0;
};

struct MatrixShape {
    std::uint64_t rows = 0;
    int dimensions = 0;
};

struct Config {
    int clusters = 16;
    int max_iterations = 100;
    double tolerance = 0.001;            // stop when fewer than this fraction of points move
    std::uint64_t seed = 42;
    std::uint64_t sample_size = 100000;  // K-Means++ uses the first min(sample_size, rows) points
    bool print_progress = true;          // one line per iteration
};

struct Result {
    std::vector<double> initial_centroids;
    std::vector<double> centroids;
    std::vector<std::uint64_t> counts;
    int iterations = 0;
    std::uint64_t changed = 0;
    double sse = 0;
    double initialization_seconds = 0;
    double computation_seconds = 0;  // Lloyd iterations only
};

class Stopwatch {
public:
    double seconds() const {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - start_).count();
    }

private:
    std::chrono::steady_clock::time_point start_ = std::chrono::steady_clock::now();
};

// The assignment step is the only part of Lloyd's loop that differs between variants.
class Assigner {
public:
    virtual ~Assigner() = default;
    // Overwrites `totals` (layout in kernel.hpp) with this assigner's cluster sums, counts,
    // changed-assignment count, and SSE.
    virtual void assign(const double* centroids, double* totals) = 0;
    virtual double sse(const double* centroids) const = 0;
};

// Adds totals across MPI processes in place; empty for single-process runs.
using Reducer = std::function<void(double* values, std::size_t count)>;

MatrixShape read_metadata(const std::string& metadata_path);
// Loads the first `row_limit` rows of the binary matrix, or every row when it is 0.
Matrix load_matrix(const std::string& binary_path, const std::string& metadata_path,
                   std::uint64_t row_limit = 0);

void validate_config(const Config& config, std::uint64_t rows);
std::vector<double> initialize_centroids(const Matrix& matrix, const Config& config);

// Lloyd iterations shared by all four variants; `result.centroids` holds the start.
void run_lloyd(Assigner& assigner, const Config& config, int dimensions,
               std::uint64_t global_rows, const Reducer& reduce, bool print_progress,
               Result& result);
// K-Means++ initialization on `matrix`, then Lloyd iterations with `assigner`.
Result run_kmeans(const Matrix& matrix, const Config& config, Assigner& assigner);
Result run_sequential(const Matrix& matrix, const Config& config);

void write_result(const std::string& path, std::uint64_t rows, int dimensions,
                  const Config& config, const Result& result, double loading_seconds,
                  const std::string& variant, int processes, int threads);

}  // namespace kmeans
