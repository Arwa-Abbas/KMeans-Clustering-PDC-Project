#include "kmeans/kmeans.hpp"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <stdexcept>

namespace kmeans {
namespace {

void write_centroids(std::ostream& out, const char* name, const std::vector<double>& values,
                     int clusters, int dimensions) {
    out << "  \"" << name << "\": [\n";
    for (int c = 0; c < clusters; ++c) {
        out << "    [";
        for (int d = 0; d < dimensions; ++d)
            out << (d ? ", " : "") << values[static_cast<std::size_t>(c) * dimensions + d];
        out << (c + 1 == clusters ? "]\n" : "],\n");
    }
    out << "  ]";
}

}  // namespace

void write_result(const std::string& path, std::uint64_t rows, int dimensions,
                  const Config& config, const Result& result, double loading_seconds,
                  const std::string& variant, int processes, int threads) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("cannot write " + path);
    // %.17g round-trips doubles; timings keep 9 significant digits.
    out << std::setprecision(17)
        << "{\n  \"variant\": \"" << variant << "\",\n"
        << "  \"processes\": " << processes << ",\n"
        << "  \"threads\": " << threads << ",\n"
        << "  \"rows\": " << rows << ",\n"
        << "  \"features\": " << dimensions << ",\n"
        << "  \"clusters\": " << config.clusters << ",\n"
        << "  \"seed\": " << config.seed << ",\n"
        << "  \"init_sample_size\": " << std::min(config.sample_size, rows) << ",\n"
        << "  \"max_iterations\": " << config.max_iterations << ",\n"
        << "  \"tolerance\": " << config.tolerance << ",\n"
        << "  \"iterations\": " << result.iterations << ",\n"
        << "  \"changed_assignments\": " << result.changed << ",\n"
        << "  \"final_sse\": " << result.sse << ",\n"
        << std::setprecision(9)
        << "  \"loading_seconds\": " << loading_seconds << ",\n"
        << "  \"initialization_seconds\": " << result.initialization_seconds << ",\n"
        << "  \"computation_seconds\": " << result.computation_seconds << ",\n"
        << std::setprecision(17);
    write_centroids(out, "initial_centroids", result.initial_centroids, config.clusters, dimensions);
    out << ",\n";
    write_centroids(out, "final_centroids", result.centroids, config.clusters, dimensions);
    out << "\n}\n";
    out.close();
    if (!out) throw std::runtime_error("cannot write " + path);
}

}  // namespace kmeans
