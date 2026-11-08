#include "kmeans/kernel.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace kmeans {
namespace {

// SplitMix64 gives the same sequence on every platform, so every variant starts from
// identical centroids for a given seed.
class SplitMix64 {
public:
    explicit SplitMix64(std::uint64_t seed) : state_(seed) {}

    std::uint64_t next() {
        state_ += 0x9e3779b97f4a7c15ULL;
        std::uint64_t z = state_;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
    }

    // Uniform in [0, 1) with 53 random bits.
    double unit() { return static_cast<double>(next() >> 11) * (1.0 / 9007199254740992.0); }

private:
    std::uint64_t state_;
};

}  // namespace

void validate_config(const Config& config, std::uint64_t rows) {
    if (rows == 0) throw std::invalid_argument("the matrix has no rows");
    if (config.clusters < 1) throw std::invalid_argument("K must be positive");
    if (config.max_iterations < 1) throw std::invalid_argument("the iteration limit must be positive");
    if (!(config.tolerance >= 0 && config.tolerance <= 1))
        throw std::invalid_argument("the tolerance must be between 0 and 1");
    const std::uint64_t sample = std::min(config.sample_size, rows);
    if (static_cast<std::uint64_t>(config.clusters) > sample) {
        throw std::invalid_argument("K=" + std::to_string(config.clusters) +
                                    " exceeds the K-Means++ sample of " +
                                    std::to_string(sample) + " points");
    }
}

std::vector<double> initialize_centroids(const Matrix& matrix, const Config& config) {
    validate_config(config, matrix.rows);
    const std::uint64_t sample = std::min(config.sample_size, matrix.rows);
    const int dims = matrix.dimensions;
    const auto point = [&](std::uint64_t row) {
        return matrix.values.data() + row * static_cast<std::uint64_t>(dims);
    };

    std::vector<double> centroids(static_cast<std::size_t>(config.clusters) * dims);
    std::vector<double> nearest(sample);
    SplitMix64 random(config.seed);

    const float* first = point(random.next() % sample);
    std::copy(first, first + dims, centroids.begin());
    for (std::uint64_t i = 0; i < sample; ++i)
        nearest[i] = squared_distance(point(i), centroids.data(), dims);

    // K-Means++: pick each next centre with probability proportional to D(x)^2.
    for (int cluster = 1; cluster < config.clusters; ++cluster) {
        double total = 0;
        for (const double value : nearest) total += value;
        std::uint64_t selected = 0;
        if (total > 0 && std::isfinite(total)) {
            const double target = random.unit() * total;
            double cumulative = 0;
            for (std::uint64_t i = 0; i < sample; ++i) {
                cumulative += nearest[i];
                if (cumulative > target && nearest[i] > 0) {
                    selected = i;
                    break;
                }
            }
        } else {
            selected = random.next() % sample;
        }
        double* center = centroids.data() + static_cast<std::size_t>(cluster) * dims;
        std::copy(point(selected), point(selected) + dims, center);
        for (std::uint64_t i = 0; i < sample; ++i)
            nearest[i] = std::min(nearest[i], squared_distance(point(i), center, dims));
    }
    return centroids;
}

}  // namespace kmeans
