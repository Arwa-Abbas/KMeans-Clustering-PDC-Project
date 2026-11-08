#include "kmeans/online_stats.hpp"

#include <cmath>

namespace kmeans {

void OnlineStats::update(const FeatureRow& row) {
    // Welford's update avoids subtracting large, nearly equal sums.
    ++count_;
    const double count = static_cast<double>(count_);
    for (int feature = 0; feature < kHiggsFeatures; ++feature) {
        const double value = row[feature];
        const double delta = value - mean_[feature];
        mean_[feature] += delta / count;
        m2_[feature] += delta * (value - mean_[feature]);
    }
}

bool OnlineStats::population_stddev(Values& stddev) const {
    if (count_ == 0) return false;
    for (int feature = 0; feature < kHiggsFeatures; ++feature) {
        const double variance = m2_[feature] / static_cast<double>(count_);
        if (variance <= 0.0 || !std::isfinite(variance)) return false;
        stddev[feature] = std::sqrt(variance);
    }
    return true;
}

}  // namespace kmeans
