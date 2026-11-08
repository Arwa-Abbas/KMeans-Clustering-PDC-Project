#pragma once

#include "kmeans/higgs_dataset.hpp"

#include <array>
#include <cstdint>

namespace kmeans {

// Welford's streaming mean and variance, so the 8 GB CSV never has to fit in memory.
class OnlineStats {
public:
    using Values = std::array<double, kHiggsFeatures>;

    void update(const FeatureRow& row);
    std::uint64_t count() const { return count_; }
    const Values& mean() const { return mean_; }
    // Population standard deviation; false if any feature has zero or invalid variance.
    bool population_stddev(Values& stddev) const;

private:
    std::uint64_t count_ = 0;
    Values mean_{};
    Values m2_{};
};

}  // namespace kmeans
