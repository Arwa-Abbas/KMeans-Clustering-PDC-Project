#include "kmeans/online_stats.hpp"

#include <cmath>
#include <iostream>

int main() {
    using namespace kmeans;
    FeatureRow first{};
    FeatureRow second{};
    first.fill(1.0F);
    second.fill(3.0F);
    OnlineStats stats;
    stats.update(first);
    stats.update(second);
    OnlineStats::Values stddev{};
    if (!stats.population_stddev(stddev)) return 1;
    for (int feature = 0; feature < kHiggsFeatures; ++feature) {
        if (std::fabs(stats.mean()[feature] - 2.0) > 1e-12 || std::fabs(stddev[feature] - 1.0) > 1e-12) {
            std::cerr << "statistics test failed at feature " << feature << '\n';
            return 1;
        }
    }
    std::cout << "online statistics test passed\n";
    return 0;
}
