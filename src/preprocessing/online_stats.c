#include "kmeans/online_stats.h"

#include <math.h>
#include <string.h>

void online_stats_init(OnlineStats *stats) {
    memset(stats, 0, sizeof(*stats));
}

void online_stats_update(OnlineStats *stats,
                         const float values[HIGGS_FEATURE_COLUMNS]) {
    ++stats->count;
    double count = (double)stats->count;
    for (int feature = 0; feature < HIGGS_FEATURE_COLUMNS; ++feature) {
        double value = values[feature];
        double delta = value - stats->mean[feature];
        stats->mean[feature] += delta / count;
        double delta_after_update = value - stats->mean[feature];
        stats->m2[feature] += delta * delta_after_update;
    }
}

int online_stats_population_stddev(
    const OnlineStats *stats, double stddev[HIGGS_FEATURE_COLUMNS]) {
    if (stats->count == 0) return -1;
    for (int feature = 0; feature < HIGGS_FEATURE_COLUMNS; ++feature) {
        double variance = stats->m2[feature] / (double)stats->count;
        if (variance <= 0.0 || !isfinite(variance)) return -1;
        stddev[feature] = sqrt(variance);
    }
    return 0;
}

