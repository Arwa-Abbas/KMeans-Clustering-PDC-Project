#include "kmeans/online_stats.h"

#include <math.h>
#include <stdio.h>

int main(void) {
    OnlineStats stats;
    float first[HIGGS_FEATURE_COLUMNS] = {0};
    float second[HIGGS_FEATURE_COLUMNS] = {0};
    double stddev[HIGGS_FEATURE_COLUMNS];
    for (int feature = 0; feature < HIGGS_FEATURE_COLUMNS; ++feature) {
        first[feature] = 1.0F;
        second[feature] = 3.0F;
    }
    online_stats_init(&stats);
    online_stats_update(&stats, first);
    online_stats_update(&stats, second);
    if (online_stats_population_stddev(&stats, stddev) != 0) return 1;
    for (int feature = 0; feature < HIGGS_FEATURE_COLUMNS; ++feature) {
        if (fabs(stats.mean[feature] - 2.0) > 1e-12 ||
            fabs(stddev[feature] - 1.0) > 1e-12) {
            fprintf(stderr, "statistics test failed at feature %d\n", feature);
            return 1;
        }
    }
    puts("online statistics test passed");
    return 0;
}

