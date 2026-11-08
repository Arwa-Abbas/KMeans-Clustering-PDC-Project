#ifndef KMEANS_ONLINE_STATS_H
#define KMEANS_ONLINE_STATS_H

#include "kmeans/higgs_dataset.h"

typedef struct {
    unsigned long long count;
    double mean[HIGGS_FEATURE_COLUMNS];
    double m2[HIGGS_FEATURE_COLUMNS];
} OnlineStats;

void online_stats_init(OnlineStats *stats);
void online_stats_update(OnlineStats *stats,
                         const float values[HIGGS_FEATURE_COLUMNS]);
int online_stats_population_stddev(
    const OnlineStats *stats, double stddev[HIGGS_FEATURE_COLUMNS]);

#endif

