#ifndef KMEANS_HIGGS_DATASET_H
#define KMEANS_HIGGS_DATASET_H

#include <stdbool.h>

#define HIGGS_EXPECTED_COLUMNS 29
#define HIGGS_FEATURE_COLUMNS 28
#define HIGGS_EXPECTED_ROWS 11000000ULL

typedef struct {
    const char *archive_path;
    const char *output_path;
    unsigned long long sample_rows;
    unsigned long long expected_rows;
    bool full_scan;
    bool overwrite;
} HiggsDatasetOptions;

int higgs_inspect_archive(const HiggsDatasetOptions *options);
int higgs_extract_archive(const HiggsDatasetOptions *options);

#endif

