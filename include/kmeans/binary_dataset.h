#ifndef KMEANS_BINARY_DATASET_H
#define KMEANS_BINARY_DATASET_H

#include <stdbool.h>
#include <stdio.h>

#include "kmeans/higgs_dataset.h"

typedef struct {
    FILE *stream;
    char *final_path;
    char *partial_path;
    unsigned long long rows_written;
} BinaryDatasetWriter;

int binary_dataset_writer_open(BinaryDatasetWriter *writer, const char *path,
                               bool overwrite);
int binary_dataset_writer_write(
    BinaryDatasetWriter *writer,
    const float normalized_features[HIGGS_FEATURE_COLUMNS]);
int binary_dataset_writer_finish(BinaryDatasetWriter *writer);
void binary_dataset_writer_abort(BinaryDatasetWriter *writer);

#endif

