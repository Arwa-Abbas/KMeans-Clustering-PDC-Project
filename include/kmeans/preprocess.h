#ifndef KMEANS_PREPROCESS_H
#define KMEANS_PREPROCESS_H

#include <stdbool.h>

typedef struct {
    const char *archive_path;
    const char *binary_path;
    const char *metadata_path;
    unsigned long long max_rows;
    bool overwrite;
} PreprocessOptions;

int preprocess_higgs(const PreprocessOptions *options);

#endif

