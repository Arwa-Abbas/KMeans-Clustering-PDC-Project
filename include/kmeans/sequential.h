#ifndef KMEANS_SEQUENTIAL_H
#define KMEANS_SEQUENTIAL_H

#include <stdint.h>

typedef struct {
    float *points;
    uint64_t rows;
    int dimensions;
} KMeansMatrix;

typedef struct {
    int clusters;
    int max_iterations;
    double tolerance;
    uint64_t seed;
    uint64_t sample_size;
} KMeansConfig;

typedef struct {
    double *initial_centroids;
    double *centroids;
    uint64_t *counts;
    int iterations;
    uint64_t changed;
    double sse;
    double initialization_seconds;
    double computation_seconds;
} KMeansResult;

int kmeans_load_matrix(const char *binary, const char *metadata, KMeansMatrix *matrix);
void kmeans_free_matrix(KMeansMatrix *matrix);
int kmeans_initialize(const KMeansMatrix *matrix, const KMeansConfig *config,
                      double *centroids);
int kmeans_sequential(const KMeansMatrix *matrix, const KMeansConfig *config,
                      KMeansResult *result);
void kmeans_free_result(KMeansResult *result);
int kmeans_write_result(const char *path, const KMeansMatrix *matrix,
                        const KMeansConfig *config, const KMeansResult *result,
                        double loading_seconds);

#endif
