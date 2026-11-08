#define _POSIX_C_SOURCE 200809L
#include "kmeans/sequential.h"
#include <inttypes.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static double distance2(const float *point, const double *center, int dims) {
    double result = 0;
    for (int d = 0; d < dims; ++d) {
        double delta = (double)point[d] - center[d];
        result += delta * delta;
    }
    return result;
}

int kmeans_openmp(const KMeansMatrix *matrix, const KMeansConfig *config,
                  int threads, KMeansResult *result) {
    memset(result, 0, sizeof *result);
    if (!matrix->points || !matrix->rows || matrix->dimensions <= 0 || threads < 1 ||
        config->clusters < 1 || config->max_iterations < 1 ||
        config->tolerance < 0 || config->tolerance > 1 ||
        (uint64_t)config->clusters > matrix->rows) return -1;
    const int dims = matrix->dimensions;
    const int clusters = config->clusters;
    const size_t cells = (size_t)clusters * dims;
    result->initial_centroids = malloc(cells * sizeof(double));
    result->centroids = malloc(cells * sizeof(double));
    result->counts = calloc((size_t)clusters, sizeof(uint64_t));
    int *assignments = malloc(matrix->rows * sizeof(int));
    double *sums = calloc((size_t)threads * cells, sizeof(double));
    uint64_t *counts = calloc((size_t)threads * clusters, sizeof(uint64_t));
    uint64_t *changes = calloc((size_t)threads, sizeof(uint64_t));
    double *errors = calloc((size_t)threads, sizeof(double));
    if (!result->initial_centroids || !result->centroids || !result->counts ||
        !assignments || !sums || !counts || !changes || !errors) {
        goto fail;
    }
    memset(assignments, 0xff, matrix->rows * sizeof(int));
    double start = omp_get_wtime();
    if (kmeans_initialize(matrix, config, result->centroids)) goto fail;
    result->initialization_seconds = omp_get_wtime() - start;
    memcpy(result->initial_centroids, result->centroids, cells * sizeof(double));
    start = omp_get_wtime();

    for (int iteration = 0; iteration < config->max_iterations; ++iteration) {
        memset(sums, 0, (size_t)threads * cells * sizeof(double));
        memset(counts, 0, (size_t)threads * clusters * sizeof(uint64_t));
        memset(changes, 0, (size_t)threads * sizeof(uint64_t));
        memset(errors, 0, (size_t)threads * sizeof(double));
        /* Each thread writes only to its own sums and counts. */
        #pragma omp parallel num_threads(threads)
        {
            int tid = omp_get_thread_num();
            double *local_sums = sums + (size_t)tid * cells;
            uint64_t *local_counts = counts + (size_t)tid * clusters;
            uint64_t local_changed = 0;
            double local_sse = 0;
            #pragma omp for schedule(static)
            for (uint64_t row = 0; row < matrix->rows; ++row) {
                const float *point = matrix->points + row * (uint64_t)dims;
                int nearest = 0;
                double best = distance2(point, result->centroids, dims);
                for (int cluster = 1; cluster < clusters; ++cluster) {
                    double candidate = distance2(point,
                        result->centroids + (size_t)cluster * dims, dims);
                    if (candidate < best) {
                        best = candidate;
                        nearest = cluster;
                    }
                }
                local_sse += best;
                if (assignments[row] != nearest) ++local_changed;
                assignments[row] = nearest;
                ++local_counts[nearest];
                for (int d = 0; d < dims; ++d)
                    local_sums[(size_t)nearest * dims + d] += point[d];
            }
            changes[tid] = local_changed;
            errors[tid] = local_sse;
        }
        uint64_t changed = 0;
        double sse = 0;
        memset(result->counts, 0, (size_t)clusters * sizeof(uint64_t));
        for (int tid = 0; tid < threads; ++tid) {
            changed += changes[tid];
            sse += errors[tid];
            for (int cluster = 0; cluster < clusters; ++cluster) {
                result->counts[cluster] += counts[(size_t)tid * clusters + cluster];
                if (tid != 0) {
                    for (int d = 0; d < dims; ++d)
                        sums[(size_t)cluster * dims + d] +=
                            sums[(size_t)tid * cells + (size_t)cluster * dims + d];
                }
            }
        }
        int empty = 0;
        for (int cluster = 0; cluster < clusters; ++cluster) {
            if (!result->counts[cluster]) {
                ++empty;
                continue;
            }
            for (int d = 0; d < dims; ++d)
                result->centroids[(size_t)cluster * dims + d] =
                    sums[(size_t)cluster * dims + d] / result->counts[cluster];
        }
        result->iterations = iteration + 1;
        result->changed = changed;
        result->sse = sse;
        printf("iteration=%d changed=%" PRIu64 " fraction=%.8f assignment_sse=%.9g empty=%d\n",
               result->iterations, changed, (double)changed / matrix->rows, sse, empty);
        if ((double)changed / matrix->rows < config->tolerance) break;
    }
    result->computation_seconds = omp_get_wtime() - start;
    double final_sse = 0;
    #pragma omp parallel for num_threads(threads) schedule(static) reduction(+:final_sse)
    for (uint64_t row = 0; row < matrix->rows; ++row) {
        const float *point = matrix->points + row * (uint64_t)dims;
        double best = distance2(point, result->centroids, dims);
        for (int cluster = 1; cluster < clusters; ++cluster) {
            double candidate = distance2(point,
                result->centroids + (size_t)cluster * dims, dims);
            if (candidate < best) best = candidate;
        }
        final_sse += best;
    }
    result->sse = final_sse;
    free(assignments);
    free(sums);
    free(counts);
    free(changes);
    free(errors);
    return 0;
fail:
    free(assignments);
    free(sums);
    free(counts);
    free(changes);
    free(errors);
    kmeans_free_result(result);
    return -1;
}
