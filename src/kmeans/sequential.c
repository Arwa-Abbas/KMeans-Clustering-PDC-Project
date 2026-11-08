#define _POSIX_C_SOURCE 200809L
#include "kmeans/sequential.h"
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double seconds(void) {
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return now.tv_sec + now.tv_nsec * 1e-9;
}

static uint64_t random_u64(uint64_t *state) {
    *state += UINT64_C(0x9e3779b97f4a7c15);
    uint64_t z = *state;
    z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
    return z ^ (z >> 31);
}

static double distance2(const float *point, const double *centroid, int dimensions) {
    double distance = 0;
    for (int d = 0; d < dimensions; ++d) {
        double delta = (double)point[d] - centroid[d];
        distance += delta * delta;
    }
    return distance;
}

int kmeans_initialize(const KMeansMatrix *matrix, const KMeansConfig *config,
                      double *centroids) {
    uint64_t sample = config->sample_size < matrix->rows ? config->sample_size : matrix->rows;
    if (!sample || config->clusters < 1 || (uint64_t)config->clusters > sample) return -1;
    double *nearest = malloc(sample * sizeof *nearest);
    if (!nearest) return -1;
    uint64_t state = config->seed;
    int dims = matrix->dimensions;
    uint64_t first = random_u64(&state) % sample;
    const float *point = matrix->points + first * (uint64_t)dims;
    for (int d = 0; d < dims; ++d) centroids[d] = point[d];
    for (uint64_t i = 0; i < sample; ++i)
        nearest[i] = distance2(matrix->points + i * (uint64_t)dims, centroids, dims);

    for (int cluster = 1; cluster < config->clusters; ++cluster) {
        double total = 0;
        for (uint64_t i = 0; i < sample; ++i) total += nearest[i];
        uint64_t selected = 0;
        if (total > 0 && isfinite(total)) {
            double target = (random_u64(&state) >> 11) * (1.0 / 9007199254740992.0) * total;
            double cumulative = 0;
            for (uint64_t i = 0; i < sample; ++i) {
                cumulative += nearest[i];
                if (cumulative > target && nearest[i] > 0) { selected = i; break; }
            }
        } else {
            selected = random_u64(&state) % sample;
        }
        point = matrix->points + selected * (uint64_t)dims;
        double *center = centroids + (size_t)cluster * dims;
        for (int d = 0; d < dims; ++d) center[d] = point[d];
        for (uint64_t i = 0; i < sample; ++i) {
            double candidate = distance2(matrix->points + i * (uint64_t)dims, center, dims);
            if (candidate < nearest[i]) nearest[i] = candidate;
        }
    }
    free(nearest);
    return 0;
}

int kmeans_sequential(const KMeansMatrix *matrix, const KMeansConfig *config,
                      KMeansResult *result) {
    memset(result, 0, sizeof *result);
    if (!matrix->points || matrix->dimensions <= 0 || !matrix->rows ||
        config->clusters <= 0 || config->max_iterations <= 0 ||
        config->tolerance < 0 || config->tolerance > 1) return -1;
    size_t cells = (size_t)config->clusters * matrix->dimensions;
    result->initial_centroids = malloc(cells * sizeof(double));
    result->centroids = malloc(cells * sizeof(double));
    result->counts = calloc((size_t)config->clusters, sizeof(uint64_t));
    double *sums = calloc(cells, sizeof(double));
    int *assignments = malloc(matrix->rows * sizeof(int));
    if (!result->initial_centroids || !result->centroids || !result->counts ||
        !sums || !assignments) goto fail;
    memset(assignments, 0xff, matrix->rows * sizeof(int));
    double start = seconds();
    if (kmeans_initialize(matrix, config, result->centroids)) goto fail;
    result->initialization_seconds = seconds() - start;
    memcpy(result->initial_centroids, result->centroids, cells * sizeof(double));
    start = seconds();
    for (int iteration = 0; iteration < config->max_iterations; ++iteration) {
        memset(sums, 0, cells * sizeof(double));
        memset(result->counts, 0, (size_t)config->clusters * sizeof(uint64_t));
        uint64_t changed = 0;
        double sse = 0;
        for (uint64_t row = 0; row < matrix->rows; ++row) {
            const float *point = matrix->points + row * (uint64_t)matrix->dimensions;
            int nearest = 0;
            double best = distance2(point, result->centroids, matrix->dimensions);
            for (int cluster = 1; cluster < config->clusters; ++cluster) {
                double candidate = distance2(point, result->centroids +
                    (size_t)cluster * matrix->dimensions, matrix->dimensions);
                if (candidate < best) { best = candidate; nearest = cluster; }
            }
            sse += best;
            if (assignments[row] != nearest) ++changed;
            assignments[row] = nearest;
            ++result->counts[nearest];
            for (int d = 0; d < matrix->dimensions; ++d)
                sums[(size_t)nearest * matrix->dimensions + d] += point[d];
        }
        int empty = 0;
        for (int cluster = 0; cluster < config->clusters; ++cluster) {
            if (!result->counts[cluster]) { ++empty; continue; }
            for (int d = 0; d < matrix->dimensions; ++d)
                result->centroids[(size_t)cluster * matrix->dimensions + d] =
                    sums[(size_t)cluster * matrix->dimensions + d] / result->counts[cluster];
        }
        result->iterations = iteration + 1;
        result->changed = changed;
        result->sse = sse;
        printf("iteration=%d changed=%" PRIu64 " fraction=%.8f assignment_sse=%.9g empty=%d\n",
               result->iterations, changed, (double)changed / matrix->rows, sse, empty);
        if ((double)changed / matrix->rows < config->tolerance) break;
    }
    result->computation_seconds = seconds() - start;
    /* SSE above uses centroids before the last update. Recompute against final centroids. */
    result->sse = 0;
    for (uint64_t row = 0; row < matrix->rows; ++row) {
        const float *point = matrix->points + row * (uint64_t)matrix->dimensions;
        double best = distance2(point, result->centroids, matrix->dimensions);
        for (int cluster = 1; cluster < config->clusters; ++cluster) {
            double candidate = distance2(point, result->centroids +
                (size_t)cluster * matrix->dimensions, matrix->dimensions);
            if (candidate < best) best = candidate;
        }
        result->sse += best;
    }
    free(sums);
    free(assignments);
    return 0;
fail:
    free(sums);
    free(assignments);
    kmeans_free_result(result);
    return -1;
}

void kmeans_free_result(KMeansResult *result) {
    free(result->initial_centroids);
    free(result->centroids);
    free(result->counts);
    memset(result, 0, sizeof *result);
}
