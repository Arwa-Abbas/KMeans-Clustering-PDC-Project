#include "kmeans/local_assignment.h"
#include <omp.h>
#include <stdlib.h>
#include <string.h>

static double distance2(const float *point, const double *center, int dimensions) {
    double value = 0;
    for (int d = 0; d < dimensions; ++d) {
        double delta = (double)point[d] - center[d];
        value += delta * delta;
    }
    return value;
}

int local_assignment_init(LocalAssignment *work, int threads, int clusters, int dimensions) {
    memset(work, 0, sizeof *work);
    if (threads < 1 || clusters < 1 || dimensions < 1) {
        return -1;
    }
    work->threads = threads;
    work->clusters = clusters;
    work->dimensions = dimensions;
    size_t cells = (size_t)threads * clusters * dimensions;
    work->thread_sums = calloc(cells, sizeof(double));
    work->thread_counts = calloc((size_t)threads * clusters, sizeof(uint64_t));
    work->thread_changes = calloc((size_t)threads, sizeof(uint64_t));
    work->thread_errors = calloc((size_t)threads, sizeof(double));
    if (!work->thread_sums || !work->thread_counts || !work->thread_changes ||
        !work->thread_errors) {
        local_assignment_free(work);
        return -1;
    }
    return 0;
}

void local_assignment_free(LocalAssignment *work) {
    free(work->thread_sums);
    free(work->thread_counts);
    free(work->thread_changes);
    free(work->thread_errors);
    memset(work, 0, sizeof *work);
}

void local_assignment_run(LocalAssignment *work, const float *points, uint64_t rows,
                          const double *centroids, int *assignments,
                          double *sums, uint64_t *counts, uint64_t *changed,
                          double *sse) {
    int dimensions = work->dimensions, clusters = work->clusters;
    size_t cells = (size_t)clusters * dimensions;
    memset(work->thread_sums, 0, (size_t)work->threads * cells * sizeof(double));
    memset(work->thread_counts, 0, (size_t)work->threads * clusters * sizeof(uint64_t));
    memset(work->thread_changes, 0, (size_t)work->threads * sizeof(uint64_t));
    memset(work->thread_errors, 0, (size_t)work->threads * sizeof(double));
    /* One private accumulator per thread avoids writes to shared cluster totals. */
    #pragma omp parallel num_threads(work->threads)
    {
        int tid = omp_get_thread_num();
        double *private_sums = work->thread_sums + (size_t)tid * cells;
        uint64_t *private_counts = work->thread_counts + (size_t)tid * clusters;
        uint64_t private_changed = 0;
        double private_sse = 0;
        #pragma omp for schedule(static)
        for (uint64_t row = 0; row < rows; ++row) {
            const float *point = points + row * (uint64_t)dimensions;
            int nearest = 0;
            double best = distance2(point, centroids, dimensions);
            for (int cluster = 1; cluster < clusters; ++cluster) {
                double candidate = distance2(point,
                    centroids + (size_t)cluster * dimensions, dimensions);
                if (candidate < best) {
                    best = candidate;
                    nearest = cluster;
                }
            }
            private_sse += best;
            if (assignments[row] != nearest) ++private_changed;
            assignments[row] = nearest;
            ++private_counts[nearest];
            for (int d = 0; d < dimensions; ++d)
                private_sums[(size_t)nearest * dimensions + d] += point[d];
        }
        work->thread_changes[tid] = private_changed;
        work->thread_errors[tid] = private_sse;
    }
    /* Combine thread results once, before the rank participates in MPI. */
    memset(sums, 0, cells * sizeof(double));
    memset(counts, 0, (size_t)clusters * sizeof(uint64_t));
    *changed = 0;
    *sse = 0;
    for (int thread = 0; thread < work->threads; ++thread) {
        *changed += work->thread_changes[thread];
        *sse += work->thread_errors[thread];
        for (int cluster = 0; cluster < clusters; ++cluster) {
            counts[cluster] += work->thread_counts[(size_t)thread * clusters + cluster];
            for (int d = 0; d < dimensions; ++d)
                sums[(size_t)cluster * dimensions + d] +=
                    work->thread_sums[(size_t)thread * cells +
                                      (size_t)cluster * dimensions + d];
        }
    }
}

double local_assignment_sse(const LocalAssignment *work, const float *points,
                            uint64_t rows, const double *centroids) {
    double sse = 0;
    int dimensions = work->dimensions, clusters = work->clusters;
    #pragma omp parallel for num_threads(work->threads) schedule(static) reduction(+:sse)
    for (uint64_t row = 0; row < rows; ++row) {
        const float *point = points + row * (uint64_t)dimensions;
        double best = distance2(point, centroids, dimensions);
        for (int cluster = 1; cluster < clusters; ++cluster) {
            double candidate = distance2(point,
                centroids + (size_t)cluster * dimensions, dimensions);
            if (candidate < best) best = candidate;
        }
        sse += best;
    }
    return sse;
}
