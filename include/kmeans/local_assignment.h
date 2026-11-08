#ifndef KMEANS_LOCAL_ASSIGNMENT_H
#define KMEANS_LOCAL_ASSIGNMENT_H
#include <stdint.h>

typedef struct {
    int threads;
    int clusters;
    int dimensions;
    double *thread_sums;
    uint64_t *thread_counts;
    uint64_t *thread_changes;
    double *thread_errors;
} LocalAssignment;

int local_assignment_init(LocalAssignment *work, int threads, int clusters, int dimensions);
void local_assignment_free(LocalAssignment *work);
void local_assignment_run(LocalAssignment *work, const float *points, uint64_t rows,
                          const double *centroids, int *assignments,
                          double *sums, uint64_t *counts, uint64_t *changed,
                          double *sse);
double local_assignment_sse(const LocalAssignment *work, const float *points,
                            uint64_t rows, const double *centroids);
#endif
