#ifndef KMEANS_MPI_KMEANS_H
#define KMEANS_MPI_KMEANS_H
#include "kmeans/sequential.h"
#include <mpi.h>
int kmeans_mpi(const char *binary, const char *metadata, const KMeansConfig *config,
               int threads, MPI_Comm comm, KMeansResult *result, uint64_t *global_rows,
               int *dimensions, double *loading_seconds);
#endif
