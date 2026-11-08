#include "kmeans/mpi_kmeans.h"
#include "kmeans/local_assignment.h"
#include <inttypes.h>
#include <limits.h>
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Every rank executes the same collective sequence, including ranks with no rows. */
static int collective_read(MPI_File file, uint64_t start, uint64_t rows,
                           int dims, float *points, MPI_Comm comm) {
    uint64_t remaining = rows * (uint64_t)dims;
    uint64_t position = 0;
    uint64_t maximum = 0;
    MPI_Allreduce(&remaining, &maximum, 1, MPI_UINT64_T, MPI_MAX, comm);
    const uint64_t chunk = 1U << 20;
    for (uint64_t offset = 0; offset < maximum; offset += chunk) {
        int count = remaining > position ?
            (int)((remaining - position) < chunk ? remaining - position : chunk) : 0;
        MPI_Offset byte_offset = (MPI_Offset)((start * (uint64_t)dims + position) * sizeof(float));
        if (MPI_File_read_at_all(file, byte_offset, points + position,
                                 count, MPI_FLOAT, MPI_STATUS_IGNORE) != MPI_SUCCESS)
            return -1;
        position += (uint64_t)count;
    }
    return 0;
}

int kmeans_mpi(const char *binary, const char *metadata, const KMeansConfig *config,
               int threads, MPI_Comm comm, KMeansResult *result, uint64_t *global_rows,
               int *dimensions, double *loading_seconds) {
    int rank, processes;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &processes);
    memset(result, 0, sizeof *result);
    uint64_t rows = 0;
    int dims = 0;
    int valid = kmeans_metadata_dimensions(metadata, &rows, &dims) == 0 &&
        config->clusters > 0 && (uint64_t)config->clusters <= rows &&
        config->max_iterations > 0 && config->sample_size >= (uint64_t)config->clusters &&
        config->tolerance >= 0 && config->tolerance <= 1 && threads >= 1;
    int all_valid;
    MPI_Allreduce(&valid, &all_valid, 1, MPI_INT, MPI_MIN, comm);
    if (!all_valid) return -1;
    *global_rows = rows;
    *dimensions = dims;
    uint64_t base = rows / (uint64_t)processes;
    uint64_t extra = rows % (uint64_t)processes;
    uint64_t local_rows = base + ((uint64_t)rank < extra);
    uint64_t start_row = (uint64_t)rank * base +
        ((uint64_t)rank < extra ? (uint64_t)rank : extra);
    float *points = malloc((local_rows ? local_rows : 1) * (uint64_t)dims * sizeof(float));
    int *assignments = malloc((local_rows ? local_rows : 1) * sizeof(int));
    size_t cells = (size_t)config->clusters * dims;
    double *local_sums = calloc(cells, sizeof(double));
    double *global_sums = calloc(cells, sizeof(double));
    uint64_t *local_counts = calloc((size_t)config->clusters, sizeof(uint64_t));
    result->counts = calloc((size_t)config->clusters, sizeof(uint64_t));
    result->centroids = malloc(cells * sizeof(double));
    result->initial_centroids = malloc(cells * sizeof(double));
    LocalAssignment work;
    int work_ready = local_assignment_init(&work, threads, config->clusters, dims) == 0;
    int ready = points && assignments && local_sums && global_sums &&
        local_counts && result->counts && result->centroids && result->initial_centroids &&
        work_ready;
    MPI_Allreduce(&ready, &all_valid, 1, MPI_INT, MPI_MIN, comm);
    if (!all_valid) goto fail;
    memset(assignments, 0xff, local_rows * sizeof(int));

    MPI_File file;
    if (MPI_File_open(comm, (char *)binary, MPI_MODE_RDONLY, MPI_INFO_NULL, &file)
        != MPI_SUCCESS) goto fail;
    MPI_Offset actual_bytes;
    MPI_File_get_size(file, &actual_bytes);
    int size_ok = actual_bytes == (MPI_Offset)(rows * (uint64_t)dims * sizeof(float));
    MPI_Allreduce(&size_ok, &all_valid, 1, MPI_INT, MPI_MIN, comm);
    if (!all_valid) { MPI_File_close(&file); goto fail; }
    double started = MPI_Wtime();
    int read_ok = collective_read(file, start_row, local_rows, dims, points, comm) == 0;
    MPI_File_close(&file);
    MPI_Allreduce(&read_ok, &all_valid, 1, MPI_INT, MPI_MIN, comm);
    if (!all_valid) goto fail;
    *loading_seconds = MPI_Wtime() - started;

    started = MPI_Wtime();
    int init_ok = 1;
    if (rank == 0) {
        uint64_t sample = config->sample_size < rows ? config->sample_size : rows;
        float *sample_points = malloc(sample * (uint64_t)dims * sizeof(float));
        if (!sample_points) init_ok = 0;
        if (init_ok) {
            FILE *source = fopen(binary, "rb");
            if (!source || fread(sample_points, sizeof(float), sample * (uint64_t)dims,
                                 source) != sample * (uint64_t)dims) init_ok = 0;
            if (source) fclose(source);
        }
        if (init_ok) {
            KMeansMatrix sample_matrix = {.points = sample_points, .rows = sample,
                                          .dimensions = dims};
            init_ok = kmeans_initialize(&sample_matrix, config, result->centroids) == 0;
        }
        free(sample_points);
    }
    MPI_Bcast(&init_ok, 1, MPI_INT, 0, comm);
    if (!init_ok) goto fail;
    MPI_Bcast(result->centroids, (int)cells, MPI_DOUBLE, 0, comm);
    memcpy(result->initial_centroids, result->centroids, cells * sizeof(double));
    result->initialization_seconds = MPI_Wtime() - started;
    started = MPI_Wtime();
    for (int iteration = 0; iteration < config->max_iterations; ++iteration) {
        uint64_t local_changed = 0, changed = 0;
        double local_sse = 0, sse = 0;
        local_assignment_run(&work, points, local_rows, result->centroids,
                             assignments, local_sums, local_counts,
                             &local_changed, &local_sse);
        MPI_Allreduce(local_sums, global_sums, (int)cells, MPI_DOUBLE, MPI_SUM, comm);
        MPI_Allreduce(local_counts, result->counts, config->clusters,
                      MPI_UINT64_T, MPI_SUM, comm);
        MPI_Allreduce(&local_changed, &changed, 1, MPI_UINT64_T, MPI_SUM, comm);
        MPI_Allreduce(&local_sse, &sse, 1, MPI_DOUBLE, MPI_SUM, comm);
        int empty = 0;
        for (int c = 0; c < config->clusters; ++c) {
            if (!result->counts[c]) { ++empty; continue; }
            for (int d = 0; d < dims; ++d)
                result->centroids[(size_t)c * dims + d] =
                    global_sums[(size_t)c * dims + d] / result->counts[c];
        }
        result->iterations = iteration + 1;
        result->changed = changed;
        result->sse = sse;
        if (rank == 0)
            printf("iteration=%d changed=%" PRIu64 " fraction=%.8f assignment_sse=%.9g empty=%d\n",
                   result->iterations, changed, (double)changed / rows, sse, empty);
        if ((double)changed / rows < config->tolerance) break;
    }
    double local_elapsed = MPI_Wtime() - started;
    MPI_Allreduce(&local_elapsed, &result->computation_seconds, 1, MPI_DOUBLE, MPI_MAX, comm);
    double local_final_sse = local_assignment_sse(&work, points, local_rows,
                                                  result->centroids);
    MPI_Allreduce(&local_final_sse, &result->sse, 1, MPI_DOUBLE, MPI_SUM, comm);
    local_assignment_free(&work);
    free(points); free(assignments); free(local_sums); free(global_sums); free(local_counts);
    return 0;
fail:
    local_assignment_free(&work);
    free(points); free(assignments); free(local_sums); free(global_sums); free(local_counts);
    kmeans_free_result(result);
    return -1;
}
