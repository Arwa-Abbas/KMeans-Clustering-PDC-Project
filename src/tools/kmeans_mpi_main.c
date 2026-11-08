#include "kmeans/mpi_kmeans.h"
#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int positive(const char *text, uint64_t *value) {
    char *end;
    errno = 0;
    if (*text == '-') {
        return -1;
    }
    *value = strtoull(text, &end, 10);
    return errno || end == text || *end || !*value ? -1 : 0;
}

int main(int argc, char **argv) {
    int provided, rank, processes;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &processes);
    if (provided < MPI_THREAD_FUNNELED) {
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    const char *data = "dataset/processed/higgs_1m_f32.bin";
    const char *metadata = "dataset/processed/higgs_1m_f32.metadata.json";
    const char *output = NULL;
    int threads = 1;
    KMeansConfig config = {.clusters = 16, .max_iterations = 100,
                           .tolerance = 0.001, .seed = 42, .sample_size = 100000};
    for (int i = 1; i < argc; ++i) {
        if (i + 1 == argc) goto usage;
        const char *key = argv[i++], *value = argv[i];
        uint64_t parsed;
        if (!strcmp(key, "--data")) data = value;
        else if (!strcmp(key, "--metadata")) metadata = value;
        else if (!strcmp(key, "--output")) output = value;
        else if (!strcmp(key, "--clusters") && !positive(value, &parsed) && parsed <= 100000)
            config.clusters = (int)parsed;
        else if (!strcmp(key, "--max-iterations") && !positive(value, &parsed) && parsed <= 100000)
            config.max_iterations = (int)parsed;
        else if (!strcmp(key, "--init-sample-size") && !positive(value, &parsed))
            config.sample_size = parsed;
        else if (!strcmp(key, "--seed") && !positive(value, &parsed)) config.seed = parsed;
#ifdef KMEANS_HYBRID_CLI
        else if (!strcmp(key, "--threads") && !positive(value, &parsed) && parsed <= 1024)
            threads = (int)parsed;
#endif
        else if (!strcmp(key, "--tolerance")) {
            char *end;
            errno = 0;
            config.tolerance = strtod(value, &end);
            if (errno || end == value || *end || !isfinite(config.tolerance) ||
                config.tolerance < 0 || config.tolerance > 1) goto usage;
        } else goto usage;
    }
    KMeansResult result;
    uint64_t rows;
    int dims;
    double loading_seconds;
    int status = kmeans_mpi(data, metadata, &config, threads, MPI_COMM_WORLD, &result,
                            &rows, &dims, &loading_seconds);
    if (status) {
        if (rank == 0) fputs("error: MPI K-Means failed\n", stderr);
        MPI_Finalize();
        return 1;
    }
    if (rank == 0) {
        KMeansMatrix description = {.points = NULL, .rows = rows, .dimensions = dims};
        printf("rows=%" PRIu64 " processes=%d threads_per_process=%d clusters=%d iterations=%d final_sse=%.12g "
               "loading_seconds=%.3f initialization_seconds=%.3f computation_seconds=%.3f\n",
               rows, processes, threads, config.clusters, result.iterations, result.sse,
               loading_seconds, result.initialization_seconds, result.computation_seconds);
        if (output) {
            status = kmeans_write_result(output, &description, &config,
                                        &result, loading_seconds,
#ifdef KMEANS_HYBRID_CLI
                                        "hybrid",
#else
                                        "mpi",
#endif
                                        processes, threads);
        }
    }
    kmeans_free_result(&result);
    MPI_Finalize();
    return status != 0;
usage:
    if (rank == 0) fprintf(stderr,
        "Usage: %s [--data BIN] [--metadata JSON] [--output JSON] "
        "[--clusters K] [--max-iterations N] [--tolerance FRACTION] "
        "[--seed N] [--init-sample-size N]"
#ifdef KMEANS_HYBRID_CLI
        " [--threads N]"
#endif
        "\n", argv[0]);
    MPI_Finalize();
    return 2;
}
