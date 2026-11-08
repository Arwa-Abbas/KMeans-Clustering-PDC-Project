#define _POSIX_C_SOURCE 200809L
#include "kmeans/sequential.h"
#include <errno.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef KMEANS_OPENMP_CLI
#include <omp.h>
#endif

static double seconds(void) {
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return now.tv_sec + now.tv_nsec * 1e-9;
}

static int number(const char *text, uint64_t *value) {
    char *end;
    errno = 0;
    if (*text == '-') return -1;
    *value = strtoull(text, &end, 10);
    return errno || end == text || *end || !*value ? -1 : 0;
}

int main(int argc, char **argv) {
    const char *data = "dataset/processed/higgs_1m_f32.bin";
    const char *metadata = "dataset/processed/higgs_1m_f32.metadata.json";
    const char *output = NULL;
    int threads = 1;
    KMeansConfig config = {.clusters = 16, .max_iterations = 100,
                           .tolerance = 0.001, .seed = 42, .sample_size = 100000};
    for (int i = 1; i < argc; ++i) {
        if (i + 1 >= argc) goto usage;
        const char *key = argv[i++];
        const char *value = argv[i];
        uint64_t parsed;
        if (!strcmp(key, "--data")) data = value;
        else if (!strcmp(key, "--metadata")) metadata = value;
        else if (!strcmp(key, "--output")) output = value;
        else if (!strcmp(key, "--clusters") && !number(value, &parsed) && parsed <= 100000)
            config.clusters = (int)parsed;
        else if (!strcmp(key, "--max-iterations") && !number(value, &parsed) && parsed <= 100000)
            config.max_iterations = (int)parsed;
        else if (!strcmp(key, "--init-sample-size") && !number(value, &parsed))
            config.sample_size = parsed;
        else if (!strcmp(key, "--seed") && !number(value, &parsed)) config.seed = parsed;
#ifdef KMEANS_OPENMP_CLI
        else if (!strcmp(key, "--threads") && !number(value, &parsed) && parsed <= 1024)
            threads = (int)parsed;
#endif
        else if (!strcmp(key, "--tolerance")) {
            char *end;
            errno = 0;
            config.tolerance = strtod(value, &end);
            if (errno || end == value || *end || config.tolerance < 0 ||
                config.tolerance > 1) goto usage;
        } else goto usage;
    }
    KMeansMatrix matrix;
    double started = seconds();
    if (kmeans_load_matrix(data, metadata, &matrix)) return 1;
    double loading_seconds = seconds() - started;
    KMeansResult result;
    int run_status;
#ifdef KMEANS_OPENMP_CLI
    run_status = kmeans_openmp(&matrix, &config, threads, &result);
#else
    run_status = kmeans_sequential(&matrix, &config, &result);
#endif
    if (run_status) {
        fprintf(stderr, "error: invalid configuration or insufficient memory\n");
        kmeans_free_matrix(&matrix);
        return 1;
    }
    printf("rows=%" PRIu64 " clusters=%d iterations=%d final_sse=%.12g "
           "loading_seconds=%.3f initialization_seconds=%.3f computation_seconds=%.3f\n",
           matrix.rows, config.clusters, result.iterations, result.sse,
           loading_seconds, result.initialization_seconds, result.computation_seconds);
    int status = output && kmeans_write_result(output, &matrix, &config, &result,
                                               loading_seconds,
#ifdef KMEANS_OPENMP_CLI
                                               "openmp", threads
#else
                                               "sequential", threads
#endif
                                               ) ? 1 : 0;
    kmeans_free_result(&result);
    kmeans_free_matrix(&matrix);
    return status;
usage:
    fprintf(stderr, "Usage: %s [--data BIN] [--metadata JSON] [--output JSON] "
            "[--clusters K] [--max-iterations N] [--tolerance FRACTION] "
            "[--seed N] [--init-sample-size N]"
#ifdef KMEANS_OPENMP_CLI
            " [--threads N]"
#endif
            "\n", argv[0]);
    return 2;
}
