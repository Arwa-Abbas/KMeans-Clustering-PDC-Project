#include "kmeans/sequential.h"
#include <inttypes.h>
#include <stdio.h>

static void write_centroids(FILE *stream, const char *name, const double *values,
                            int clusters, int dimensions) {
    fprintf(stream, "  \"%s\": [\n", name);
    for (int c = 0; c < clusters; ++c) {
        fputs("    [", stream);
        for (int d = 0; d < dimensions; ++d)
            fprintf(stream, "%s%.17g", d ? ", " : "", values[(size_t)c * dimensions + d]);
        fprintf(stream, "]%s\n", c + 1 == clusters ? "" : ",");
    }
    fputs("  ]", stream);
}

int kmeans_write_result(const char *path, const KMeansMatrix *matrix,
                        const KMeansConfig *config, const KMeansResult *result,
                        double loading_seconds, const char *variant, int processes,
                        int threads) {
    FILE *stream = fopen(path, "w");
    if (!stream) {
        perror(path);
        return -1;
    }
    fprintf(stream, "{\n  \"variant\": \"%s\",\n  \"processes\": %d,\n  \"threads\": %d,\n"
            "  \"rows\": %" PRIu64 ",\n  \"features\": %d,\n"
            "  \"clusters\": %d,\n  \"seed\": %" PRIu64 ",\n"
            "  \"init_sample_size\": %" PRIu64 ",\n"
            "  \"iterations\": %d,\n  \"changed_assignments\": %" PRIu64 ",\n"
            "  \"final_sse\": %.17g,\n  \"loading_seconds\": %.9g,\n"
            "  \"initialization_seconds\": %.9g,\n"
            "  \"computation_seconds\": %.9g,\n",
            variant, processes, threads, matrix->rows, matrix->dimensions, config->clusters, config->seed,
            config->sample_size < matrix->rows ? config->sample_size : matrix->rows,
            result->iterations, result->changed, result->sse, loading_seconds,
            result->initialization_seconds, result->computation_seconds);
    write_centroids(stream, "initial_centroids", result->initial_centroids,
                    config->clusters, matrix->dimensions);
    fputs(",\n", stream);
    write_centroids(stream, "final_centroids", result->centroids,
                    config->clusters, matrix->dimensions);
    fputs("\n}\n", stream);
    return fclose(stream) == 0 ? 0 : -1;
}
