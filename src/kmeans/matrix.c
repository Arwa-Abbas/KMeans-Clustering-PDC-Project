#define _POSIX_C_SOURCE 200809L
#include "kmeans/sequential.h"
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

static int metadata_number(const char *text, const char *key, uint64_t *value) {
    char needle[80];
    snprintf(needle, sizeof needle, "\"%s\"", key);
    const char *location = strstr(text, needle);
    if (!location || !(location = strchr(location + strlen(needle), ':'))) return -1;
    char *end;
    *value = strtoull(location + 1, &end, 10);
    return end == location + 1 || *value == 0 ? -1 : 0;
}

int kmeans_metadata_dimensions(const char *metadata, uint64_t *rows_out, int *dimensions_out) {
    FILE *meta = fopen(metadata, "rb");
    if (!meta) {
        perror(metadata);
        return -1;
    }
    char text[8192];
    size_t length = fread(text, 1, sizeof text - 1, meta);
    int too_large = !feof(meta);
    fclose(meta);
    text[length] = '\0';
    uint64_t rows, dimensions;
    if (too_large || metadata_number(text, "rows", &rows) ||
        metadata_number(text, "features", &dimensions) || dimensions != 28 ||
        !strstr(text, "\"float32\"") || !strstr(text, "\"little-endian\"") ||
        rows > SIZE_MAX / (dimensions * sizeof(float))) {
        fprintf(stderr, "error: invalid or unsupported matrix metadata\n");
        return -1;
    }
    *rows_out = rows;
    *dimensions_out = (int)dimensions;
    return 0;
}

int kmeans_load_matrix(const char *binary, const char *metadata, KMeansMatrix *matrix) {
    memset(matrix, 0, sizeof *matrix);
    uint64_t rows;
    int dimensions;
    if (kmeans_metadata_dimensions(metadata, &rows, &dimensions)) {
        return -1;
    }
    struct stat info;
    size_t bytes = (size_t)rows * (size_t)dimensions * sizeof(float);
    if (stat(binary, &info) || info.st_size < 0 || (uint64_t)info.st_size != bytes) {
        fprintf(stderr, "error: matrix size differs from metadata\n");
        return -1;
    }
    FILE *stream = fopen(binary, "rb");
    if (!stream) {
        perror(binary);
        return -1;
    }
    float *points = malloc(bytes);
    if (!points || fread(points, 1, bytes, stream) != bytes) {
        fprintf(stderr, "error: cannot load matrix\n");
        free(points);
        fclose(stream);
        return -1;
    }
    fclose(stream);
    matrix->points = points;
    matrix->rows = rows;
    matrix->dimensions = (int)dimensions;
    return 0;
}

void kmeans_free_matrix(KMeansMatrix *matrix) {
    free(matrix->points);
    memset(matrix, 0, sizeof *matrix);
}
