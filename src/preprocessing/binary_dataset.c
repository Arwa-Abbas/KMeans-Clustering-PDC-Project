#define _POSIX_C_SOURCE 200809L

#include "kmeans/binary_dataset.h"

#include <errno.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define OUTPUT_BUFFER_BYTES (4U * 1024U * 1024U)

static int host_is_little_endian(void) {
    const uint16_t value = 1;
    return *(const unsigned char *)&value == 1;
}

static int make_parent_directories(const char *path) {
    char *copy = strdup(path);
    if (copy == NULL) return -1;
    for (char *cursor = copy + 1; *cursor != '\0'; ++cursor) {
        if (*cursor == '/') {
            *cursor = '\0';
            if (mkdir(copy, 0755) != 0 && errno != EEXIST) {
                free(copy);
                return -1;
            }
            *cursor = '/';
        }
    }
    free(copy);
    return 0;
}

static char *copy_with_suffix(const char *path, const char *suffix) {
    size_t bytes = strlen(path) + strlen(suffix) + 1;
    char *copy = malloc(bytes);
    if (copy != NULL) snprintf(copy, bytes, "%s%s", path, suffix);
    return copy;
}

static void writer_release(BinaryDatasetWriter *writer) {
    free(writer->final_path);
    free(writer->partial_path);
    writer->final_path = NULL;
    writer->partial_path = NULL;
}

int binary_dataset_writer_open(BinaryDatasetWriter *writer, const char *path,
                               bool overwrite) {
    memset(writer, 0, sizeof(*writer));
    if (!host_is_little_endian()) {
        fprintf(stderr, "error: binary output currently requires little-endian hardware\n");
        return -1;
    }
    writer->final_path = strdup(path);
    writer->partial_path = copy_with_suffix(path, ".part");
    if (writer->final_path == NULL || writer->partial_path == NULL ||
        make_parent_directories(path) != 0) {
        writer_release(writer);
        return -1;
    }
    if (!overwrite &&
        (access(writer->final_path, F_OK) == 0 ||
         access(writer->partial_path, F_OK) == 0)) {
        fprintf(stderr, "error: binary output exists; use --overwrite\n");
        writer_release(writer);
        return -1;
    }
    if (overwrite) {
        unlink(writer->final_path);
        unlink(writer->partial_path);
    }
    writer->stream = fopen(writer->partial_path, "wb");
    if (writer->stream == NULL) {
        writer_release(writer);
        return -1;
    }
    if (setvbuf(writer->stream, NULL, _IOFBF, OUTPUT_BUFFER_BYTES) != 0) {
        fprintf(stderr, "warning: could not configure the large output buffer\n");
    }
    return 0;
}

int binary_dataset_writer_write(
    BinaryDatasetWriter *writer,
    const float normalized_features[HIGGS_FEATURE_COLUMNS]) {
    size_t written = fwrite(normalized_features, sizeof(float),
                            HIGGS_FEATURE_COLUMNS, writer->stream);
    if (written != HIGGS_FEATURE_COLUMNS) return -1;
    ++writer->rows_written;
    return 0;
}

int binary_dataset_writer_finish(BinaryDatasetWriter *writer) {
    if (fclose(writer->stream) != 0) {
        writer->stream = NULL;
        return -1;
    }
    writer->stream = NULL;
    if (rename(writer->partial_path, writer->final_path) != 0) return -1;
    writer_release(writer);
    return 0;
}

void binary_dataset_writer_abort(BinaryDatasetWriter *writer) {
    if (writer->stream != NULL) fclose(writer->stream);
    writer->stream = NULL;
    writer_release(writer);
}

