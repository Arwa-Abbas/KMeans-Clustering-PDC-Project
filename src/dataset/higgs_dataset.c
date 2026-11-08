#define _POSIX_C_SOURCE 200809L

#include "kmeans/higgs_dataset.h"

#include <errno.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#include <zlib.h>

#define LINE_CAPACITY 32768

static bool is_numeric_token(const char *start, const char *end) {
    const char *cursor = start;
    if (cursor < end && (*cursor == '+' || *cursor == '-')) ++cursor;

    bool has_digits = false;
    while (cursor < end && *cursor >= '0' && *cursor <= '9') {
        has_digits = true;
        ++cursor;
    }
    if (cursor < end && *cursor == '.') {
        ++cursor;
        while (cursor < end && *cursor >= '0' && *cursor <= '9') {
            has_digits = true;
            ++cursor;
        }
    }
    if (!has_digits) return false;

    if (cursor < end && (*cursor == 'e' || *cursor == 'E')) {
        ++cursor;
        if (cursor < end && (*cursor == '+' || *cursor == '-')) ++cursor;
        const char *exponent_start = cursor;
        while (cursor < end && *cursor >= '0' && *cursor <= '9') ++cursor;
        if (cursor == exponent_start) return false;
    }
    return cursor == end;
}

static int validate_row(const char *line, unsigned long long row_number,
                        int *label_out) {
    const char *cursor = line;
    for (int column = 0; column < HIGGS_EXPECTED_COLUMNS; ++column) {
        const char *end = cursor;
        while (*end != '\0' && *end != ',' && *end != '\r' && *end != '\n') ++end;

        if (!is_numeric_token(cursor, end)) {
            fprintf(stderr, "error: row %llu, column %d is not a finite number\n",
                    row_number, column + 1);
            return -1;
        }
        if (column == 0) {
            char *parsed_end = NULL;
            double value = strtod(cursor, &parsed_end);
            if (parsed_end != end || (value != 0.0 && value != 1.0)) {
                fprintf(stderr, "error: row %llu label must be 0 or 1\n", row_number);
                return -1;
            }
            *label_out = (int)value;
        }

        if (column < HIGGS_EXPECTED_COLUMNS - 1) {
            if (*end != ',') {
                fprintf(stderr,
                        "error: row %llu has fewer than %d comma-separated columns\n",
                        row_number, HIGGS_EXPECTED_COLUMNS);
                return -1;
            }
            cursor = end + 1;
        } else {
            while (*end == '\r' || *end == '\n') ++end;
            if (*end != '\0') {
                fprintf(stderr,
                        "error: row %llu has more than %d columns or trailing data\n",
                        row_number, HIGGS_EXPECTED_COLUMNS);
                return -1;
            }
        }
    }
    return 0;
}

static int read_checked_line(gzFile input, char *line, size_t capacity,
                             unsigned long long row_number, int *label_out) {
    if (gzgets(input, line, (int)capacity) == NULL) {
        if (gzeof(input)) return 0;
        int zlib_error = Z_OK;
        const char *message = gzerror(input, &zlib_error);
        fprintf(stderr, "error: cannot read gzip archive: %s\n", message);
        return -1;
    }

    size_t length = strlen(line);
    if (length == 0 || (line[length - 1] != '\n' && !gzeof(input))) {
        fprintf(stderr, "error: row %llu exceeds the %d-byte line limit\n",
                row_number, LINE_CAPACITY - 1);
        return -1;
    }
    return validate_row(line, row_number, label_out) == 0 ? 1 : -1;
}

int higgs_inspect_archive(const HiggsDatasetOptions *options) {
    gzFile input = gzopen(options->archive_path, "rb");
    if (input == NULL) {
        fprintf(stderr, "error: cannot open %s\n", options->archive_path);
        return 1;
    }

    char *line = malloc(LINE_CAPACITY);
    if (line == NULL) {
        fprintf(stderr, "error: cannot allocate line buffer\n");
        gzclose(input);
        return 1;
    }

    unsigned long long rows = 0;
    unsigned long long labels[2] = {0, 0};
    int status = 0;
    clock_t started = clock();
    while (options->full_scan || rows < options->sample_rows) {
        int label = 0;
        int result = read_checked_line(input, line, LINE_CAPACITY, rows + 1, &label);
        if (result == 0) break;
        if (result < 0) {
            status = 1;
            break;
        }
        ++rows;
        ++labels[label];
        if (options->full_scan && rows % 1000000ULL == 0) {
            fprintf(stderr, "validated %llu rows\n", rows);
        }
    }

    if (status == 0 && options->full_scan && rows != options->expected_rows) {
        fprintf(stderr, "error: expected %llu rows, found %llu\n",
                options->expected_rows, rows);
        status = 1;
    }
    if (status == 0) {
        double elapsed = (double)(clock() - started) / CLOCKS_PER_SEC;
        printf("{\n"
               "  \"archive\": \"%s\",\n"
               "  \"scan\": \"%s\",\n"
               "  \"rows_checked\": %llu,\n"
               "  \"columns\": %d,\n"
               "  \"feature_columns\": %d,\n"
               "  \"labels_0\": %llu,\n"
               "  \"labels_1\": %llu,\n"
               "  \"cpu_seconds\": %.3f,\n"
               "  \"status\": \"valid\"\n"
               "}\n",
               options->archive_path, options->full_scan ? "full" : "sample", rows,
               HIGGS_EXPECTED_COLUMNS, HIGGS_FEATURE_COLUMNS, labels[0], labels[1],
               elapsed);
    }

    free(line);
    gzclose(input);
    return status;
}

static int make_parent_directories(const char *path) {
    char *copy = strdup(path);
    if (copy == NULL) return -1;
    for (char *cursor = copy + 1; *cursor != '\0'; ++cursor) {
        if (*cursor == '/') {
            *cursor = '\0';
            if (mkdir(copy, 0755) != 0 && errno != EEXIST) {
                fprintf(stderr, "error: cannot create directory %s: %s\n",
                        copy, strerror(errno));
                free(copy);
                return -1;
            }
            *cursor = '/';
        }
    }
    free(copy);
    return 0;
}

static char *append_suffix(const char *path, const char *suffix) {
    size_t length = strlen(path) + strlen(suffix) + 1;
    char *result = malloc(length);
    if (result != NULL) snprintf(result, length, "%s%s", path, suffix);
    return result;
}

static char *metadata_path_for(const char *output) {
    size_t length = strlen(output);
    size_t stem_length = length >= 4 && strcmp(output + length - 4, ".csv") == 0
                             ? length - 4
                             : length;
    const char *suffix = ".metadata.json";
    char *result = malloc(stem_length + strlen(suffix) + 1);
    if (result != NULL) {
        memcpy(result, output, stem_length);
        strcpy(result + stem_length, suffix);
    }
    return result;
}

static int remove_if_requested(const char *path, bool overwrite) {
    if (access(path, F_OK) != 0) return 0;
    if (!overwrite) {
        fprintf(stderr, "error: %s already exists; use --overwrite\n", path);
        return -1;
    }
    if (unlink(path) != 0) {
        fprintf(stderr, "error: cannot remove %s: %s\n", path, strerror(errno));
        return -1;
    }
    return 0;
}

int higgs_extract_archive(const HiggsDatasetOptions *options) {
    char *partial = append_suffix(options->output_path, ".part");
    char *metadata = metadata_path_for(options->output_path);
    char *line = malloc(LINE_CAPACITY);
    if (partial == NULL || metadata == NULL || line == NULL) {
        fprintf(stderr, "error: cannot allocate extraction buffers\n");
        free(partial);
        free(metadata);
        free(line);
        return 1;
    }
    if (make_parent_directories(options->output_path) != 0 ||
        remove_if_requested(options->output_path, options->overwrite) != 0 ||
        remove_if_requested(partial, options->overwrite) != 0 ||
        remove_if_requested(metadata, options->overwrite) != 0) {
        free(partial);
        free(metadata);
        free(line);
        return 1;
    }

    gzFile input = gzopen(options->archive_path, "rb");
    FILE *output = fopen(partial, "wb");
    if (input == NULL || output == NULL) {
        fprintf(stderr, "error: cannot open archive or output: %s\n", strerror(errno));
        if (input != NULL) gzclose(input);
        if (output != NULL) fclose(output);
        free(partial);
        free(metadata);
        free(line);
        return 1;
    }

    unsigned long long rows = 0;
    int status = 0;
    clock_t started = clock();
    for (;;) {
        int label = 0;
        int result = read_checked_line(input, line, LINE_CAPACITY, rows + 1, &label);
        if (result == 0) break;
        if (result < 0 || fputs(line, output) == EOF) {
            if (result >= 0) fprintf(stderr, "error: cannot write %s\n", partial);
            status = 1;
            break;
        }
        ++rows;
        if (rows % 1000000ULL == 0) {
            fprintf(stderr, "extracted and validated %llu rows\n", rows);
        }
    }
    if (fclose(output) != 0) status = 1;
    gzclose(input);

    if (status == 0 && rows != options->expected_rows) {
        fprintf(stderr, "error: expected %llu rows, found %llu; kept %s\n",
                options->expected_rows, rows, partial);
        status = 1;
    }
    if (status == 0 && rename(partial, options->output_path) != 0) {
        fprintf(stderr, "error: cannot rename %s to %s: %s\n", partial,
                options->output_path, strerror(errno));
        status = 1;
    }
    if (status == 0) {
        FILE *meta = fopen(metadata, "w");
        if (meta == NULL) {
            fprintf(stderr, "error: cannot write %s\n", metadata);
            status = 1;
        } else {
            double elapsed = (double)(clock() - started) / CLOCKS_PER_SEC;
            fprintf(meta,
                    "{\n  \"source_archive\": \"%s\",\n"
                    "  \"output_csv\": \"%s\",\n  \"rows\": %llu,\n"
                    "  \"columns\": %d,\n  \"label_column\": 0,\n"
                    "  \"feature_columns\": %d,\n  \"cpu_seconds\": %.3f\n}\n",
                    options->archive_path, options->output_path, rows,
                    HIGGS_EXPECTED_COLUMNS, HIGGS_FEATURE_COLUMNS, elapsed);
            fclose(meta);
            printf("extracted %llu validated rows to %s\nmetadata: %s\n", rows,
                   options->output_path, metadata);
        }
    }

    free(partial);
    free(metadata);
    free(line);
    return status;
}

