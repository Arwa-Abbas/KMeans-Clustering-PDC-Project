#include "kmeans/higgs_reader.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <zlib.h>

#define HIGGS_LINE_CAPACITY 32768
#define ERROR_CAPACITY 256

struct HiggsReader {
    gzFile stream;
    unsigned long long row;
    char line[HIGGS_LINE_CAPACITY];
    char error[ERROR_CAPACITY];
};

HiggsReader *higgs_reader_open(const char *path) {
    HiggsReader *reader = calloc(1, sizeof(*reader));
    if (reader == NULL) return NULL;
    reader->stream = gzopen(path, "rb");
    if (reader->stream == NULL) {
        free(reader);
        return NULL;
    }
    return reader;
}

static int parse_row(HiggsReader *reader,
                     float features[HIGGS_FEATURE_COLUMNS], int *label) {
    const char *cursor = reader->line;
    for (int column = 0; column < HIGGS_EXPECTED_COLUMNS; ++column) {
        char *end = NULL;
        float value = strtof(cursor, &end);
        if (end == cursor || !isfinite(value)) {
            snprintf(reader->error, sizeof(reader->error),
                     "row %llu column %d is not a finite number", reader->row,
                     column + 1);
            return -1;
        }

        if (column == 0) {
            if (value != 0.0F && value != 1.0F) {
                snprintf(reader->error, sizeof(reader->error),
                         "row %llu label is not 0 or 1", reader->row);
                return -1;
            }
            *label = (int)value;
        } else {
            features[column - 1] = value;
        }

        if (column < HIGGS_EXPECTED_COLUMNS - 1) {
            if (*end != ',') {
                snprintf(reader->error, sizeof(reader->error),
                         "row %llu does not contain %d columns", reader->row,
                         HIGGS_EXPECTED_COLUMNS);
                return -1;
            }
            cursor = end + 1;
        } else {
            while (*end == '\r' || *end == '\n') ++end;
            if (*end != '\0') {
                snprintf(reader->error, sizeof(reader->error),
                         "row %llu contains trailing data", reader->row);
                return -1;
            }
        }
    }
    return 1;
}

int higgs_reader_next(HiggsReader *reader, float features[HIGGS_FEATURE_COLUMNS],
                      int *label) {
    if (gzgets(reader->stream, reader->line, sizeof(reader->line)) == NULL) {
        if (gzeof(reader->stream)) return 0;
        int error_code = Z_OK;
        const char *message = gzerror(reader->stream, &error_code);
        snprintf(reader->error, sizeof(reader->error), "gzip read failed: %s",
                 message);
        return -1;
    }
    ++reader->row;
    size_t length = strlen(reader->line);
    if (length == 0 ||
        (reader->line[length - 1] != '\n' && !gzeof(reader->stream))) {
        snprintf(reader->error, sizeof(reader->error),
                 "row %llu exceeds the line buffer", reader->row);
        return -1;
    }
    return parse_row(reader, features, label);
}

unsigned long long higgs_reader_row(const HiggsReader *reader) {
    return reader->row;
}

const char *higgs_reader_error(const HiggsReader *reader) {
    return reader->error;
}

void higgs_reader_close(HiggsReader *reader) {
    if (reader == NULL) return;
    if (reader->stream != NULL) gzclose(reader->stream);
    free(reader);
}

