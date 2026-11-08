#include "kmeans/preprocess.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void usage(const char *program) {
    fprintf(stderr,
            "Usage: %s [--archive PATH] [--output PATH] [--metadata PATH]\n"
            "          [--max-rows N] [--overwrite]\n",
            program);
}

static int parse_count(const char *text, unsigned long long *value) {
    char *end = NULL;
    errno = 0;
    unsigned long long parsed = strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || parsed == 0) return -1;
    *value = parsed;
    return 0;
}

int main(int argc, char **argv) {
    PreprocessOptions options = {
        .archive_path = "dataset/HIGGS.csv.gz",
        .binary_path = "dataset/processed/higgs_f32.bin",
        .metadata_path = "dataset/processed/higgs_f32.metadata.json",
        .max_rows = 0,
        .overwrite = false,
    };
    for (int index = 1; index < argc; ++index) {
        if (strcmp(argv[index], "--archive") == 0 && index + 1 < argc) {
            options.archive_path = argv[++index];
        } else if (strcmp(argv[index], "--output") == 0 && index + 1 < argc) {
            options.binary_path = argv[++index];
        } else if (strcmp(argv[index], "--metadata") == 0 && index + 1 < argc) {
            options.metadata_path = argv[++index];
        } else if (strcmp(argv[index], "--max-rows") == 0 && index + 1 < argc) {
            if (parse_count(argv[++index], &options.max_rows) != 0) {
                usage(argv[0]);
                return 2;
            }
        } else if (strcmp(argv[index], "--overwrite") == 0) {
            options.overwrite = true;
        } else {
            usage(argv[0]);
            return 2;
        }
    }
    return preprocess_higgs(&options);
}

