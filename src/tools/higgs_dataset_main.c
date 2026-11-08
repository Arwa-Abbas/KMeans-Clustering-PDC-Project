#include "kmeans/higgs_dataset.h"

#include <errno.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void usage(const char *program) {
    fprintf(stderr,
            "Usage:\n"
            "  %s inspect [--archive PATH] [--sample-rows N] [--full-scan]\n"
            "  %s extract [--archive PATH] [--output PATH] "
            "[--expected-rows N] [--overwrite]\n",
            program, program);
}

static bool parse_positive_ull(const char *text, unsigned long long *value) {
    char *end = NULL;
    errno = 0;
    unsigned long long parsed = strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || parsed == 0) return false;
    *value = parsed;
    return true;
}

static int parse_options(int argc, char **argv, HiggsDatasetOptions *options) {
    for (int index = 2; index < argc; ++index) {
        const char *argument = argv[index];
        if (strcmp(argument, "--archive") == 0 && index + 1 < argc) {
            options->archive_path = argv[++index];
        } else if (strcmp(argument, "--output") == 0 && index + 1 < argc) {
            options->output_path = argv[++index];
        } else if (strcmp(argument, "--sample-rows") == 0 && index + 1 < argc) {
            if (!parse_positive_ull(argv[++index], &options->sample_rows)) return -1;
        } else if (strcmp(argument, "--expected-rows") == 0 && index + 1 < argc) {
            if (!parse_positive_ull(argv[++index], &options->expected_rows)) return -1;
        } else if (strcmp(argument, "--full-scan") == 0) {
            options->full_scan = true;
        } else if (strcmp(argument, "--overwrite") == 0) {
            options->overwrite = true;
        } else {
            fprintf(stderr, "error: invalid or incomplete option: %s\n", argument);
            return -1;
        }
    }
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        usage(argv[0]);
        return 2;
    }

    HiggsDatasetOptions options = {
        .archive_path = "dataset/HIGGS.csv.gz",
        .output_path = "dataset/raw/HIGGS.csv",
        .sample_rows = 1000,
        .expected_rows = HIGGS_EXPECTED_ROWS,
        .full_scan = false,
        .overwrite = false,
    };
    if (parse_options(argc, argv, &options) != 0) {
        usage(argv[0]);
        return 2;
    }
    if (strcmp(argv[1], "inspect") == 0) {
        return higgs_inspect_archive(&options);
    }
    if (strcmp(argv[1], "extract") == 0) {
        return higgs_extract_archive(&options);
    }
    usage(argv[0]);
    return 2;
}

