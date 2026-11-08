#define _POSIX_C_SOURCE 200809L

#include "kmeans/preprocess.h"

#include "kmeans/binary_dataset.h"
#include "kmeans/higgs_dataset.h"
#include "kmeans/higgs_reader.h"
#include "kmeans/online_stats.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define VERIFY_ROWS 1000ULL

static int collect_statistics(const PreprocessOptions *options,
                              OnlineStats *stats) {
    /* First pass: keep only 28 running means and variances in memory. */
    HiggsReader *reader = higgs_reader_open(options->archive_path);
    if (reader == NULL) {
        fprintf(stderr, "error: cannot open %s\n", options->archive_path);
        return -1;
    }
    float features[HIGGS_FEATURE_COLUMNS];
    int label = 0;
    int result = 0;
    while ((result = higgs_reader_next(reader, features, &label)) > 0) {
        online_stats_update(stats, features);
        if (stats->count % 1000000ULL == 0) {
            fprintf(stderr, "statistics: %llu rows\n", stats->count);
        }
        if (options->max_rows != 0 && stats->count == options->max_rows) break;
    }
    if (result < 0) {
        fprintf(stderr, "error: %s\n", higgs_reader_error(reader));
        higgs_reader_close(reader);
        return -1;
    }
    higgs_reader_close(reader);
    return 0;
}

static int write_normalized_binary(const PreprocessOptions *options,
                                   const OnlineStats *stats,
                                   const double stddev[HIGGS_FEATURE_COLUMNS]) {
    /* Second pass: write features in row-major order, without the label. */
    HiggsReader *reader = higgs_reader_open(options->archive_path);
    BinaryDatasetWriter writer;
    if (reader == NULL ||
        binary_dataset_writer_open(&writer, options->binary_path,
                                   options->overwrite) != 0) {
        fprintf(stderr, "error: cannot open normalization input or output\n");
        higgs_reader_close(reader);
        return -1;
    }

    float features[HIGGS_FEATURE_COLUMNS];
    float normalized[HIGGS_FEATURE_COLUMNS];
    int label = 0;
    int result = 0;
    while ((result = higgs_reader_next(reader, features, &label)) > 0) {
        for (int feature = 0; feature < HIGGS_FEATURE_COLUMNS; ++feature) {
            normalized[feature] =
                (float)(((double)features[feature] - stats->mean[feature]) /
                        stddev[feature]);
        }
        if (binary_dataset_writer_write(&writer, normalized) != 0) {
            fprintf(stderr, "error: failed writing normalized row\n");
            binary_dataset_writer_abort(&writer);
            higgs_reader_close(reader);
            return -1;
        }
        if (writer.rows_written % 1000000ULL == 0) {
            fprintf(stderr, "binary output: %llu rows\n", writer.rows_written);
        }
        if (writer.rows_written == stats->count) break;
    }
    if (result < 0 || writer.rows_written != stats->count) {
        fprintf(stderr, "error: normalization pass ended after %llu of %llu rows\n",
                writer.rows_written, stats->count);
        binary_dataset_writer_abort(&writer);
        higgs_reader_close(reader);
        return -1;
    }
    higgs_reader_close(reader);
    return binary_dataset_writer_finish(&writer);
}

static int verify_binary(const PreprocessOptions *options,
                         const OnlineStats *stats,
                         const double stddev[HIGGS_FEATURE_COLUMNS]) {
    /* Check both byte count and values read back from the finished file. */
    struct stat info;
    unsigned long long expected_bytes =
        stats->count * HIGGS_FEATURE_COLUMNS * sizeof(float);
    if (stat(options->binary_path, &info) != 0 ||
        (unsigned long long)info.st_size != expected_bytes) {
        fprintf(stderr, "error: binary file size does not match its dimensions\n");
        return -1;
    }

    HiggsReader *reader = higgs_reader_open(options->archive_path);
    FILE *binary = fopen(options->binary_path, "rb");
    if (reader == NULL || binary == NULL) {
        higgs_reader_close(reader);
        if (binary != NULL) fclose(binary);
        return -1;
    }
    float source[HIGGS_FEATURE_COLUMNS];
    float stored[HIGGS_FEATURE_COLUMNS];
    int label = 0;
    unsigned long long rows_to_check = stats->count < VERIFY_ROWS
                                           ? stats->count
                                           : VERIFY_ROWS;
    for (unsigned long long row = 0; row < rows_to_check; ++row) {
        if (higgs_reader_next(reader, source, &label) <= 0 ||
            fread(stored, sizeof(float), HIGGS_FEATURE_COLUMNS, binary) !=
                HIGGS_FEATURE_COLUMNS) {
            higgs_reader_close(reader);
            fclose(binary);
            return -1;
        }
        for (int feature = 0; feature < HIGGS_FEATURE_COLUMNS; ++feature) {
            float expected =
                (float)(((double)source[feature] - stats->mean[feature]) /
                        stddev[feature]);
            if (fabsf(stored[feature] - expected) > 1e-6F) {
                fprintf(stderr, "error: verification failed at row %llu feature %d\n",
                        row + 1, feature + 1);
                higgs_reader_close(reader);
                fclose(binary);
                return -1;
            }
        }
    }
    higgs_reader_close(reader);
    fclose(binary);
    return 0;
}

static void write_array(FILE *stream, const double values[HIGGS_FEATURE_COLUMNS]) {
    fputs("[", stream);
    for (int feature = 0; feature < HIGGS_FEATURE_COLUMNS; ++feature) {
        fprintf(stream, "%s%.17g", feature == 0 ? "" : ", ", values[feature]);
    }
    fputs("]", stream);
}

static int write_metadata(const PreprocessOptions *options,
                          const OnlineStats *stats,
                          const double stddev[HIGGS_FEATURE_COLUMNS]) {
    if (!options->overwrite && access(options->metadata_path, F_OK) == 0) {
        fprintf(stderr, "error: metadata output exists; use --overwrite\n");
        return -1;
    }
    FILE *stream = fopen(options->metadata_path, "w");
    if (stream == NULL) return -1;
    fprintf(stream,
            "{\n  \"format_version\": 1,\n  \"source_archive\": \"%s\",\n"
            "  \"binary_file\": \"%s\",\n  \"rows\": %llu,\n"
            "  \"features\": %d,\n  \"data_type\": \"float32\",\n"
            "  \"byte_order\": \"little-endian\",\n"
            "  \"layout\": \"row-major, headerless\",\n"
            "  \"normalization\": \"z-score using population standard deviation\",\n"
            "  \"label_included\": false,\n  \"mean\": ",
            options->archive_path, options->binary_path, stats->count,
            HIGGS_FEATURE_COLUMNS);
    write_array(stream, stats->mean);
    fputs(",\n  \"standard_deviation\": ", stream);
    write_array(stream, stddev);
    fputs("\n}\n", stream);
    return fclose(stream) == 0 ? 0 : -1;
}

int preprocess_higgs(const PreprocessOptions *options) {
    OnlineStats stats;
    double stddev[HIGGS_FEATURE_COLUMNS];
    online_stats_init(&stats);
    if (collect_statistics(options, &stats) != 0 ||
        online_stats_population_stddev(&stats, stddev) != 0) {
        fprintf(stderr, "error: could not compute valid feature statistics\n");
        return 1;
    }
    if (options->max_rows == 0 && stats.count != HIGGS_EXPECTED_ROWS) {
        fprintf(stderr, "error: expected %llu rows, found %llu\n",
                HIGGS_EXPECTED_ROWS, stats.count);
        return 1;
    }
    if (write_normalized_binary(options, &stats, stddev) != 0 ||
        verify_binary(options, &stats, stddev) != 0 ||
        write_metadata(options, &stats, stddev) != 0) {
        return 1;
    }
    printf("preprocessed %llu rows x %d features\nbinary: %s\nmetadata: %s\n",
           stats.count, HIGGS_FEATURE_COLUMNS, options->binary_path,
           options->metadata_path);
    return 0;
}

