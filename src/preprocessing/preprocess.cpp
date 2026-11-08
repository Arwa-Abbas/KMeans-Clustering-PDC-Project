#include "kmeans/preprocess.hpp"

#include "kmeans/binary_dataset.hpp"
#include "kmeans/higgs_reader.hpp"
#include "kmeans/online_stats.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>

namespace fs = std::filesystem;

namespace kmeans {
namespace {

constexpr std::uint64_t kVerifyRows = 1000;
using Values = OnlineStats::Values;

FeatureRow normalize(const FeatureRow& row, const Values& mean, const Values& stddev) {
    FeatureRow normalized{};
    for (int f = 0; f < kHiggsFeatures; ++f)
        normalized[f] = static_cast<float>((static_cast<double>(row[f]) - mean[f]) / stddev[f]);
    return normalized;
}

// First pass: keep only 28 running means and variances in memory.
bool collect_statistics(const PreprocessOptions& options, OnlineStats& stats) {
    HiggsReader reader(options.archive_path);
    if (!reader.is_open()) {
        std::cerr << "error: cannot open " << options.archive_path << '\n';
        return false;
    }
    FeatureRow features{};
    int label = 0;
    int status = 0;
    while ((options.max_rows == 0 || stats.count() < options.max_rows) &&
           (status = reader.next(features, label)) > 0) {
        stats.update(features);
        if (stats.count() % 1'000'000 == 0) std::cerr << "statistics: " << stats.count() << " rows\n";
    }
    if (status < 0) {
        std::cerr << "error: " << reader.error() << '\n';
        return false;
    }
    return true;
}

// Second pass: write features in row-major order, without the label.
bool write_normalized_binary(const PreprocessOptions& options, const OnlineStats& stats,
                             const Values& stddev) {
    HiggsReader reader(options.archive_path);
    BinaryDatasetWriter writer;
    if (!reader.is_open() || !writer.open(options.binary_path, options.overwrite)) {
        std::cerr << "error: cannot open normalization input or output\n";
        return false;
    }
    FeatureRow features{};
    int label = 0;
    int status = 0;
    while (writer.rows_written() < stats.count() && (status = reader.next(features, label)) > 0) {
        if (!writer.write(normalize(features, stats.mean(), stddev))) {
            std::cerr << "error: failed writing normalized row\n";
            return false;
        }
        if (writer.rows_written() % 1'000'000 == 0)
            std::cerr << "binary output: " << writer.rows_written() << " rows\n";
    }
    if (status < 0 || writer.rows_written() != stats.count()) {
        std::cerr << "error: normalization pass ended after " << writer.rows_written() << " of "
                  << stats.count() << " rows\n";
        return false;
    }
    return writer.finish();
}

// Check both the byte count and values read back from the finished file.
bool verify_binary(const PreprocessOptions& options, const OnlineStats& stats,
                   const Values& stddev) {
    std::error_code error;
    const std::uintmax_t bytes = fs::file_size(options.binary_path, error);
    if (error || bytes != stats.count() * kHiggsFeatures * sizeof(float)) {
        std::cerr << "error: binary file size does not match its dimensions\n";
        return false;
    }
    HiggsReader reader(options.archive_path);
    std::ifstream binary(options.binary_path, std::ios::binary);
    if (!reader.is_open() || !binary) return false;

    FeatureRow source{};
    FeatureRow stored{};
    int label = 0;
    const std::uint64_t rows = std::min(stats.count(), kVerifyRows);
    for (std::uint64_t row = 0; row < rows; ++row) {
        if (reader.next(source, label) <= 0 ||
            !binary.read(reinterpret_cast<char*>(stored.data()),
                         static_cast<std::streamsize>(sizeof(float) * stored.size()))) {
            return false;
        }
        const FeatureRow expected = normalize(source, stats.mean(), stddev);
        for (int f = 0; f < kHiggsFeatures; ++f) {
            if (std::fabs(stored[f] - expected[f]) > 1e-6F) {
                std::cerr << "error: verification failed at row " << row + 1 << " feature "
                          << f + 1 << '\n';
                return false;
            }
        }
    }
    return true;
}

void write_array(std::ostream& out, const Values& values) {
    out << '[';
    for (int f = 0; f < kHiggsFeatures; ++f) out << (f ? ", " : "") << values[f];
    out << ']';
}

bool write_metadata(const PreprocessOptions& options, const OnlineStats& stats,
                    const Values& stddev) {
    std::error_code error;
    if (!options.overwrite && fs::exists(options.metadata_path, error)) {
        std::cerr << "error: metadata output exists; use --overwrite\n";
        return false;
    }
    std::ofstream out(options.metadata_path);
    if (!out) return false;
    out << std::setprecision(17)
        << "{\n  \"format_version\": 1,\n"
        << "  \"source_archive\": \"" << options.archive_path << "\",\n"
        << "  \"binary_file\": \"" << options.binary_path << "\",\n"
        << "  \"rows\": " << stats.count() << ",\n"
        << "  \"features\": " << kHiggsFeatures << ",\n"
        << "  \"data_type\": \"float32\",\n"
        << "  \"byte_order\": \"little-endian\",\n"
        << "  \"layout\": \"row-major, headerless\",\n"
        << "  \"normalization\": \"z-score using population standard deviation\",\n"
        << "  \"label_included\": false,\n  \"mean\": ";
    write_array(out, stats.mean());
    out << ",\n  \"standard_deviation\": ";
    write_array(out, stddev);
    out << "\n}\n";
    out.close();
    return static_cast<bool>(out);
}

}  // namespace

int preprocess_higgs(const PreprocessOptions& options) {
    OnlineStats stats;
    Values stddev{};
    if (!collect_statistics(options, stats) || !stats.population_stddev(stddev)) {
        std::cerr << "error: could not compute valid feature statistics\n";
        return 1;
    }
    if (options.max_rows == 0 && stats.count() != kHiggsRows) {
        std::cerr << "error: expected " << kHiggsRows << " rows, found " << stats.count() << '\n';
        return 1;
    }
    if (!write_normalized_binary(options, stats, stddev) ||
        !verify_binary(options, stats, stddev) || !write_metadata(options, stats, stddev)) {
        return 1;
    }
    std::cout << "preprocessed " << stats.count() << " rows x " << kHiggsFeatures
              << " features\nbinary: " << options.binary_path
              << "\nmetadata: " << options.metadata_path << '\n';
    return 0;
}

}  // namespace kmeans
