#include "kmeans/higgs_dataset.hpp"

#include "kmeans/higgs_reader.hpp"

#include <array>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

namespace fs = std::filesystem;

namespace kmeans {
namespace {

bool is_digit(char c) { return c >= '0' && c <= '9'; }

bool is_numeric_token(const char* cursor, const char* end) {
    if (cursor < end && (*cursor == '+' || *cursor == '-')) ++cursor;
    bool has_digits = false;
    while (cursor < end && is_digit(*cursor)) {
        has_digits = true;
        ++cursor;
    }
    if (cursor < end && *cursor == '.') {
        ++cursor;
        while (cursor < end && is_digit(*cursor)) {
            has_digits = true;
            ++cursor;
        }
    }
    if (!has_digits) return false;
    if (cursor < end && (*cursor == 'e' || *cursor == 'E')) {
        ++cursor;
        if (cursor < end && (*cursor == '+' || *cursor == '-')) ++cursor;
        const char* exponent = cursor;
        while (cursor < end && is_digit(*cursor)) ++cursor;
        if (cursor == exponent) return false;
    }
    return cursor == end;
}

bool validate_row(const char* line, std::uint64_t row, int& label) {
    const char* cursor = line;
    for (int column = 0; column < kHiggsColumns; ++column) {
        const char* end = cursor;
        while (*end != '\0' && *end != ',' && *end != '\r' && *end != '\n') ++end;
        if (!is_numeric_token(cursor, end)) {
            std::cerr << "error: row " << row << ", column " << column + 1
                      << " is not a finite number\n";
            return false;
        }
        if (column == 0) {
            char* parsed_end = nullptr;
            const double value = std::strtod(cursor, &parsed_end);
            if (parsed_end != end || (value != 0.0 && value != 1.0)) {
                std::cerr << "error: row " << row << " label must be 0 or 1\n";
                return false;
            }
            label = static_cast<int>(value);
        }
        if (column < kHiggsColumns - 1) {
            if (*end != ',') {
                std::cerr << "error: row " << row << " has fewer than " << kHiggsColumns
                          << " comma-separated columns\n";
                return false;
            }
            cursor = end + 1;
        } else {
            while (*end == '\r' || *end == '\n') ++end;
            if (*end != '\0') {
                std::cerr << "error: row " << row << " has more than " << kHiggsColumns
                          << " columns or trailing data\n";
                return false;
            }
        }
    }
    return true;
}

// Returns 1 for a valid row, 0 at the end of the archive, -1 on error.
int read_checked_line(gzFile input, std::vector<char>& line, std::uint64_t row, int& label) {
    if (gzgets(input, line.data(), static_cast<int>(line.size())) == nullptr) {
        if (gzeof(input)) return 0;
        int code = Z_OK;
        std::cerr << "error: cannot read gzip archive: " << gzerror(input, &code) << '\n';
        return -1;
    }
    const std::size_t length = std::strlen(line.data());
    if (length == 0 || (line[length - 1] != '\n' && !gzeof(input))) {
        std::cerr << "error: row " << row << " exceeds the " << line.size() - 1
                  << "-byte line limit\n";
        return -1;
    }
    return validate_row(line.data(), row, label) ? 1 : -1;
}

double cpu_seconds_since(std::clock_t started) {
    return static_cast<double>(std::clock() - started) / CLOCKS_PER_SEC;
}

bool make_parent_directories(const std::string& path) {
    const fs::path parent = fs::path(path).parent_path();
    if (parent.empty()) return true;
    std::error_code error;
    fs::create_directories(parent, error);
    if (error) {
        std::cerr << "error: cannot create directory " << parent.string() << ": "
                  << error.message() << '\n';
        return false;
    }
    return true;
}

std::string metadata_path_for(const std::string& output) {
    const std::string suffix = ".csv";
    const bool has_suffix = output.size() >= suffix.size() &&
                            output.compare(output.size() - suffix.size(), suffix.size(), suffix) == 0;
    return (has_suffix ? output.substr(0, output.size() - suffix.size()) : output) +
           ".metadata.json";
}

bool remove_if_requested(const std::string& path, bool overwrite) {
    std::error_code error;
    if (!fs::exists(path, error)) return true;
    if (!overwrite) {
        std::cerr << "error: " << path << " already exists; use --overwrite\n";
        return false;
    }
    if (!fs::remove(path, error)) {
        std::cerr << "error: cannot remove " << path << ": " << error.message() << '\n';
        return false;
    }
    return true;
}

}  // namespace

int inspect_archive(const HiggsDatasetOptions& options) {
    GzipFile input(gzopen(options.archive_path.c_str(), "rb"));
    if (!input) {
        std::cerr << "error: cannot open " << options.archive_path << '\n';
        return 1;
    }
    std::vector<char> line(kHiggsLineCapacity);
    std::uint64_t rows = 0;
    std::array<std::uint64_t, 2> labels{};
    int status = 0;
    const std::clock_t started = std::clock();
    while (options.full_scan || rows < options.sample_rows) {
        int label = 0;
        const int result = read_checked_line(input.get(), line, rows + 1, label);
        if (result == 0) break;
        if (result < 0) {
            status = 1;
            break;
        }
        ++rows;
        ++labels[label];
        if (options.full_scan && rows % 1'000'000 == 0)
            std::cerr << "validated " << rows << " rows\n";
    }
    if (status == 0 && options.full_scan && rows != options.expected_rows) {
        std::cerr << "error: expected " << options.expected_rows << " rows, found " << rows << '\n';
        status = 1;
    }
    if (status == 0) {
        std::cout << "{\n"
                  << "  \"archive\": \"" << options.archive_path << "\",\n"
                  << "  \"scan\": \"" << (options.full_scan ? "full" : "sample") << "\",\n"
                  << "  \"rows_checked\": " << rows << ",\n"
                  << "  \"columns\": " << kHiggsColumns << ",\n"
                  << "  \"feature_columns\": " << kHiggsFeatures << ",\n"
                  << "  \"labels_0\": " << labels[0] << ",\n"
                  << "  \"labels_1\": " << labels[1] << ",\n"
                  << "  \"cpu_seconds\": " << std::fixed << std::setprecision(3)
                  << cpu_seconds_since(started) << ",\n"
                  << "  \"status\": \"valid\"\n"
                  << "}\n";
    }
    return status;
}

int extract_archive(const HiggsDatasetOptions& options) {
    const std::string partial = options.output_path + ".part";
    const std::string metadata = metadata_path_for(options.output_path);
    if (!make_parent_directories(options.output_path) ||
        !remove_if_requested(options.output_path, options.overwrite) ||
        !remove_if_requested(partial, options.overwrite) ||
        !remove_if_requested(metadata, options.overwrite)) {
        return 1;
    }

    GzipFile input(gzopen(options.archive_path.c_str(), "rb"));
    std::ofstream output(partial, std::ios::binary);
    if (!input || !output) {
        std::cerr << "error: cannot open " << options.archive_path << " or " << partial << '\n';
        return 1;
    }

    std::vector<char> line(kHiggsLineCapacity);
    std::uint64_t rows = 0;
    int status = 0;
    const std::clock_t started = std::clock();
    for (;;) {
        int label = 0;
        const int result = read_checked_line(input.get(), line, rows + 1, label);
        if (result == 0) break;
        if (result < 0) {
            status = 1;
            break;
        }
        if (!(output << line.data())) {
            std::cerr << "error: cannot write " << partial << '\n';
            status = 1;
            break;
        }
        ++rows;
        if (rows % 1'000'000 == 0) std::cerr << "extracted and validated " << rows << " rows\n";
    }
    output.close();
    if (!output) status = 1;

    if (status == 0 && rows != options.expected_rows) {
        std::cerr << "error: expected " << options.expected_rows << " rows, found " << rows
                  << "; kept " << partial << '\n';
        status = 1;
    }
    std::error_code error;
    if (status == 0) {
        fs::rename(partial, options.output_path, error);
        if (error) {
            std::cerr << "error: cannot rename " << partial << " to " << options.output_path
                      << ": " << error.message() << '\n';
            status = 1;
        }
    }
    if (status == 0) {
        std::ofstream meta(metadata);
        meta << "{\n  \"source_archive\": \"" << options.archive_path << "\",\n"
             << "  \"output_csv\": \"" << options.output_path << "\",\n"
             << "  \"rows\": " << rows << ",\n"
             << "  \"columns\": " << kHiggsColumns << ",\n  \"label_column\": 0,\n"
             << "  \"feature_columns\": " << kHiggsFeatures << ",\n"
             << "  \"cpu_seconds\": " << std::fixed << std::setprecision(3)
             << cpu_seconds_since(started) << "\n}\n";
        meta.close();
        if (!meta) {
            std::cerr << "error: cannot write " << metadata << '\n';
            status = 1;
        } else {
            std::cout << "extracted " << rows << " validated rows to " << options.output_path
                      << "\nmetadata: " << metadata << '\n';
        }
    }
    return status;
}

}  // namespace kmeans
