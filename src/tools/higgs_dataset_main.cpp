#include "kmeans/higgs_dataset.hpp"

#include <cerrno>
#include <cstdlib>
#include <iostream>
#include <string>

namespace {

void usage(const std::string& program) {
    std::cerr << "Usage:\n"
              << "  " << program << " inspect [--archive PATH] [--sample-rows N] [--full-scan]\n"
              << "  " << program << " extract [--archive PATH] [--output PATH] "
              << "[--expected-rows N] [--overwrite]\n";
}

bool parse_positive(const std::string& text, std::uint64_t& value) {
    char* end = nullptr;
    errno = 0;
    const unsigned long long parsed = std::strtoull(text.c_str(), &end, 10);
    if (text.empty() || text[0] == '-' || errno != 0 || end == text.c_str() || *end != '\0' ||
        parsed == 0) {
        return false;
    }
    value = parsed;
    return true;
}

bool parse_options(int argc, char** argv, kmeans::HiggsDatasetOptions& options) {
    for (int i = 2; i < argc; ++i) {
        const std::string argument = argv[i];
        const bool has_value = i + 1 < argc;
        if (argument == "--archive" && has_value) {
            options.archive_path = argv[++i];
        } else if (argument == "--output" && has_value) {
            options.output_path = argv[++i];
        } else if (argument == "--sample-rows" && has_value) {
            if (!parse_positive(argv[++i], options.sample_rows)) return false;
        } else if (argument == "--expected-rows" && has_value) {
            if (!parse_positive(argv[++i], options.expected_rows)) return false;
        } else if (argument == "--full-scan") {
            options.full_scan = true;
        } else if (argument == "--overwrite") {
            options.overwrite = true;
        } else {
            std::cerr << "error: invalid or incomplete option: " << argument << '\n';
            return false;
        }
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        usage(argv[0]);
        return 2;
    }
    kmeans::HiggsDatasetOptions options;
    if (!parse_options(argc, argv, options)) {
        usage(argv[0]);
        return 2;
    }
    const std::string command = argv[1];
    if (command == "inspect") return kmeans::inspect_archive(options);
    if (command == "extract") return kmeans::extract_archive(options);
    usage(argv[0]);
    return 2;
}
