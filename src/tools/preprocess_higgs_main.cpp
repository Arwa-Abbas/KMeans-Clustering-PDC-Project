#include "kmeans/preprocess.hpp"

#include <cerrno>
#include <cstdlib>
#include <iostream>
#include <string>

namespace {

void usage(const std::string& program) {
    std::cerr << "Usage: " << program << " [--archive PATH] [--output PATH] [--metadata PATH]\n"
              << "          [--max-rows N] [--overwrite]\n";
}

bool parse_count(const std::string& text, std::uint64_t& value) {
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

}  // namespace

int main(int argc, char** argv) {
    kmeans::PreprocessOptions options;
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        const bool has_value = i + 1 < argc;
        if (argument == "--archive" && has_value) {
            options.archive_path = argv[++i];
        } else if (argument == "--output" && has_value) {
            options.binary_path = argv[++i];
        } else if (argument == "--metadata" && has_value) {
            options.metadata_path = argv[++i];
        } else if (argument == "--max-rows" && has_value) {
            if (!parse_count(argv[++i], options.max_rows)) {
                usage(argv[0]);
                return 2;
            }
        } else if (argument == "--overwrite") {
            options.overwrite = true;
        } else {
            usage(argv[0]);
            return 2;
        }
    }
    return kmeans::preprocess_higgs(options);
}
