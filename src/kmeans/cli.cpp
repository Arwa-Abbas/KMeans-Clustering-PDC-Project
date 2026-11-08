#include "kmeans/cli.hpp"

#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <limits>

namespace kmeans {
namespace {

std::uint64_t whole_number(const std::string& key, const std::string& text, std::uint64_t limit) {
    char* end = nullptr;
    errno = 0;
    const unsigned long long value =
        text.empty() || text[0] == '-' ? 0 : std::strtoull(text.c_str(), &end, 10);
    if (end == nullptr || end == text.c_str() || *end != '\0' || errno != 0 || value == 0 ||
        value > limit) {
        throw UsageError(key + " expects a whole number from 1 to " + std::to_string(limit));
    }
    return value;
}

double fraction(const std::string& key, const std::string& text) {
    char* end = nullptr;
    errno = 0;
    const double value = std::strtod(text.c_str(), &end);
    if (end == text.c_str() || *end != '\0' || errno != 0 || !std::isfinite(value) || value < 0 ||
        value > 1) {
        throw UsageError(key + " expects a fraction from 0 to 1");
    }
    return value;
}

}  // namespace

RunOptions parse_run_options(int argc, char** argv, bool accepts_threads) {
    constexpr std::uint64_t any = std::numeric_limits<std::uint64_t>::max();
    RunOptions options;
    for (int i = 1; i < argc; i += 2) {
        const std::string key = argv[i];
        if (i + 1 >= argc) throw UsageError(key + " needs a value");
        const std::string value = argv[i + 1];
        if (key == "--data") options.data = value;
        else if (key == "--metadata") options.metadata = value;
        else if (key == "--output") options.output = value;
        else if (key == "--rows") options.rows = whole_number(key, value, any);
        else if (key == "--clusters")
            options.config.clusters = static_cast<int>(whole_number(key, value, 100000));
        else if (key == "--max-iterations")
            options.config.max_iterations = static_cast<int>(whole_number(key, value, 100000));
        else if (key == "--tolerance") options.config.tolerance = fraction(key, value);
        else if (key == "--seed") options.config.seed = whole_number(key, value, any);
        else if (key == "--init-sample-size") options.config.sample_size = whole_number(key, value, any);
        else if (key == "--threads" && accepts_threads)
            options.threads = static_cast<int>(whole_number(key, value, 1024));
        else throw UsageError("unknown option " + key);
    }
    return options;
}

std::string run_usage(const std::string& program, bool accepts_threads) {
    return "Usage: " + program +
           " [--data BIN] [--metadata JSON] [--output JSON] [--rows N]\n"
           "       [--clusters K] [--max-iterations N] [--tolerance FRACTION]\n"
           "       [--seed N] [--init-sample-size N]" +
           (accepts_threads ? " [--threads N]" : "") + "\n";
}

}  // namespace kmeans
