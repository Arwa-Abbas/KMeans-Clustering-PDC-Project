#pragma once

#include "kmeans/kmeans.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>

namespace kmeans {

struct RunOptions {
    std::string data = "dataset/processed/higgs_1m_f32.bin";
    std::string metadata = "dataset/processed/higgs_1m_f32.metadata.json";
    std::string output;
    std::uint64_t rows = 0;  // 0: every row in the file
    int threads = 1;
    Config config;
};

class UsageError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// Options shared by the four K-Means programs; --threads only where `accepts_threads`.
RunOptions parse_run_options(int argc, char** argv, bool accepts_threads);
std::string run_usage(const std::string& program, bool accepts_threads);

}  // namespace kmeans
