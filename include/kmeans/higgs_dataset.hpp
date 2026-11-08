#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

namespace kmeans {

constexpr int kHiggsColumns = 29;  // label followed by 28 features
constexpr int kHiggsFeatures = 28;
constexpr std::uint64_t kHiggsRows = 11'000'000;
constexpr std::size_t kHiggsLineCapacity = 32768;

using FeatureRow = std::array<float, kHiggsFeatures>;

struct HiggsDatasetOptions {
    std::string archive_path = "dataset/HIGGS.csv.gz";
    std::string output_path = "dataset/raw/HIGGS.csv";
    std::uint64_t sample_rows = 1000;
    std::uint64_t expected_rows = kHiggsRows;
    bool full_scan = false;
    bool overwrite = false;
};

int inspect_archive(const HiggsDatasetOptions& options);
int extract_archive(const HiggsDatasetOptions& options);

}  // namespace kmeans
