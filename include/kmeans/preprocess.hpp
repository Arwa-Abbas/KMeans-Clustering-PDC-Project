#pragma once

#include <cstdint>
#include <string>

namespace kmeans {

struct PreprocessOptions {
    std::string archive_path = "dataset/HIGGS.csv.gz";
    std::string binary_path = "dataset/processed/higgs_f32.bin";
    std::string metadata_path = "dataset/processed/higgs_f32.metadata.json";
    std::uint64_t max_rows = 0;  // 0: every row
    bool overwrite = false;
};

int preprocess_higgs(const PreprocessOptions& options);

}  // namespace kmeans
