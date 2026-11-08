#pragma once

#include "kmeans/higgs_dataset.hpp"

#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

namespace kmeans {

// Writes rows to "<path>.part" and renames it to <path> only after a complete write.
class BinaryDatasetWriter {
public:
    BinaryDatasetWriter() = default;
    ~BinaryDatasetWriter() { abort(); }
    BinaryDatasetWriter(const BinaryDatasetWriter&) = delete;
    BinaryDatasetWriter& operator=(const BinaryDatasetWriter&) = delete;

    bool open(const std::string& path, bool overwrite);
    bool write(const FeatureRow& row);
    bool finish();
    void abort();
    std::uint64_t rows_written() const { return rows_written_; }

private:
    std::vector<char> buffer_;  // declared before stream_ so it outlives the stream
    std::ofstream stream_;
    std::string final_path_;
    std::string partial_path_;
    std::uint64_t rows_written_ = 0;
};

}  // namespace kmeans
