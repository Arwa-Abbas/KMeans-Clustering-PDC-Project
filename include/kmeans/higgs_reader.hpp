#pragma once

#include "kmeans/higgs_dataset.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <zlib.h>

namespace kmeans {

struct GzipClose {
    void operator()(gzFile file) const noexcept { gzclose(file); }
};
using GzipFile = std::unique_ptr<gzFile_s, GzipClose>;

// Streams parsed rows from HIGGS.csv.gz without decompressing it to disk.
class HiggsReader {
public:
    explicit HiggsReader(const std::string& path);

    bool is_open() const { return stream_ != nullptr; }
    // Returns 1 for a parsed row, 0 at the end of the archive, -1 on error (see error()).
    int next(FeatureRow& features, int& label);
    std::uint64_t row() const { return row_; }
    const std::string& error() const { return error_; }

private:
    int parse_line(FeatureRow& features, int& label);

    GzipFile stream_;
    std::vector<char> line_;
    std::uint64_t row_ = 0;
    std::string error_;
};

}  // namespace kmeans
