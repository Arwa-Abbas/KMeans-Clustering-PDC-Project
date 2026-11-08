#include "kmeans/higgs_reader.hpp"

#include <cmath>
#include <cstdlib>
#include <cstring>

namespace kmeans {

HiggsReader::HiggsReader(const std::string& path)
    : stream_(gzopen(path.c_str(), "rb")), line_(kHiggsLineCapacity) {}

int HiggsReader::next(FeatureRow& features, int& label) {
    if (gzgets(stream_.get(), line_.data(), static_cast<int>(line_.size())) == nullptr) {
        if (gzeof(stream_.get())) return 0;
        int code = Z_OK;
        error_ = std::string("gzip read failed: ") + gzerror(stream_.get(), &code);
        return -1;
    }
    ++row_;
    const std::size_t length = std::strlen(line_.data());
    if (length == 0 || (line_[length - 1] != '\n' && !gzeof(stream_.get()))) {
        error_ = "row " + std::to_string(row_) + " exceeds the line buffer";
        return -1;
    }
    return parse_line(features, label);
}

int HiggsReader::parse_line(FeatureRow& features, int& label) {
    const char* cursor = line_.data();
    for (int column = 0; column < kHiggsColumns; ++column) {
        char* end = nullptr;
        const float value = std::strtof(cursor, &end);
        if (end == cursor || !std::isfinite(value)) {
            error_ = "row " + std::to_string(row_) + " column " + std::to_string(column + 1) +
                     " is not a finite number";
            return -1;
        }
        if (column == 0) {
            if (value != 0.0F && value != 1.0F) {
                error_ = "row " + std::to_string(row_) + " label is not 0 or 1";
                return -1;
            }
            label = static_cast<int>(value);
        } else {
            features[column - 1] = value;
        }

        if (column < kHiggsColumns - 1) {
            if (*end != ',') {
                error_ = "row " + std::to_string(row_) + " does not contain " +
                         std::to_string(kHiggsColumns) + " columns";
                return -1;
            }
            cursor = end + 1;
        } else {
            while (*end == '\r' || *end == '\n') ++end;
            if (*end != '\0') {
                error_ = "row " + std::to_string(row_) + " contains trailing data";
                return -1;
            }
        }
    }
    return 1;
}

}  // namespace kmeans
