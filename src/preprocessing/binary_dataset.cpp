#include "kmeans/binary_dataset.hpp"

#include <cstring>
#include <filesystem>
#include <iostream>

namespace fs = std::filesystem;

namespace kmeans {
namespace {

constexpr std::size_t kOutputBufferBytes = 4U * 1024U * 1024U;

bool host_is_little_endian() {
    const std::uint16_t value = 1;
    unsigned char first = 0;
    std::memcpy(&first, &value, 1);
    return first == 1;
}

}  // namespace

bool BinaryDatasetWriter::open(const std::string& path, bool overwrite) {
    if (!host_is_little_endian()) {
        std::cerr << "error: binary output currently requires little-endian hardware\n";
        return false;
    }
    final_path_ = path;
    partial_path_ = path + ".part";
    rows_written_ = 0;

    std::error_code error;
    const fs::path parent = fs::path(path).parent_path();
    if (!parent.empty()) fs::create_directories(parent, error);
    if (error) return false;
    if (!overwrite && (fs::exists(final_path_, error) || fs::exists(partial_path_, error))) {
        std::cerr << "error: binary output exists; use --overwrite\n";
        return false;
    }
    if (overwrite) {
        fs::remove(final_path_, error);
        fs::remove(partial_path_, error);
    }
    buffer_.resize(kOutputBufferBytes);
    stream_.rdbuf()->pubsetbuf(buffer_.data(), static_cast<std::streamsize>(buffer_.size()));
    stream_.open(partial_path_, std::ios::binary | std::ios::trunc);
    return stream_.is_open();
}

bool BinaryDatasetWriter::write(const FeatureRow& row) {
    stream_.write(reinterpret_cast<const char*>(row.data()),
                  static_cast<std::streamsize>(sizeof(float) * row.size()));
    if (!stream_) return false;
    ++rows_written_;
    return true;
}

bool BinaryDatasetWriter::finish() {
    // Promote the partial file only after it has been completely written.
    stream_.close();
    if (!stream_) return false;
    std::error_code error;
    fs::rename(partial_path_, final_path_, error);
    return !error;
}

void BinaryDatasetWriter::abort() {
    if (stream_.is_open()) stream_.close();
}

}  // namespace kmeans
