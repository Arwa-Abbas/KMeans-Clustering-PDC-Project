#include "kmeans/higgs_dataset.hpp"
#include "kmeans/kmeans.hpp"

#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;

namespace kmeans {
namespace {

std::uint64_t metadata_number(const std::string& text, const std::string& key) {
    const std::string needle = "\"" + key + "\"";
    std::size_t at = text.find(needle);
    if (at != std::string::npos) at = text.find(':', at + needle.size());
    if (at == std::string::npos) throw std::runtime_error("metadata has no \"" + key + "\" field");
    const char* start = text.c_str() + at + 1;
    char* end = nullptr;
    errno = 0;
    const unsigned long long value = std::strtoull(start, &end, 10);
    if (end == start || errno != 0 || value == 0)
        throw std::runtime_error("metadata field \"" + key + "\" is not a positive integer");
    return value;
}

}  // namespace

MatrixShape read_metadata(const std::string& metadata_path) {
    std::ifstream stream(metadata_path, std::ios::binary);
    if (!stream) throw std::runtime_error("cannot open " + metadata_path);
    std::ostringstream buffer;
    buffer << stream.rdbuf();
    const std::string text = buffer.str();

    const std::uint64_t rows = metadata_number(text, "rows");
    const std::uint64_t features = metadata_number(text, "features");
    if (features != static_cast<std::uint64_t>(kHiggsFeatures) ||
        text.find("\"float32\"") == std::string::npos ||
        text.find("\"little-endian\"") == std::string::npos) {
        throw std::runtime_error("unsupported matrix metadata in " + metadata_path);
    }
    if (rows > std::numeric_limits<std::size_t>::max() / (features * sizeof(float)))
        throw std::runtime_error("matrix in " + metadata_path + " is too large to address");
    return {rows, static_cast<int>(features)};
}

Matrix load_matrix(const std::string& binary_path, const std::string& metadata_path,
                   std::uint64_t row_limit) {
    const MatrixShape shape = read_metadata(metadata_path);
    const std::uint64_t row_bytes = static_cast<std::uint64_t>(shape.dimensions) * sizeof(float);
    std::error_code error;
    const std::uintmax_t file_bytes = fs::file_size(binary_path, error);
    if (error || file_bytes != shape.rows * row_bytes)
        throw std::runtime_error(binary_path + " does not match the size in " + metadata_path);
    if (row_limit > shape.rows) {
        throw std::invalid_argument("--rows " + std::to_string(row_limit) + " exceeds the " +
                                    std::to_string(shape.rows) + " rows in " + binary_path);
    }

    Matrix matrix;
    matrix.rows = row_limit != 0 ? row_limit : shape.rows;
    matrix.dimensions = shape.dimensions;
    matrix.values.resize(matrix.rows * static_cast<std::uint64_t>(shape.dimensions));
    std::ifstream stream(binary_path, std::ios::binary);
    if (!stream.read(reinterpret_cast<char*>(matrix.values.data()),
                     static_cast<std::streamsize>(matrix.rows * row_bytes))) {
        throw std::runtime_error("cannot read " + binary_path);
    }
    return matrix;
}

}  // namespace kmeans
