/// EQUIVALENT CODE IN fbcode/dwio/text/TextReader.cpp

#include "velox/dwio/text/reader/TextReader.h"
#include <cstdint>
#include <string>

namespace facebook::velox::text {

const std::string TEXTFILE_CODEC = "org.apache.hadoop.io.compress.GzipCodec";
const std::string TEXTFILE_COMPRESSION_EXTENSION = ".gz";
const std::string TEXTFILE_COMPRESSION_EXTENSION_RAW = ".deflate";

/// TODO: dummy implementation
FileContents::FileContents(
    MemoryPool& pool,
    const std::shared_ptr<const RowType>& t)
    : pool{pool} {
  return;
}

} // namespace facebook::velox::text
