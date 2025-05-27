/// EQUIVALENT CODE IN fbcode/dwio/text/TextReader.h

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>

#include "velox/dwio/common/Reader.h"

namespace facebook::velox::text {

/**
 * State shared between Reader and Row Reader
 */
using velox::dwio::common::ColumnSelector;
using MemoryPool = facebook::velox::memory::MemoryPool;

struct FileContents {
  const size_t COLUMN_POSITION_INVALID = std::numeric_limits<size_t>::max();

  const std::shared_ptr<const RowType> schema;

  /// TODO: mising PreloadableReader
  // std::unique_ptr<utils::PreloadableReader> stream;

  MemoryPool& pool;
  uint64_t fileLength;
  common::CompressionKind compression;

  /// TODO: missing ReaderDecompressor
  // std::unique_ptr<compression::ReaderDecompressor> decompressedStream;
  velox::dwio::common::SerDeOptions serDeOptions;
  std::array<bool, 128> needsEscape;

  FileContents(MemoryPool& pool, const std::shared_ptr<const RowType>& t);
};

using DelimType = uint8_t;

constexpr DelimType DelimTypeNone = 0;
constexpr DelimType DelimTypeEOR = 1;
constexpr DelimType DelimTypeEOE = 2;

} // namespace facebook::velox::text
