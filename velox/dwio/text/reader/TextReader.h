/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>

#include "velox/dwio/text/reader/ReaderDecompressor.h"

using velox::dwio::text::compression::ReaderDecompressor;

namespace facebook::velox::text {

using namespace velox::dwio::common;

struct FileContents {
  const size_t COLUMN_POSITION_INVALID = std::numeric_limits<size_t>::max();

  const std::shared_ptr<const RowType> schema;

  /// TODO: mising member stream
  // std::unique_ptr<PreloadableReader> stream;

  memory::MemoryPool& pool;
  uint64_t fileLength;
  common::CompressionKind compression;

  std::unique_ptr<ReaderDecompressor> decompressedStream;
  SerDeOptions serDeOptions;
  std::array<bool, 128> needsEscape;

  FileContents(
      memory::MemoryPool& pool,
      const std::shared_ptr<const RowType>& t);
};

using DelimType = uint8_t;

constexpr DelimType DelimTypeNone = 0;
constexpr DelimType DelimTypeEOR = 1;
constexpr DelimType DelimTypeEOE = 2;

} // namespace facebook::velox::text
