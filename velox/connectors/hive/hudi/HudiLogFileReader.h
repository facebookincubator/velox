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

#include <string_view>
#include <vector>

#include "velox/connectors/hive/hudi/HudiLogBlock.h"

namespace facebook::velox::connector::hive::hudi {

/// Parses the binary framing of a single Hudi log file into a sequence of
/// blocks. All multi-byte integers in the format are big-endian, matching the
/// Java DataOutputStream encoding Hudi writers use. Block content bytes are
/// returned undecoded; type-specific decoding happens in later stages.
class HudiLogFileReader {
 public:
  /// Constructs a reader over the raw bytes of one log file. The referenced
  /// buffer must outlive the reader.
  explicit HudiLogFileReader(std::string_view data) : data_{data} {}

  /// Reads every block in the file, in file order, and returns them. A clean
  /// end of file (no bytes left for another magic marker) terminates the
  /// scan; a magic marker mismatch elsewhere indicates a corrupted file and
  /// throws.
  std::vector<HudiLogBlock> readAllBlocks();

 private:
  // Reads the next block starting at the current position into `block`. Returns
  // false at a clean end of file (no further magic marker).
  bool readNextBlock(HudiLogBlock& block);

  // Reads a header/footer metadata map: a 4-byte entry count followed by that
  // many {keyOrdinal, valueLength, valueBytes} entries.
  HudiLogBlockMetadata readMetadataMap();

  // Reads `size` bytes at the current position and advances past them.
  std::string_view readBytes(size_t size);

  // Reads a big-endian unsigned integer of type T and advances past it.
  template <typename T>
  T readBigEndian();

  const std::string_view data_;
  size_t pos_{0};
};

} // namespace facebook::velox::connector::hive::hudi
