/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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

#include <string>
#include <string_view>

#include "velox/dwio/nimble/common/Types.h"

namespace facebook::nimble {

/// Postscript information (last 20 bytes of a Nimble file).
/// Contains file format version, checksum, and footer metadata.
class Postscript {
 public:
  // 4 (footerSize) + 1 (compressionType) + 1 (checksumType) +
  // 8 (checksum) + 2 (major) + 2 (minor) + 2 (magic) = 20 bytes.
  static constexpr uint32_t kSize{20};

  uint32_t footerSize() const {
    return footerSize_;
  }

  CompressionType footerCompressionType() const {
    return footerCompressionType_;
  }

  uint64_t checksum() const {
    return checksum_;
  }

  ChecksumType checksumType() const {
    return checksumType_;
  }

  uint32_t majorVersion() const {
    return majorVersion_;
  }

  uint32_t minorVersion() const {
    return minorVersion_;
  }

  Postscript() = default;

  Postscript(
      uint32_t footerSize,
      CompressionType footerCompressionType,
      ChecksumType checksumType,
      uint32_t majorVersion,
      uint32_t minorVersion);

  /// Parses a postscript from the last kPostscriptSize bytes of data.
  static Postscript parse(std::string_view data);

  /// Serializes the postscript into a kPostscriptSize-byte string.
  std::string serialize() const;

 private:
  uint32_t footerSize_{0};
  CompressionType footerCompressionType_{CompressionType::Uncompressed};
  uint64_t checksum_{0};
  ChecksumType checksumType_{ChecksumType::XXH3_64};
  uint32_t majorVersion_{0};
  uint32_t minorVersion_{0};
};

} // namespace facebook::nimble
