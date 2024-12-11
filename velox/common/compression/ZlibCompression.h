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

#include <memory>
#include "velox/common/compression/Compression.h"

namespace facebook::velox::common {

// Maximum window size.
constexpr int32_t kZlibMaxWindowBits = 15;
// Minimum window size.
constexpr int32_t kZlibMinWindowBits = 9;
// Default window size.
constexpr int32_t kZlibDefaultWindowBits = 15;

enum class ZlibFormat {
  kZlib,
  kDeflate,
  kGzip,
};

class ZlibCodecOptions : public CodecOptions {
 public:
  explicit ZlibCodecOptions(
      ZlibFormat format,
      int32_t compressionLevel = kDefaultCompressionLevel,
      std::optional<int32_t> windowBits = kZlibDefaultWindowBits)
      : CodecOptions(compressionLevel),
        format(format),
        windowBits(windowBits) {}

  ZlibFormat format;
  std::optional<int32_t> windowBits;
};

/// The windowBits argument controls the size of the history buffer (or the
/// “window size”) used when compressing data. Valid range for windowBits
/// is 9..15. The default value is 15.
std::unique_ptr<Codec> makeZlibCodec(
    ZlibFormat format,
    int32_t compressionLevel = kDefaultCompressionLevel,
    std::optional<int32_t> windowBits = std::nullopt);

} // namespace facebook::velox::common
