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

#include <string_view>

#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/encodings/common/Encoding.h"

namespace facebook::nimble {

/// Creates encoded streams for row ranges from existing encoded streams.
///
/// Uses native slicing for supported encodings and falls back to materializing
/// and re-encoding the requested range when metadata-only slicing is not
/// available.
class EncodingSliceFactory {
 public:
  /// Returns an encoded stream containing rows [offset, offset + length).
  static std::string_view slice(
      std::string_view encoded,
      uint32_t offset,
      uint32_t length,
      Buffer& buffer,
      const Encoding::Options& options = {});
};

} // namespace facebook::nimble
