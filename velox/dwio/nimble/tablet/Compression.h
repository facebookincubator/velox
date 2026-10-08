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

#include <optional>
#include <string_view>

#include "velox/buffer/Buffer.h"

namespace facebook::nimble {

class ZstdCompression {
 public:
  static std::optional<velox::BufferPtr> compress(
      std::string_view source,
      velox::memory::MemoryPool* pool,
      int32_t compressionLevel = 1);

  static velox::BufferPtr uncompress(
      std::string_view source,
      velox::memory::MemoryPool* pool);

  /// Decompresses into a caller-provided destination buffer.
  /// Skips the 4-byte uncompressed size header in source.
  static void uncompress(
      std::string_view source,
      void* destination,
      uint64_t destinationSize);
};

} // namespace facebook::nimble
