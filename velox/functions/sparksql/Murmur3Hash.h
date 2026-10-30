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

#include <cstdint>

#include "velox/functions/lib/Murmur3Hash32Base.h"
#include "velox/type/StringView.h"

namespace facebook::velox::functions::sparksql {

/// Computes Spark-compatible Murmur3 x86 32-bit hashes for byte sequences.
class SparkMurmur3Hash final : private Murmur3Hash32Base {
 public:
  /// Hashes a byte sequence using Spark's hashUnsafeBytes semantics.
  static uint32_t hashBytes(const char* data, int32_t length, uint32_t seed);

  /// Hashes the bytes stored in a StringView.
  static uint32_t hashBytes(const StringView& input, uint32_t seed);

 private:
  /// Loads a 32-bit block using Spark's little-endian byte order.
  static uint32_t loadLittleEndian32(const char* data);
};

} // namespace facebook::velox::functions::sparksql
