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

#include <initializer_list>
#include <string>
#include <utility>
#include <vector>

#include "velox/common/base/BitUtil.h"
#include "velox/functions/sparksql/aggregates/BitmapConstructAggAggregate.h"
#include "velox/vector/FlatVector.h"
#include "velox/vector/tests/utils/VectorMaker.h"

namespace facebook::velox::functions::aggregate::sparksql::test {

/// Utility for constructing test bitmaps. Consolidates bitmap creation
/// helpers into a consistent API for use across bitmap aggregate tests.
class BitmapBuilder {
 public:
  /// Returns a 4096-byte bitmap string with specified bit positions set.
  static std::string fromBits(const std::vector<int64_t>& positions) {
    std::string bitmap(kBitmapNumBytes, '\0');
    auto* data = reinterpret_cast<uint8_t*>(bitmap.data());
    for (auto position : positions) {
      bits::setBit(data, position);
    }
    return bitmap;
  }

  /// Returns a 4096-byte bitmap string with consecutive bytes starting
  /// at index 0, rest zeros.
  static std::string fromBytes(std::initializer_list<uint8_t> bytes) {
    std::string bitmap(kBitmapNumBytes, '\0');
    int32_t index = 0;
    for (auto byte : bytes) {
      bitmap[index++] = static_cast<char>(byte);
    }
    return bitmap;
  }

  /// Returns a 4096-byte bitmap string with specific byte positions
  /// set.
  static std::string fromBytes(
      const std::vector<std::pair<int32_t, uint8_t>>& bytesAndValues) {
    std::string bitmap(kBitmapNumBytes, '\0');
    for (const auto& [byteIndex, value] : bytesAndValues) {
      bitmap[byteIndex] = static_cast<char>(value);
    }
    return bitmap;
  }

  static std::string fromBytes(
      std::initializer_list<std::pair<int32_t, uint8_t>> bytesAndValues) {
    return fromBytes(std::vector<std::pair<int32_t, uint8_t>>(bytesAndValues));
  }

  /// Creates a VARBINARY FlatVector with one bitmap per entry, each
  /// constructed from bit positions.
  static VectorPtr vectorFromBits(
      memory::MemoryPool* pool,
      std::initializer_list<std::vector<int64_t>> bitmaps) {
    std::vector<std::string> strings;
    strings.reserve(bitmaps.size());
    for (const auto& positions : bitmaps) {
      strings.push_back(fromBits(positions));
    }
    return makeVector(pool, strings);
  }

  /// Creates a VARBINARY FlatVector with one bitmap per entry, each
  /// constructed from byte position/value pairs.
  static VectorPtr vectorFromBytes(
      memory::MemoryPool* pool,
      std::initializer_list<std::vector<std::pair<int32_t, uint8_t>>> bitmaps) {
    std::vector<std::string> strings;
    strings.reserve(bitmaps.size());
    for (const auto& bytesAndValues : bitmaps) {
      strings.push_back(fromBytes(bytesAndValues));
    }
    return makeVector(pool, strings);
  }

 private:
  static VectorPtr makeVector(
      memory::MemoryPool* pool,
      const std::vector<std::string>& bitmaps) {
    velox::test::VectorMaker maker(pool);
    return maker.flatVector(bitmaps, VARBINARY());
  }
};

} // namespace facebook::velox::functions::aggregate::sparksql::test
