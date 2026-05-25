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

namespace facebook::velox::functions::aggregate::sparksql::test {

/// Utility for constructing test bitmaps. Consolidates bitmap creation helpers
/// into a consistent API for use across bitmap aggregate tests.
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

  /// Returns a 4096-byte bitmap string with specific byte positions set.
  static std::string fromBytes(
      std::initializer_list<std::pair<int32_t, uint8_t>> bytesAndValues) {
    std::string bitmap(kBitmapNumBytes, '\0');
    for (const auto& [byteIndex, value] : bytesAndValues) {
      bitmap[byteIndex] = static_cast<char>(value);
    }
    return bitmap;
  }
};

} // namespace facebook::velox::functions::aggregate::sparksql::test
