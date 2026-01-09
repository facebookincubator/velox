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

#include "velox/core/QueryConfig.h"
#include "velox/functions/Macros.h"

#include <folly/CPortability.h>
#include <roaring/roaring.hh>

namespace facebook::velox::functions::delta {

/// The C++ implementation of 64-bit array-based roaring bitmap which
/// is used by Delta Lake deletion vectors. See
/// https://github.com/delta-io/delta/blob/master/PROTOCOL.md#deletion-vector-format.
class RoaringBitmapArray {
 public:
  RoaringBitmapArray() = default;

  /// Deserializes the bitmap array from a buffer.
  void deserialize(const char* serialized);

  /// Serializes the bitmap into a buffer.
  void serialize(char* buf) const;

  /// Adds an int64 value to the bitmap array. Returns an error
  /// when the value is invalid.
  Status add(int64_t value);

  /// Adds an int64 value to the bitmap array. Throws when the
  /// value is invalid.
  void addSafe(int64_t value);

  /// Tests whether the value is in the bitmap array. Returns an error
  /// when the value is invalid.
  Status contains(bool& result, int64_t value);

  /// Tests whether the value is in the bitmap array. Throws when the
  /// value is invalid.
  bool containsSafe(int64_t value);

  /// Returns the serialized size in bytes of the bitmap array.
  int64_t serializedSizeInBytes() const;

 private:
  static Status checkValue(int64_t value);

  static constexpr int32_t kPortableSerializationFormatMagicNumber{1681511377};
  std::vector<std::shared_ptr<roaring::Roaring>> bitmaps_{};
  std::vector<std::shared_ptr<roaring::BulkContext>> buckContexts_{};
  int32_t lastHighBytes_{-1};
  roaring::Roaring* lastBitmap_{nullptr};
  roaring::BulkContext* lastContext_{nullptr};
};

template <typename TExec>
struct RoaringBitmapArrayContains {
  VELOX_DEFINE_FUNCTION_TYPES(TExec);

  void initialize(
      const std::vector<TypePtr>& /*inputTypes*/,
      const core::QueryConfig& /*config*/,
      const arg_type<Varbinary>* serialized,
      const arg_type<int64_t>* /*value*/) {
    if (serialized != nullptr) {
      array_.deserialize(serialized->str().data());
    }
  }

  FOLLY_ALWAYS_INLINE Status
  call(bool& result, const arg_type<Varbinary>&, const int64_t& input) {
    return array_.contains(result, input);
  }

 private:
  RoaringBitmapArray array_{};
};
} // namespace facebook::velox::functions::delta
