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

#include "velox/dwio/nimble/common/Types.h"
#include "velox/dwio/nimble/common/Varint.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrimitives.h"

/// Binary layout constants and helpers for the common encoding prefix.
///
/// Every Encoding begins with the same prefix:
///   1 byte:  EncodingType
///   1 byte:  DataType
///   4 bytes: uint32_t row count  (fixed format)
///     OR 1-5 bytes: varint row count (when useVarintRowCount is set)

namespace facebook::nimble {

struct EncodingPrefix {
  static constexpr int kEncodingTypeOffset = 0;
  static constexpr int kDataTypeOffset = 1;
  static constexpr int kRowCountOffset = 2;
  /// Prefix size for the fixed (non-varint) row count format.
  static constexpr int kFixedPrefixSize = 6;

  static void serialize(
      EncodingType encodingType,
      DataType dataType,
      uint32_t rowCount,
      bool useVarint,
      char*& pos) {
    encoding::writeChar(static_cast<char>(encodingType), pos);
    encoding::writeChar(static_cast<char>(dataType), pos);
    if (useVarint) {
      varint::writeVarint(rowCount, &pos);
    } else {
      encoding::writeUint32(rowCount, pos);
    }
  }

  /// Returns kFixedPrefixSize for fixed format, or 2 + varintSize(rowCount)
  /// for varint format.
  static uint32_t serializedSize(uint32_t rowCount, bool useVarint) {
    if (!useVarint) {
      return kFixedPrefixSize;
    }
    return kRowCountOffset + varint::varintSize(rowCount);
  }

  static EncodingType encodingType(std::string_view data) {
    return static_cast<EncodingType>(data[kEncodingTypeOffset]);
  }

  static DataType dataType(std::string_view data) {
    return static_cast<DataType>(data[kDataTypeOffset]);
  }

  static DataType readDataType(std::string_view data) {
    return dataType(data);
  }

  static uint32_t readRowCount(std::string_view data, bool useVarint) {
    if (useVarint) {
      const char* pos = data.data() + kRowCountOffset;
      return varint::readVarint32(&pos);
    }
    return *reinterpret_cast<const uint32_t*>(data.data() + kRowCountOffset);
  }

  static uint32_t prefixSize(std::string_view data, bool useVarint) {
    if (useVarint) {
      const char* pos = data.data() + kRowCountOffset;
      varint::readVarint32(&pos);
      return pos - data.data();
    }
    return kFixedPrefixSize;
  }
};

} // namespace facebook::nimble
