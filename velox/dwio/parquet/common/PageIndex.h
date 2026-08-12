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
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "velox/common/file/Region.h"
#include "velox/dwio/common/SeekableInputStream.h"
#include "velox/dwio/parquet/thrift/ParquetThrift.h"

namespace facebook::velox::parquet {

enum class PageIndexFallbackReason {
  kNone,
  kDisabled,
  kNoFilter,
  kMissingColumnIndex,
  kMissingOffsetIndex,
  kMissingColumnOrder,
  kUnsupportedColumnOrder,
  kUnsupportedPhysicalType,
  kUnsupportedLogicalType,
  kUnsupportedNestedColumn,
  kInvalidLocation,
  kIndexTooLarge,
  kDeserializeFailure,
  kCardinalityMismatch,
  kInvalidPageLocation,
  kInvalidRowOrder,
  kInvalidBounds,
  kCostModel,
};

enum class PageBoundsCapability {
  kNone,
  kNullabilityOnly,
  kOrderedBounds,
};

template <typename T>
struct PageIndexResult {
  std::optional<T> value;
  PageIndexFallbackReason reason{PageIndexFallbackReason::kNone};
  std::string detail;

  explicit operator bool() const {
    return value.has_value();
  }

  static PageIndexResult success(T result) {
    PageIndexResult output;
    output.value = std::move(result);
    return output;
  }

  static PageIndexResult fallback(
      PageIndexFallbackReason fallbackReason,
      std::string fallbackDetail = {}) {
    PageIndexResult output;
    output.reason = fallbackReason;
    output.detail = std::move(fallbackDetail);
    return output;
  }
};

struct ValidatedPageLocation {
  uint64_t offset{0};
  uint32_t compressedSize{0};
  uint64_t firstRow{0};
  uint64_t numRows{0};
};

struct ValidatedOffsetIndex {
  std::vector<ValidatedPageLocation> pages;
};

using PageIndexValue =
    std::variant<bool, int32_t, int64_t, float, double, std::string>;

struct DecodedPageBounds {
  std::optional<PageIndexValue> minimum;
  std::optional<PageIndexValue> maximum;
};

class DecodedColumnIndex {
 public:
  DecodedColumnIndex() = default;

  DecodedColumnIndex(
      PageBoundsCapability capability,
      std::vector<bool> nullPages,
      std::vector<std::optional<uint64_t>> nullCounts,
      std::vector<DecodedPageBounds> bounds)
      : capability_(capability),
        nullPages_(std::move(nullPages)),
        nullCounts_(std::move(nullCounts)),
        bounds_(std::move(bounds)) {}

  size_t numPages() const {
    return nullPages_.size();
  }

  bool isNullPage(size_t page) const {
    return nullPages_.at(page);
  }

  std::optional<uint64_t> nullCount(size_t page) const {
    return nullCounts_.at(page);
  }

  PageBoundsCapability capability() const {
    return capability_;
  }

  const DecodedPageBounds& bounds(size_t page) const {
    return bounds_.at(page);
  }

  const std::vector<bool>& nullPages() const {
    return nullPages_;
  }

  const std::vector<std::optional<uint64_t>>& nullCounts() const {
    return nullCounts_;
  }

 private:
  PageBoundsCapability capability_{PageBoundsCapability::kNone};
  std::vector<bool> nullPages_;
  std::vector<std::optional<uint64_t>> nullCounts_;
  std::vector<DecodedPageBounds> bounds_;
};

struct ValidatedPageIndexes {
  std::optional<DecodedColumnIndex> column;
  ValidatedOffsetIndex offset;
};

/// Deserializes a bounded ColumnIndex stream.
PageIndexResult<thrift::ColumnIndex> deserializeColumnIndex(
    dwio::common::SeekableInputStream& input);

/// Deserializes a bounded OffsetIndex stream.
PageIndexResult<thrift::OffsetIndex> deserializeOffsetIndex(
    dwio::common::SeekableInputStream& input);

/// Validates an OffsetIndex independently of ColumnIndex availability.
PageIndexResult<ValidatedOffsetIndex> decodeOffsetIndex(
    const thrift::OffsetIndex& offsetIndex,
    uint64_t rowGroupRows,
  uint64_t fileLength,
  std::optional<uint64_t> dataPageOffset = std::nullopt,
  std::optional<common::Region> columnChunkRegion = std::nullopt);

/// Validates and decodes one already-deserialized pair of page indexes.
PageIndexResult<ValidatedPageIndexes> decodeColumnIndex(
    const thrift::ColumnIndex& columnIndex,
    const thrift::OffsetIndex& offsetIndex,
    thrift::Type physicalType,
    std::optional<int32_t> typeLength,
    PageBoundsCapability requestedCapability,
    uint64_t rowGroupRows,
    uint64_t fileLength);

/// Validates a signed footer region before converting it to a Region.
PageIndexResult<common::Region>
validatePageIndexRegion(int64_t offset, int32_t length, uint64_t fileLength);

const char* pageIndexFallbackReasonName(PageIndexFallbackReason reason);

} // namespace facebook::velox::parquet
