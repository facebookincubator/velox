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

#include "velox/dwio/parquet/common/PageIndex.h"

#include <cmath>
#include <cstring>
#include <limits>

#include "velox/common/base/CheckedArithmetic.h"

namespace facebook::velox::parquet {
namespace {

template <typename T>
std::optional<T> decodeFixed(std::string_view bytes) {
  if (bytes.size() != sizeof(T)) {
    return std::nullopt;
  }
  T value;
  std::memcpy(&value, bytes.data(), sizeof(T));
  return value;
}

PageIndexResult<DecodedPageBounds> decodeBounds(
    const std::string& minimum,
    const std::string& maximum,
    thrift::Type physicalType,
    std::optional<int32_t> typeLength,
    PageBoundsCapability capability) {
  if (capability != PageBoundsCapability::kOrderedBounds) {
    return PageIndexResult<DecodedPageBounds>::success(DecodedPageBounds{});
  }

  DecodedPageBounds result;
  switch (physicalType) {
    case thrift::Type::BOOLEAN: {
      if (minimum.size() != 1 || maximum.size() != 1) {
        return PageIndexResult<DecodedPageBounds>::fallback(
            PageIndexFallbackReason::kInvalidBounds,
            "BOOLEAN bounds must contain one byte");
      }
      const auto minimumValue = static_cast<uint8_t>(minimum[0]);
      const auto maximumValue = static_cast<uint8_t>(maximum[0]);
      if (minimumValue > 1 || maximumValue > 1) {
        return PageIndexResult<DecodedPageBounds>::fallback(
            PageIndexFallbackReason::kInvalidBounds,
            "BOOLEAN bounds must be zero or one");
      }
      result.minimum = minimumValue != 0;
      result.maximum = maximumValue != 0;
      break;
    }
    case thrift::Type::INT32: {
      auto min = decodeFixed<int32_t>(minimum);
      auto max = decodeFixed<int32_t>(maximum);
      if (!min || !max) {
        return PageIndexResult<DecodedPageBounds>::fallback(
            PageIndexFallbackReason::kInvalidBounds,
            "INT32 bounds have the wrong width");
      }
      result.minimum = *min;
      result.maximum = *max;
      break;
    }
    case thrift::Type::INT64: {
      auto min = decodeFixed<int64_t>(minimum);
      auto max = decodeFixed<int64_t>(maximum);
      if (!min || !max) {
        return PageIndexResult<DecodedPageBounds>::fallback(
            PageIndexFallbackReason::kInvalidBounds,
            "INT64 bounds have the wrong width");
      }
      result.minimum = *min;
      result.maximum = *max;
      break;
    }
    case thrift::Type::FLOAT: {
      auto min = decodeFixed<float>(minimum);
      auto max = decodeFixed<float>(maximum);
      if (!min || !max) {
        return PageIndexResult<DecodedPageBounds>::fallback(
            PageIndexFallbackReason::kInvalidBounds,
            "FLOAT bounds have the wrong width");
      }
      result.minimum = *min;
      result.maximum = *max;
      break;
    }
    case thrift::Type::DOUBLE: {
      auto min = decodeFixed<double>(minimum);
      auto max = decodeFixed<double>(maximum);
      if (!min || !max) {
        return PageIndexResult<DecodedPageBounds>::fallback(
            PageIndexFallbackReason::kInvalidBounds,
            "DOUBLE bounds have the wrong width");
      }
      result.minimum = *min;
      result.maximum = *max;
      break;
    }
    case thrift::Type::BYTE_ARRAY:
      result.minimum = minimum;
      result.maximum = maximum;
      break;
    case thrift::Type::FIXED_LEN_BYTE_ARRAY:
      if (!typeLength || *typeLength <= 0 ||
          minimum.size() != static_cast<size_t>(*typeLength) ||
          maximum.size() != static_cast<size_t>(*typeLength)) {
        return PageIndexResult<DecodedPageBounds>::fallback(
            PageIndexFallbackReason::kInvalidBounds,
            "FIXED_LEN_BYTE_ARRAY bounds have the wrong width");
      }
      result.minimum = minimum;
      result.maximum = maximum;
      break;
    case thrift::Type::INT96:
      return PageIndexResult<DecodedPageBounds>::fallback(
          PageIndexFallbackReason::kUnsupportedPhysicalType,
          "INT96 bounds are unsupported");
    default:
      return PageIndexResult<DecodedPageBounds>::fallback(
          PageIndexFallbackReason::kUnsupportedPhysicalType,
          "Physical type does not have a supported bound decoder");
  }

  const bool unusableBounds = std::visit(
      [&](const auto& min) {
        using Value = std::decay_t<decltype(min)>;
        if (!std::holds_alternative<Value>(*result.maximum)) {
          return true;
        }
        const auto& max = std::get<Value>(*result.maximum);
        if constexpr (
            std::is_same_v<Value, float> || std::is_same_v<Value, double>) {
          if (std::isnan(min) || std::isnan(max)) {
            return true;
          }
        }
        return min > max;
      },
      *result.minimum);
  if (unusableBounds) {
    return PageIndexResult<DecodedPageBounds>::success(DecodedPageBounds{});
  }
  return PageIndexResult<DecodedPageBounds>::success(std::move(result));
}

} // namespace

template <typename T>
PageIndexResult<T> deserializeIndex(dwio::common::SeekableInputStream& input) {
  T result;
  try {
    thrift::deserializePageIndex(&result, &input, nullptr, 0);
  } catch (const VeloxRuntimeError& error) {
    return PageIndexResult<T>::fallback(
        PageIndexFallbackReason::kDeserializeFailure, error.what());
  }
  return PageIndexResult<T>::success(std::move(result));
}

PageIndexResult<thrift::ColumnIndex> deserializeColumnIndex(
    dwio::common::SeekableInputStream& input) {
  return deserializeIndex<thrift::ColumnIndex>(input);
}

PageIndexResult<thrift::OffsetIndex> deserializeOffsetIndex(
    dwio::common::SeekableInputStream& input) {
  return deserializeIndex<thrift::OffsetIndex>(input);
}

PageIndexResult<common::Region>
validatePageIndexRegion(int64_t offset, int32_t length, uint64_t fileLength) {
  if (offset < 0 || length <= 0) {
    return PageIndexResult<common::Region>::fallback(
        PageIndexFallbackReason::kInvalidLocation,
        "Page-index region has a non-positive length or negative offset");
  }
  const auto unsignedOffset = static_cast<uint64_t>(offset);
  const auto unsignedLength = static_cast<uint64_t>(length);
  if (unsignedOffset > fileLength ||
      unsignedLength > fileLength - unsignedOffset) {
    return PageIndexResult<common::Region>::fallback(
        PageIndexFallbackReason::kInvalidLocation,
        "Page-index region extends beyond the file");
  }
  return PageIndexResult<common::Region>::success(
      common::Region{unsignedOffset, unsignedLength});
}

PageIndexResult<ValidatedPageIndexes> decodeColumnIndex(
    const thrift::ColumnIndex& columnIndex,
    const ValidatedOffsetIndex& offsetIndex,
    thrift::Type physicalType,
    std::optional<int32_t> typeLength,
    PageBoundsCapability requestedCapability,
    uint64_t rowGroupRows,
    uint64_t fileLength) {
  const auto numPages = offsetIndex.pages.size();
  if (numPages == 0 || numPages > kMaxPageIndexPages ||
      columnIndex.null_pages()->size() != numPages ||
      columnIndex.min_values()->size() != numPages ||
      columnIndex.max_values()->size() != numPages ||
      (columnIndex.null_counts() &&
       columnIndex.null_counts()->size() != numPages)) {
    return PageIndexResult<ValidatedPageIndexes>::fallback(
        PageIndexFallbackReason::kCardinalityMismatch,
        "Page-index vector cardinalities do not match");
  }

  ValidatedPageIndexes result;
  result.offset.pages.reserve(numPages);
  std::vector<bool> nullPages;
  std::vector<std::optional<uint64_t>> nullCounts;
  std::vector<DecodedPageBounds> decodedBounds;
  nullPages.reserve(numPages);
  nullCounts.reserve(numPages);
  decodedBounds.reserve(numPages);

  for (size_t i = 0; i < numPages; ++i) {
    const auto& location = offsetIndex.pages.at(i);
    const auto nullCount = columnIndex.null_counts()
        ? std::optional<int64_t>((*columnIndex.null_counts())[i])
        : std::nullopt;
    if (nullCount.has_value() &&
        (nullCount.value() < 0 ||
         static_cast<uint64_t>(nullCount.value()) > location.numRows)) {
      return PageIndexResult<ValidatedPageIndexes>::fallback(
          PageIndexFallbackReason::kInvalidBounds,
          "Null count exceeds the page row count");
    }

    const bool nullPage = (*columnIndex.null_pages())[i];
    if (nullCount.has_value() &&
        ((nullPage &&
          nullCount.value() != static_cast<int64_t>(location.numRows)) ||
         (!nullPage &&
          nullCount.value() == static_cast<int64_t>(location.numRows)))) {
      return PageIndexResult<ValidatedPageIndexes>::fallback(
          PageIndexFallbackReason::kInvalidBounds,
          "Null-page and null-count metadata disagree");
    }
    result.offset.pages.push_back(location);
    nullPages.push_back(nullPage);
    nullCounts.push_back(
        nullCount.has_value()
            ? std::optional<uint64_t>(static_cast<uint64_t>(nullCount.value()))
            : std::nullopt);

    if (nullPage) {
      decodedBounds.push_back({});
    } else {
      auto decodedPageBounds = decodeBounds(
          (*columnIndex.min_values())[i],
          (*columnIndex.max_values())[i],
          physicalType,
          typeLength,
          requestedCapability);
      if (!decodedPageBounds &&
          requestedCapability == PageBoundsCapability::kOrderedBounds) {
        return PageIndexResult<ValidatedPageIndexes>::fallback(
            decodedPageBounds.reason, decodedPageBounds.detail);
      }
      decodedBounds.push_back(
          decodedPageBounds.value.value_or(DecodedPageBounds{}));
    }
  }
  result.column = DecodedColumnIndex(
      requestedCapability,
      std::move(nullPages),
      std::move(nullCounts),
      std::move(decodedBounds));
  return PageIndexResult<ValidatedPageIndexes>::success(std::move(result));
}

PageIndexResult<ValidatedOffsetIndex> decodeOffsetIndex(
    const thrift::OffsetIndex& offsetIndex,
    uint64_t rowGroupRows,
    uint64_t fileLength,
    std::optional<uint64_t> dataPageOffset,
    std::optional<common::Region> columnChunkRegion) {
  const auto numPages = offsetIndex.page_locations()->size();
  if (numPages == 0 || numPages > kMaxPageIndexPages) {
    return PageIndexResult<ValidatedOffsetIndex>::fallback(
        PageIndexFallbackReason::kCardinalityMismatch,
        "Offset-index page count is invalid");
  }

  ValidatedOffsetIndex result;
  result.pages.reserve(numPages);
  uint64_t previousFirstRow{0};
  uint64_t previousEnd{0};
  for (size_t i = 0; i < numPages; ++i) {
    const auto& location = offsetIndex.page_locations()->at(i);
    const auto firstRow = *location.first_row_index();
    const auto offset = *location.offset();
    const auto compressedSize = *location.compressed_page_size();
    if (firstRow < 0 || static_cast<uint64_t>(firstRow) >= rowGroupRows ||
        (i == 0 ? firstRow != 0
                : static_cast<uint64_t>(firstRow) <= previousFirstRow)) {
      return PageIndexResult<ValidatedOffsetIndex>::fallback(
          PageIndexFallbackReason::kInvalidRowOrder,
          "Offset-index row or page location is invalid");
    }
    if (offset < 0 || compressedSize <= 0) {
      return PageIndexResult<ValidatedOffsetIndex>::fallback(
          PageIndexFallbackReason::kInvalidLocation,
          "Offset-index page location is invalid");
    }

    const auto unsignedOffset = static_cast<uint64_t>(offset);
    const auto unsignedSize = static_cast<uint64_t>(compressedSize);
    if (unsignedOffset > fileLength ||
        unsignedSize > fileLength - unsignedOffset ||
        (dataPageOffset.has_value() &&
         unsignedOffset < dataPageOffset.value()) ||
        (columnChunkRegion.has_value() &&
         (unsignedOffset < columnChunkRegion->offset ||
          unsignedOffset - columnChunkRegion->offset >
              columnChunkRegion->length ||
          unsignedSize > columnChunkRegion->length -
                  (unsignedOffset - columnChunkRegion->offset))) ||
        (i > 0 && unsignedOffset < previousEnd)) {
      return PageIndexResult<ValidatedOffsetIndex>::fallback(
          PageIndexFallbackReason::kInvalidLocation,
          "Data page region is outside the file or overlaps a prior page");
    }

    const auto endRow = i + 1 < numPages
        ? static_cast<uint64_t>(
              *offsetIndex.page_locations()->at(i + 1).first_row_index())
        : rowGroupRows;
    if (endRow <= static_cast<uint64_t>(firstRow)) {
      return PageIndexResult<ValidatedOffsetIndex>::fallback(
          PageIndexFallbackReason::kInvalidRowOrder,
          "Data page row span is empty");
    }

    result.pages.push_back(
        {unsignedOffset,
         static_cast<uint32_t>(compressedSize),
         static_cast<uint64_t>(firstRow),
         endRow - static_cast<uint64_t>(firstRow)});
    previousFirstRow = static_cast<uint64_t>(firstRow);
    if (unsignedOffset > std::numeric_limits<uint64_t>::max() - unsignedSize) {
      return PageIndexResult<ValidatedOffsetIndex>::fallback(
          PageIndexFallbackReason::kInvalidLocation,
          "Data page region end overflows");
    }
    previousEnd = unsignedOffset + unsignedSize;
  }
  return PageIndexResult<ValidatedOffsetIndex>::success(std::move(result));
}

const char* pageIndexFallbackReasonName(PageIndexFallbackReason reason) {
  switch (reason) {
    case PageIndexFallbackReason::kNone:
      return "none";
    case PageIndexFallbackReason::kDisabled:
      return "disabled";
    case PageIndexFallbackReason::kNoFilter:
      return "noFilter";
    case PageIndexFallbackReason::kMissingColumnIndex:
      return "missingColumnIndex";
    case PageIndexFallbackReason::kMissingOffsetIndex:
      return "missingOffsetIndex";
    case PageIndexFallbackReason::kMissingColumnOrder:
      return "missingColumnOrder";
    case PageIndexFallbackReason::kUnsupportedColumnOrder:
      return "unsupportedColumnOrder";
    case PageIndexFallbackReason::kUnsupportedPhysicalType:
      return "unsupportedPhysicalType";
    case PageIndexFallbackReason::kUnsupportedLogicalType:
      return "unsupportedLogicalType";
    case PageIndexFallbackReason::kUnsupportedNestedColumn:
      return "unsupportedNestedColumn";
    case PageIndexFallbackReason::kInvalidLocation:
      return "invalidLocation";
    case PageIndexFallbackReason::kIndexTooLarge:
      return "indexTooLarge";
    case PageIndexFallbackReason::kDeserializeFailure:
      return "deserializeFailure";
    case PageIndexFallbackReason::kCardinalityMismatch:
      return "cardinalityMismatch";
    case PageIndexFallbackReason::kInvalidPageLocation:
      return "invalidPageLocation";
    case PageIndexFallbackReason::kInvalidRowOrder:
      return "invalidRowOrder";
    case PageIndexFallbackReason::kInvalidBounds:
      return "invalidBounds";
    case PageIndexFallbackReason::kCostModel:
      return "costModel";
  }
  return "unknown";
}

} // namespace facebook::velox::parquet
