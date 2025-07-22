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

#include "velox/dwio/common/RowRanges.h"
#include "velox/dwio/common/Statistics.h"
#include "velox/dwio/parquet/thrift/ParquetThriftTypes.h"
#include "velox/type/Type.h"

#include <folly/container/F14Map.h>
#include <cstdint>

namespace facebook::velox::parquet {

struct ColumnPageIndexInformation {
  int64_t offsetIndexOffset{0};

  int32_t offsetIndexLength{0};

  int64_t columnIndexOffset{0};

  int32_t columnIndexLength{0};
};

using PageIndexInfoMap =
    folly::F14FastMap<uint32_t, ColumnPageIndexInformation>;

/// Encapsulates the Parquet ColumnIndex and OffsetIndex for a single column,
/// and allows querying per-page metadata as well as marking pages to skip
/// based on a set of row ranges.
class ColumnPageIndex {
 public:
  /// Construct from a Thrift-deserialized ColumnIndex and OffsetIndex,
  /// plus the total number of rows in the RowGroup.
  ColumnPageIndex(
      thrift::ColumnIndex columnIndex,
      thrift::OffsetIndex offsetIndex,
      uint64_t totalRows)
      : columnIndex_(std::move(columnIndex)),
        offsetIndex_(std::move(offsetIndex)),
        totalRows_(totalRows),
        isPageSkipped_(offsetIndex_.page_locations.size(), false) {
    VELOX_CHECK(
        columnIndex_.max_values.size() == offsetIndex_.page_locations.size());
    // Precompute row counts for each page using OffsetIndex.first_row_index
    size_t n = offsetIndex_.page_locations.size();
    pageRowCount_.reserve(n);
    pageToStreamIdx.reserve(n + 1);
    for (size_t i = 0; i < n; ++i) {
      int64_t start = offsetIndex_.page_locations[i].first_row_index;
      int64_t end =
          (i + 1 < n ? offsetIndex_.page_locations[i + 1].first_row_index
                     : totalRows_);
      pageRowCount_.push_back(end - start);
      pageToStreamIdx.push_back(-1);
    }
    pageToStreamIdx.push_back(-1);
  }

  /// @returns the number of pages in this column chunk.
  size_t numPages() const {
    return offsetIndex_.page_locations.size();
  }

  /// @returns the number of rows in page `i`.
  int64_t pageRowCount(size_t i) const {
    return pageRowCount_.at(i);
  }

  /// @returns the global first-row index for page `i` in the RowGroup.
  int64_t pageFirstRowIndex(size_t i) const {
    return offsetIndex_.page_locations.at(i).first_row_index;
  }

  /// @returns the file byte offset of page `i`.
  int64_t pageOffset(size_t i) const {
    return offsetIndex_.page_locations.at(i).offset;
  }

  /// @returns the compressed page size in bytes for page `i`.
  int32_t compressedPageSize(size_t i) const {
    return offsetIndex_.page_locations.at(i).compressed_page_size;
  }

  /// @returns whether page `i` has been marked to skip.
  bool isPageSkipped(size_t i) const {
    return isPageSkipped_.at(i);
  }

  /// @returns whether any page in this column chunk has been marked to skip.
  bool hasSkippedPages() const {
    return hasSkippedPages_;
  }

  /// Update skip flags for all pages based on the given RowRanges.
  /// Any page whose [firstRow, firstRow+rowCount-1] does not overlap
  /// any range in `ranges` will be marked skipped.
  int32_t updateSkippedPages(const dwio::common::RowRanges& ranges) {
    int32_t skippedPages = 0;
    size_t n = numPages();
    for (size_t i = 0; i < n; ++i) {
      int64_t start = pageFirstRowIndex(i);
      int64_t end = start + pageRowCount(i) - 1;
      bool overlap = ranges.isOverlapping(start, end);
      isPageSkipped_[i] = !overlap;
      hasSkippedPages_ |= isPageSkipped_[i];
      if (isPageSkipped_[i]) {
        skippedPages++;
      }
    }
    return skippedPages;
  }

  /// Builds column statistics for a specific page in a Parquet column chunk..
  std::unique_ptr<dwio::common::ColumnStatistics> buildColumnStatisticsForPage(
      size_t index,
      const velox::Type& type) {
    std::optional<uint64_t> valueCount = columnIndex_.__isset.null_counts
        ? std::optional<uint64_t>(
              pageRowCount_.at(index) - columnIndex_.null_counts.at(index))
        : std::nullopt;
    std::optional<bool> hasNull = columnIndex_.__isset.null_counts
        ? std::optional<bool>(columnIndex_.null_counts.at(index) > 0)
        : std::nullopt;

    switch (type.kind()) {
      case TypeKind::BOOLEAN:
        return std::make_unique<dwio::common::BooleanColumnStatistics>(
            valueCount, hasNull, std::nullopt, std::nullopt, std::nullopt);
      case TypeKind::TINYINT:
        return std::make_unique<dwio::common::IntegerColumnStatistics>(
            valueCount,
            hasNull,
            std::nullopt,
            std::nullopt,
            getMin<int8_t>(index),
            getMax<int8_t>(index),
            std::nullopt);
      case TypeKind::SMALLINT:
        return std::make_unique<dwio::common::IntegerColumnStatistics>(
            valueCount,
            hasNull,
            std::nullopt,
            std::nullopt,
            getMin<int16_t>(index),
            getMax<int16_t>(index),
            std::nullopt);
      case TypeKind::INTEGER:
        return std::make_unique<dwio::common::IntegerColumnStatistics>(
            valueCount,
            hasNull,
            std::nullopt,
            std::nullopt,
            getMin<int32_t>(index),
            getMax<int32_t>(index),
            std::nullopt);
      case TypeKind::BIGINT:
        return std::make_unique<dwio::common::IntegerColumnStatistics>(
            valueCount,
            hasNull,
            std::nullopt,
            std::nullopt,
            getMin<int64_t>(index),
            getMax<int64_t>(index),
            std::nullopt);
      case TypeKind::REAL:
        return std::make_unique<dwio::common::DoubleColumnStatistics>(
            valueCount,
            hasNull,
            std::nullopt,
            std::nullopt,
            getMin<float>(index),
            getMax<float>(index),
            std::nullopt);
      case TypeKind::DOUBLE:
        return std::make_unique<dwio::common::DoubleColumnStatistics>(
            valueCount,
            hasNull,
            std::nullopt,
            std::nullopt,
            getMin<double>(index),
            getMax<double>(index),
            std::nullopt);
      case TypeKind::VARCHAR:
      case TypeKind::VARBINARY:
        return std::make_unique<dwio::common::StringColumnStatistics>(
            valueCount,
            hasNull,
            std::nullopt,
            std::nullopt,
            getMin<std::string>(index),
            getMax<std::string>(index),
            std::nullopt);

      default:
        return std::make_unique<dwio::common::ColumnStatistics>(
            valueCount, hasNull, std::nullopt, std::nullopt);
    }
  }

  void setPageStreamIndex(size_t pageIndex, int32_t streamIndex) {
    if (pageIndex < pageToStreamIdx.size()) {
      pageToStreamIdx[pageIndex] = streamIndex;
    } else {
      VELOX_FAIL(
          "Page index {} out of bounds for pageToStreamIdx {}",
          pageIndex,
          pageToStreamIdx.size());
    }
  }

  int32_t getPageStreamIndex(size_t pageIndex) const {
    if (pageIndex < pageToStreamIdx.size()) {
      return pageToStreamIdx[pageIndex];
    } else {
      VELOX_FAIL(
          "Page index {} out of bounds for pageToStreamIdx {}",
          pageIndex,
          pageToStreamIdx.size());
    }
  }

 private:
  template <typename T>
  inline const T load(const char* ptr) {
    T ret;
    std::memcpy(&ret, ptr, sizeof(ret));
    return ret;
  }

  template <typename T>
  inline std::optional<T> getMin(size_t i) {
    if (columnIndex_.null_pages.at(i)) {
      return std::nullopt;
    }
    if constexpr (std::is_same_v<T, std::string>) {
      return std::optional<T>(columnIndex_.min_values.at(i));
    } else {
      return std::optional<T>(load<T>(columnIndex_.min_values.at(i).data()));
    }
  }

  template <typename T>
  inline std::optional<T> getMax(size_t i) {
    if (columnIndex_.null_pages.at(i)) {
      return std::nullopt;
    }
    if constexpr (std::is_same_v<T, std::string>) {
      return std::optional<T>(columnIndex_.max_values.at(i));
    } else {
      return std::optional<T>(load<T>(columnIndex_.max_values.at(i).data()));
    }
  }

  thrift::ColumnIndex columnIndex_;
  thrift::OffsetIndex offsetIndex_;
  uint64_t totalRows_;

  std::vector<bool> isPageSkipped_;
  std::vector<uint64_t> pageRowCount_;
  bool hasSkippedPages_{false};

  std::vector<int32_t> pageToStreamIdx;
};

} // namespace facebook::velox::parquet
