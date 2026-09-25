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

#include "velox/dwio/nimble/velox/selective/SelectiveNimbleReader.h"

#include <folly/container/F14Set.h>
#include <algorithm>
#include <cmath>
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/legacy/EncodingFactory.h"
#include "velox/dwio/nimble/index/ClusterIndex.h"
#include "velox/dwio/nimble/index/IndexConstants.h"
#include "velox/dwio/nimble/index/IndexFilter.h"
#include "velox/dwio/nimble/index/IndexLookup.h"

#include "velox/common/base/RuntimeMetrics.h"
#include "velox/dwio/common/ScanSpec.h"
#include "velox/dwio/common/Statistics.h"
#include "velox/dwio/nimble/common/FeatureGate.h"
#include "velox/dwio/nimble/velox/SchemaUtils.h"
#include "velox/dwio/nimble/velox/selective/ColumnReader.h"
#include "velox/dwio/nimble/velox/selective/ReaderBase.h"
#include "velox/dwio/nimble/velox/selective/RowSizeTracker.h"
#include "velox/dwio/nimble/velox/selective/SelectiveNimbleIndexReader.h"
#include "velox/serializers/KeyEncoder.h"

namespace facebook::nimble {
namespace detail {

// Provide external initialization code for a selective reader instance.
void initHook();

} // namespace detail

using namespace facebook::velox;

namespace {

Encoding::Options encodingOptions(const TabletReader& tablet) {
  Encoding::Options options;
  options.useVarintRowCount = tablet.properties().compactRowCountEncoding();
  return options;
}

// Converts index column names from nimble schema (internal file names) to
// file schema (user-facing names). The nimble schema and file schema have
// the same structure but potentially different column names due to the
// options.fileSchema() mapping applied during file schema construction.
//
// @param nimbleIndexColumns Index column names from ClusterIndex (nimble
// schema)
// @param nimbleSchema The nimble schema from the file
// @param fileSchema The file schema (table schema with name mapping applied)
// @return Vector of index column names in file schema
std::vector<std::string> convertIndexColumnsToFileSchema(
    const std::vector<std::string>& nimbleIndexColumns,
    const std::shared_ptr<const Type>& nimbleSchema,
    const RowTypePtr& fileSchema) {
  const auto nimbleRowType = asRowType(convertToVeloxType(*nimbleSchema));

  std::vector<std::string> convertedIndexColumns;
  convertedIndexColumns.reserve(nimbleIndexColumns.size());
  for (const auto& nimbleColName : nimbleIndexColumns) {
    const auto colIndex = nimbleRowType->getChildIdxIfExists(nimbleColName);
    NIMBLE_CHECK(
        colIndex.has_value(),
        "Index column '{}' not found in nimble schema: {}",
        nimbleColName,
        nimbleRowType->toString());
    convertedIndexColumns.push_back(fileSchema->nameOf(colIndex.value()));
  }
  return convertedIndexColumns;
}

// Converts nimble SortOrders to velox::core::SortOrders.
// Only converts the first numBoundColumns sort orders, which corresponds to the
// columns that were converted to index bounds (a prefix of all index columns).
std::vector<velox::core::SortOrder> toVeloxSortOrders(
    const std::vector<SortOrder>& sortOrders,
    size_t numBoundColumns) {
  std::vector<velox::core::SortOrder> veloxSortOrders;
  veloxSortOrders.reserve(numBoundColumns);
  for (size_t i = 0; i < numBoundColumns; ++i) {
    veloxSortOrders.push_back(sortOrders[i].toVeloxSortOrder());
  }
  return veloxSortOrders;
}

// Sums the logical sizes of projected top-level columns from the column stats.
// The stats are already recursively rolled up, so each column's logicalSize
// includes all descendants and null overhead. We just need to find the
// projected top-level columns and sum their sizes.
uint64_t sumProjectedLogicalSize(
    const velox::dwio::common::TypeWithId& rootTypeWithId,
    const std::vector<std::unique_ptr<ColumnStatistics>>& columnStats,
    const velox::common::ScanSpec& scanSpec) {
  uint64_t size = 0;
  for (uint32_t i = 0; i < rootTypeWithId.size(); ++i) {
    if (auto* childSpec =
            scanSpec.childByName(rootTypeWithId.type()->asRow().nameOf(i));
        childSpec != nullptr && rootTypeWithId.childAt(i) != nullptr) {
      auto id = rootTypeWithId.childAt(i)->id();
      if (id < columnStats.size() && columnStats[id]) {
        size += columnStats[id]->getLogicalSize();
      }
    }
  }
  return size;
}

// Converts exact-width bounds to Velox's int64 statistics representation.
template <typename T>
bool readIntegerBounds(
    const std::pair<ChunkStatValue, ChunkStatValue>& bounds,
    std::optional<int64_t>& min,
    std::optional<int64_t>& max) {
  const auto* typedMin = std::get_if<T>(&bounds.first);
  const auto* typedMax = std::get_if<T>(&bounds.second);
  if (typedMin == nullptr || typedMax == nullptr) {
    return false;
  }
  min = static_cast<int64_t>(*typedMin);
  max = static_cast<int64_t>(*typedMax);
  return true;
}

// Adapts V2's typed bounds to the statistics objects used by Velox filters.
std::unique_ptr<dwio::common::ColumnStatistics> toColumnStatistics(
    const index::StreamIndex& streamIndex,
    uint32_t chunkIndex,
    uint32_t numChunkRows,
    TypeKind typeKind) {
  const auto nullCount = streamIndex.chunkNullCount(chunkIndex);
  if (nullCount.has_value()) {
    NIMBLE_CHECK_FILE_LE(
        *nullCount, numChunkRows, "Chunk null count exceeds its row count.");
  }
  const std::optional<uint64_t> numValues = nullCount.has_value()
      ? std::make_optional<uint64_t>(numChunkRows - *nullCount)
      : std::nullopt;
  const std::optional<bool> hasNull =
      nullCount.has_value() ? std::make_optional(*nullCount > 0) : std::nullopt;
  const auto bounds = streamIndex.chunkBounds(chunkIndex);

  switch (typeKind) {
    case TypeKind::BIGINT:
    case TypeKind::INTEGER:
    case TypeKind::SMALLINT:
    case TypeKind::TINYINT: {
      std::optional<int64_t> min;
      std::optional<int64_t> max;
      if (bounds.has_value()) {
        const bool hasValidType = [&] {
          switch (typeKind) {
            case TypeKind::TINYINT:
              return readIntegerBounds<int8_t>(*bounds, min, max);
            case TypeKind::SMALLINT:
              return readIntegerBounds<int16_t>(*bounds, min, max);
            case TypeKind::INTEGER:
              return readIntegerBounds<int32_t>(*bounds, min, max);
            case TypeKind::BIGINT:
              return readIntegerBounds<int64_t>(*bounds, min, max);
            default:
              NIMBLE_UNREACHABLE("Expected an integral type.");
          }
        }();
        if (!hasValidType) {
          return nullptr;
        }
      }
      return std::make_unique<dwio::common::IntegerColumnStatistics>(
          numValues,
          hasNull,
          std::nullopt,
          std::nullopt,
          min,
          max,
          std::nullopt);
    }
    case TypeKind::REAL: {
      std::optional<double> min;
      std::optional<double> max;
      if (bounds.has_value()) {
        const auto* typedMin = std::get_if<float>(&bounds->first);
        const auto* typedMax = std::get_if<float>(&bounds->second);
        if (typedMin == nullptr || typedMax == nullptr) {
          return nullptr;
        }
        NIMBLE_CHECK_FILE(
            !std::isnan(*typedMin) && !std::isnan(*typedMax),
            "Chunk bounds must not be NaN.");
        NIMBLE_CHECK_FILE_LE(
            *typedMin, *typedMax, "Chunk minimum must not exceed maximum.");
        min = *typedMin;
        max = *typedMax;
      }
      return std::make_unique<dwio::common::DoubleColumnStatistics>(
          numValues,
          hasNull,
          std::nullopt,
          std::nullopt,
          min,
          max,
          std::nullopt);
    }
    case TypeKind::DOUBLE: {
      std::optional<double> min;
      std::optional<double> max;
      if (bounds.has_value()) {
        const auto* typedMin = std::get_if<double>(&bounds->first);
        const auto* typedMax = std::get_if<double>(&bounds->second);
        if (typedMin == nullptr || typedMax == nullptr) {
          return nullptr;
        }
        NIMBLE_CHECK_FILE(
            !std::isnan(*typedMin) && !std::isnan(*typedMax),
            "Chunk bounds must not be NaN.");
        NIMBLE_CHECK_FILE_LE(
            *typedMin, *typedMax, "Chunk minimum must not exceed maximum.");
        min = *typedMin;
        max = *typedMax;
      }
      return std::make_unique<dwio::common::DoubleColumnStatistics>(
          numValues,
          hasNull,
          std::nullopt,
          std::nullopt,
          min,
          max,
          std::nullopt);
    }
    case TypeKind::BOOLEAN: {
      std::optional<uint64_t> trueCount;
      if (bounds.has_value()) {
        const auto* min = std::get_if<bool>(&bounds->first);
        const auto* max = std::get_if<bool>(&bounds->second);
        if (min == nullptr || max == nullptr) {
          return nullptr;
        }
        if (!*max) {
          trueCount = 0;
        } else if (*min && numValues.has_value()) {
          trueCount = *numValues;
        }
      }
      return std::make_unique<dwio::common::BooleanColumnStatistics>(
          numValues, hasNull, std::nullopt, std::nullopt, trueCount);
    }
    case TypeKind::VARCHAR: {
      std::optional<std::string> min;
      std::optional<std::string> max;
      if (bounds.has_value()) {
        const auto* typedMin = std::get_if<std::string>(&bounds->first);
        const auto* typedMax = std::get_if<std::string>(&bounds->second);
        if (typedMin == nullptr || typedMax == nullptr) {
          return nullptr;
        }
        NIMBLE_CHECK_FILE_LE(
            *typedMin, *typedMax, "Chunk minimum must not exceed maximum.");
        min = *typedMin;
        max = *typedMax;
      }
      return std::make_unique<dwio::common::StringColumnStatistics>(
          numValues,
          hasNull,
          std::nullopt,
          std::nullopt,
          std::move(min),
          std::move(max),
          std::nullopt);
    }
    default:
      return std::make_unique<dwio::common::ColumnStatistics>(
          numValues, hasNull, std::nullopt, std::nullopt);
  }
}

// Collects ranges that a filter proves cannot contain matching rows.
void filterStreamByChunkStats(
    const index::StreamIndex& streamIndex,
    const common::Filter* filter,
    const TypePtr& columnType,
    uint32_t numStripeRows,
    std::vector<RowRange>& ranges) {
  NIMBLE_CHECK_FILE_EQ(
      streamIndex.rowCount(),
      numStripeRows,
      "Chunk rows do not cover the stripe.");
  const auto [startChunk, endChunk] = streamIndex.chunkRange();
  for (uint32_t chunkIndex = startChunk; chunkIndex < endChunk; ++chunkIndex) {
    const uint32_t endRow = streamIndex.chunkEndRow(chunkIndex);
    const uint32_t startRow =
        chunkIndex == startChunk ? 0 : streamIndex.chunkEndRow(chunkIndex - 1);
    NIMBLE_CHECK_FILE_GE(
        endRow, startRow, "Chunk rows must be non-decreasing.");
    if (endRow == startRow) {
      continue;
    }
    auto stats = toColumnStatistics(
        streamIndex, chunkIndex, endRow - startRow, columnType->kind());
    if (stats != nullptr &&
        !common::testFilter(
            filter, stats.get(), endRow - startRow, columnType)) {
      ranges.emplace_back(startRow, endRow);
    }
  }
}

// Merges overlapping ranges from independently filtered columns.
void mergeRowRanges(std::vector<RowRange>& ranges) {
  if (ranges.empty()) {
    return;
  }
  std::sort(ranges.begin(), ranges.end(), [](const auto& lhs, const auto& rhs) {
    return lhs.startRow != rhs.startRow ? lhs.startRow < rhs.startRow
                                        : lhs.endRow < rhs.endRow;
  });
  size_t outputIndex{0};
  for (size_t inputIndex = 1; inputIndex < ranges.size(); ++inputIndex) {
    if (ranges[inputIndex].startRow <= ranges[outputIndex].endRow) {
      ranges[outputIndex].endRow =
          std::max(ranges[outputIndex].endRow, ranges[inputIndex].endRow);
    } else {
      ranges[++outputIndex] = ranges[inputIndex];
    }
  }
  ranges.resize(outputIndex + 1);
}

} // namespace

namespace {

class SelectiveNimbleRowReader : public dwio::common::RowReader {
 public:
  SelectiveNimbleRowReader(
      const std::shared_ptr<ReaderBase>& readerBase,
      const dwio::common::RowReaderOptions& options)
      : lazyIoColumns_{computeLazyIoColumns(options)},
        readerBase_{readerBase},
        options_{options},
        enableStripeStats_{featureGate()->enabled(
            FeatureGate::FeatureSet::kStripeStatsPruning,
            /*defaultValue=*/false)},
        encodingFactory_(
            options.stringDecoderZeroCopy()
                ? std::make_unique<const EncodingFactory>(
                      encodingOptions(readerBase_->tablet()))
                : std::make_unique<const legacy::EncodingFactory>(
                      encodingOptions(readerBase_->tablet()))),
        streams_(readerBase_),
        rowSizeTracker_{
            std::make_unique<RowSizeTracker>(readerBase->fileSchemaWithId())} {
    splitStats_.initColumnStatsCollection(
        *readerBase_->fileSchemaWithId(), options);
    initReadRange();
    initIndexBounds();
    if (options.eagerFirstStripeLoad()) {
      nextRowNumber();
    }
  }

  ~SelectiveNimbleRowReader() override {
    restoreFilters();
  }

  int64_t nextRowNumber() override;

  int64_t nextReadSize(uint64_t size) override;

  uint64_t next(
      uint64_t size,
      VectorPtr& result,
      const dwio::common::Mutation* mutation) override;

  void updateRuntimeStats(dwio::common::RuntimeStats& stats) const override;

  void resetFilterCaches() override;

  std::optional<size_t> estimatedRowSize() const override;

  bool allPrefetchIssued() const final;

  uint32_t currentStripe() const override;

 private:
  // Initializes the stripe range to read based on row offset bounds
  // specified in options. Sets startStripe_ and endStripe_.
  void initReadRange();

  // Loads the current stripe by initializing streams and column readers.
  // Also calls setStripeRowRange() to apply index-based row range filtering.
  void loadCurrentStripe();

  // Initializes index bounds for key-based filtering. Performs a single
  // lookup to get the exact file-level row range and derives stripe
  // boundaries from it.
  void initIndexBounds();

  // Returns true if index bounds are active.
  bool hasIndexBounds() const;

  // Advances to the next stripe by incrementing the stripe index and resetting
  // the row position state.
  void advanceToNextStripe();

  // Sets the row range to read within the current stripe based on cluster
  // index bounds. For the first stripe, adjusts rowInCurrentStripe_ to the
  // lower bound position. For the last stripe, sets endRowInCurrentStripe_ to
  // the upper bound position. Also skips to the starting row position if
  // needed.
  void setStripeRowRange();

  // Updates the random skip tracker when rows are skipped due to index bounds.
  void maybeUpdateRandomSkip(int64_t rowsSkipped);

  // Sets nextRowNumber_ to kAtEnd and restores any filters that were removed
  // during index bound conversion.
  void setAtEnd();

  // Restores filters that were removed during index bound conversion.
  // Called from both setAtEnd() and the destructor.
  void restoreFilters();

  // Computes chunk-statistics skip ranges for the current stripe.
  void computeSkipRowRanges();

  // Advances past skip ranges and returns readable rows before the next one.
  int64_t applySkipRowRanges(int64_t numStripeRows);

  // Refreshes future skip ranges after dynamic filters change.
  void recomputeSkipRowRanges();

  // Computes estimated projected row size from file-level vectorized
  // statistics. Only counts columns in the scan spec. Sets statsBasedRowSize_
  // if stats are available.
  void computeStatsBasedRowSize() const;

  bool skipStripe(uint32_t stripe) const;

  // Computes which top-level columns should use lazy I/O based on the scan
  // spec and remaining filter columns. Returns a const set used for the
  // lifetime of this reader.
  static folly::F14FastSet<std::string> computeLazyIoColumns(
      const dwio::common::RowReaderOptions& options);

  // Precomputed set of top-level column names that should use lazy I/O.
  // Const after construction — computed once per split.
  const folly::F14FastSet<std::string> lazyIoColumns_;

  const std::shared_ptr<ReaderBase> readerBase_;
  const dwio::common::RowReaderOptions options_;
  // Reader-side killswitch, evaluated once at construction (not per stripe):
  // FeatureGate maps to JK dwio/nimble:enable_stripe_stats_pruning. When false,
  // skipStripe does no pruning and every stripe is read.
  const bool enableStripeStats_;
  const std::unique_ptr<const EncodingFactory> encodingFactory_;
  StripeStreams streams_;
  std::vector<int64_t> stripeRowOffsets_;
  // The inclusive lower bound of the stripe range to read.
  int32_t startStripe_{};
  // The exclusive upper bound of the stripe range to read.
  int32_t endStripe_{};

  // Index related fields.
  const ClusterIndex* clusterIndex_{nullptr};
  // File-level row range from index lookup, if index bounds are active.
  std::optional<RowRange> indexRowRange_;

  // The current stripe being read.
  int32_t currentStripe_{};
  // The current row position within the current stripe (0-based).
  int64_t rowInCurrentStripe_{};
  // Optional end row position for the current stripe, set when reading the
  // last stripe with upper index bounds.
  std::optional<int64_t> endRowInCurrentStripe_;
  std::optional<int64_t> nextRowNumber_;

  int32_t skippedStripes_{0};
  // Tracks the number of rows from trailing stripes filtered out by upper index
  // bound. These rows are combined with rows skipped at the end of the last
  // stripe when updating the random skip tracker.
  int64_t trailingSkippedRows_{0};

  std::unique_ptr<dwio::common::SelectiveColumnReader> columnReader_;
  dwio::common::SplitStats splitStats_{dwio::common::FileFormat::NIMBLE};
  std::unique_ptr<RowSizeTracker> rowSizeTracker_;

  // Sorted, non-overlapping row ranges to skip in the current stripe.
  std::vector<RowRange> skipRanges_;
  // Identifies the next range that can affect the current read position.
  size_t skipRangeIndex_{0};
  // Excludes rows skipped by index bounds or random sampling.
  int64_t chunkStatsSkippedRows_{0};
  // Excludes rows outside the split and index bounds.
  int64_t chunkStatsProcessedRows_{0};
  // Preserves the current effective range before stripe completion.
  int64_t currentStripeRows_{0};

  // Cached row size estimate derived from file-level statistics.
  mutable std::optional<size_t> statsBasedRowSize_;
  // Whether we've already attempted to compute statsBasedRowSize_.
  mutable bool statsBasedRowSizeAttempted_{false};

  // Filters that were removed from the scan spec during index bound conversion.
  // These need to be restored when the row reader is destroyed, as the scan
  // spec may be shared across multiple split readers.
  std::vector<std::pair<std::string, std::shared_ptr<common::Filter>>>
      filtersToRestore_;
};

int64_t SelectiveNimbleRowReader::nextRowNumber() {
  if (nextRowNumber_.has_value()) {
    return *nextRowNumber_;
  }
  while (currentStripe_ < endStripe_) {
    auto numStripeRows = readerBase_->tablet().stripeRowCount(currentStripe_);
    if (rowInCurrentStripe_ == 0) {
      // Apply random skip at stripe level (before loading the stripe).
      if (readerBase_->randomSkip() &&
          readerBase_->randomSkip()->nextSkip() >= numStripeRows) {
        readerBase_->randomSkip()->consume(numStripeRows);
        ++skippedStripes_;
        advanceToNextStripe();
        continue;
      }
      if (skipStripe(currentStripe_)) {
        maybeUpdateRandomSkip(numStripeRows);
        ++skippedStripes_;
        advanceToNextStripe();
        continue;
      }
      loadCurrentStripe();
      computeSkipRowRanges();
    }
    if (endRowInCurrentStripe_.has_value()) {
      NIMBLE_CHECK_LE(endRowInCurrentStripe_.value(), numStripeRows);
      numStripeRows = endRowInCurrentStripe_.value();
    }
    applySkipRowRanges(numStripeRows);
    if (rowInCurrentStripe_ < numStripeRows) {
      nextRowNumber_ = stripeRowOffsets_[currentStripe_] + rowInCurrentStripe_;
      return *nextRowNumber_;
    }
    advanceToNextStripe();
  }
  // Update random skip tracker for trailing rows that were skipped due to upper
  // index bound filtering (includes both entire trailing stripes and rows at
  // the end of the last stripe).
  maybeUpdateRandomSkip(trailingSkippedRows_);

  setAtEnd();
  return kAtEnd;
}

void SelectiveNimbleRowReader::advanceToNextStripe() {
  chunkStatsProcessedRows_ += currentStripeRows_;
  ++currentStripe_;
  rowInCurrentStripe_ = 0;
  endRowInCurrentStripe_.reset();
  skipRanges_.clear();
  skipRangeIndex_ = 0;
  currentStripeRows_ = 0;
}

uint32_t SelectiveNimbleRowReader::currentStripe() const {
  return static_cast<uint32_t>(currentStripe_);
}

int64_t SelectiveNimbleRowReader::nextReadSize(uint64_t size) {
  NIMBLE_DCHECK_GT(size, 0, "Read size must be greater than 0");
  while (true) {
    if (nextRowNumber() == kAtEnd) {
      return kAtEnd;
    }
    int64_t numStripeRows =
        readerBase_->tablet().stripeRowCount(currentStripe_);
    if (endRowInCurrentStripe_.has_value()) {
      numStripeRows = endRowInCurrentStripe_.value();
    }
    const auto readableRows = applySkipRowRanges(numStripeRows);
    if (rowInCurrentStripe_ >= numStripeRows) {
      nextRowNumber_.reset();
      continue;
    }
    const auto rowsToRead = std::min(size, static_cast<uint64_t>(readableRows));
    NIMBLE_DCHECK_GT(rowsToRead, 0);
    return static_cast<int64_t>(rowsToRead);
  }
}

uint64_t SelectiveNimbleRowReader::next(
    uint64_t size,
    VectorPtr& result,
    const dwio::common::Mutation* mutation) {
  const auto rowsToRead = nextReadSize(size);
  if (rowsToRead == kAtEnd) {
    return 0;
  }
  columnReader_->setCurrentRowNumber(nextRowNumber());
  if (options_.rowNumberColumnInfo().has_value()) {
    readWithRowNumber(
        columnReader_, options_, nextRowNumber(), rowsToRead, mutation, result);
  } else {
    columnReader_->next(rowsToRead, result, mutation);
  }
  nextRowNumber_.reset();
  rowInCurrentStripe_ += rowsToRead;
  return rowsToRead;
}

void SelectiveNimbleRowReader::updateRuntimeStats(
    dwio::common::RuntimeStats& stats) const {
  stats.skippedStrides += skippedStripes_;
  const auto& tabletStats = readerBase_->tablet().stats();
  stats.footerBufferOverread += tabletStats.footerBufferOverread;
  stats.footerBufferUnderread += tabletStats.footerBufferUnderread;
  stats.footerCacheHit += tabletStats.footerCacheHit ? 1 : 0;
  stats.chunkStatsSkippedRows += chunkStatsSkippedRows_;
  stats.chunkStatsProcessedRows += chunkStatsProcessedRows_;
  if (currentStripe_ < endStripe_) {
    stats.chunkStatsProcessedRows += currentStripeRows_;
  }
}

void SelectiveNimbleRowReader::resetFilterCaches() {
  if (columnReader_) {
    columnReader_->resetFilterCaches();
    if (currentStripe_ < endStripe_) {
      recomputeSkipRowRanges();
    }
  }
}

void SelectiveNimbleRowReader::computeSkipRowRanges() {
  skipRanges_.clear();
  skipRangeIndex_ = 0;
  if (readerBase_->randomSkip()) {
    return;
  }

  const auto* scanSpec = options_.scanSpec().get();
  if (columnReader_ == nullptr || scanSpec == nullptr) {
    return;
  }

  const auto& nimbleRoot = readerBase_->nimbleSchema()->asRow();
  const auto& rootType = readerBase_->fileSchemaWithId();
  const auto numStripeRows =
      readerBase_->tablet().stripeRowCount(currentStripe_);
  for (uint32_t i = 0; i < rootType->size() && i < nimbleRoot.childrenCount();
       ++i) {
    const auto* childSpec =
        scanSpec->childByName(rootType->type()->asRow().nameOf(i));
    if (childSpec == nullptr || childSpec->filter() == nullptr ||
        !childSpec->readFromFile() || childSpec->hasTransform() ||
        !nimbleRoot.childAt(i)->isScalar()) {
      continue;
    }
    const auto streamId =
        nimbleRoot.childAt(i)->asScalar().scalarDescriptor().offset();
    const auto streamIndex = streams_.streamIndex(streamId);
    if (streamIndex == nullptr) {
      continue;
    }
    filterStreamByChunkStats(
        *streamIndex,
        childSpec->filter(),
        rootType->childAt(i)->type(),
        numStripeRows,
        skipRanges_);
  }
  mergeRowRanges(skipRanges_);
}

int64_t SelectiveNimbleRowReader::applySkipRowRanges(int64_t numStripeRows) {
  NIMBLE_CHECK_GE(numStripeRows, 0);
  while (skipRangeIndex_ < skipRanges_.size()) {
    const auto currentRow = static_cast<uint32_t>(rowInCurrentStripe_);
    const auto& skipRange = skipRanges_[skipRangeIndex_];
    if (skipRange.endRow <= currentRow) {
      ++skipRangeIndex_;
      continue;
    }
    if (currentRow < skipRange.startRow) {
      return std::min<int64_t>(skipRange.startRow, numStripeRows) - currentRow;
    }

    const auto effectiveSkipEnd = static_cast<uint32_t>(
        std::min<int64_t>(skipRange.endRow, numStripeRows));
    chunkStatsSkippedRows_ += effectiveSkipEnd - currentRow;
    rowInCurrentStripe_ = effectiveSkipEnd;
    nextRowNumber_.reset();
    ++skipRangeIndex_;
    if (effectiveSkipEnd < numStripeRows) {
      columnReader_->seekTo(effectiveSkipEnd, /*readsNullsOnly=*/false);
    }
  }
  return numStripeRows - rowInCurrentStripe_;
}

void SelectiveNimbleRowReader::recomputeSkipRowRanges() {
  nextRowNumber_.reset();
  const auto currentRow = static_cast<uint32_t>(rowInCurrentStripe_);
  computeSkipRowRanges();
  const auto firstFutureRange = std::lower_bound(
      skipRanges_.begin(),
      skipRanges_.end(),
      currentRow,
      [](const auto& range, uint32_t row) { return range.endRow <= row; });
  skipRanges_.erase(skipRanges_.begin(), firstFutureRange);
  skipRangeIndex_ = 0;
  if (!skipRanges_.empty() && skipRanges_.front().startRow < currentRow) {
    skipRanges_.front().startRow = currentRow;
  }
}

void SelectiveNimbleRowReader::computeStatsBasedRowSize() const {
  if (statsBasedRowSizeAttempted_) {
    return;
  }
  statsBasedRowSizeAttempted_ = true;
  const auto& columnStats = readerBase_->fileColumnStats();
  if (columnStats.empty()) {
    return;
  }
  auto totalLogicalSize = sumProjectedLogicalSize(
      *readerBase_->fileSchemaWithId(), columnStats, *options_.scanSpec());
  auto totalRows = readerBase_->tablet().tabletRowCount();
  if (totalRows > 0) {
    statsBasedRowSize_ = std::max<size_t>(1, totalLogicalSize / totalRows);
  }
}

bool SelectiveNimbleRowReader::skipStripe(uint32_t stripe) const {
  // Reader-side killswitch: when disabled, read every stripe (no pruning).
  if (!enableStripeStats_) {
    return false;
  }
  const auto& stripeStats = readerBase_->stripeColumnStats();
  if (stripe >= stripeStats.size() || stripeStats[stripe].empty()) {
    return false;
  }
  const auto& rootType = *readerBase_->fileSchemaWithId();
  const auto& rowType = rootType.type()->asRow();
  const auto stableChildren = options_.scanSpec()->stableChildren();
  for (const auto& childSpec : *stableChildren) {
    if (!childSpec->hasFilter() || childSpec->filter() == nullptr ||
        childSpec->isConstant() || !childSpec->readFromFile()) {
      continue;
    }
    const auto columnIndex =
        rowType.getChildIdxIfExists(childSpec->fieldName());
    if (!columnIndex.has_value()) {
      continue;
    }
    const auto& childType = rootType.childAt(columnIndex.value());
    if (childType == nullptr) {
      continue;
    }
    // Prune top-level scalar columns whose per-stripe min/max prove the filter
    // cannot match. Integral (incl. DATE, which is represented as INTEGER),
    // floating-point, and string/bytes columns are supported; the writer emits
    // min/max for all three in the stripe-stats section. Other kinds (map,
    // array, row, timestamp) are handled by other pruning paths or not at all.
    // Explicit comparisons are used instead of a switch over TypeKind to avoid
    // -Wswitch-enum requiring every enumerator to be listed.
    const auto kind = childType->type()->kind();

    // Per-stripe stats are laid out by schema type id: the writer snapshots
    // statsCollectors_ (indexed by TypeWithId::id()) in order, so
    // stripeStats[stripe][id] matches childType->id() just like the file-level
    // column stats. The bound check below guards against a stats section that
    // covers fewer columns than the current schema.
    const auto columnId = childType->id();
    if (columnId >= stripeStats[stripe].size() ||
        !stripeStats[stripe][columnId]) {
      continue;
    }
    const auto& stripeStat = stripeStats[stripe][columnId];
    const auto* filter = childSpec->filter();

    if (kind == TypeKind::TINYINT || kind == TypeKind::SMALLINT ||
        kind == TypeKind::INTEGER || kind == TypeKind::BIGINT) {
      const auto* stats = stripeStat->as<IntegralStatistics>();
      if (stats == nullptr || !stats->getMin().has_value() ||
          !stats->getMax().has_value()) {
        continue;
      }
      if (!filter->testInt64Range(
              *stats->getMin(), *stats->getMax(), stats->getNullCount() > 0)) {
        return true;
      }
    } else if (kind == TypeKind::REAL || kind == TypeKind::DOUBLE) {
      const auto* stats = stripeStat->as<FloatingPointStatistics>();
      if (stats == nullptr || !stats->getMin().has_value() ||
          !stats->getMax().has_value()) {
        continue;
      }
      if (!filter->testDoubleRange(
              *stats->getMin(), *stats->getMax(), stats->getNullCount() > 0)) {
        return true;
      }
    } else if (kind == TypeKind::VARCHAR || kind == TypeKind::VARBINARY) {
      const auto* stats = stripeStat->as<StringStatistics>();
      if (stats == nullptr) {
        continue;
      }
      const auto min = stats->getMin();
      const auto max = stats->getMax();
      if (!min.has_value() || !max.has_value()) {
        continue;
      }
      if (!filter->testBytesRange(
              std::string_view(*min),
              std::string_view(*max),
              stats->getNullCount() > 0)) {
        return true;
      }
    }
  }
  return false;
}

std::optional<size_t> SelectiveNimbleRowReader::estimatedRowSize() const {
  if (const_cast<SelectiveNimbleRowReader*>(this)->nextRowNumber() == kAtEnd) {
    return std::nullopt;
  }

  computeStatsBasedRowSize();
  if (statsBasedRowSize_.has_value()) {
    return statsBasedRowSize_;
  }

  size_t byteSize, rowCount;
  if (!columnReader_->estimateMaterializedSize(byteSize, rowCount)) {
    return options_.trackRowSize() ? rowSizeTracker_->getCurrentMaxRowSize()
                                   : 1UL << 20;
  }
  return rowCount == 0 ? 0 : byteSize / rowCount;
}

bool SelectiveNimbleRowReader::allPrefetchIssued() const {
  return true;
}

void SelectiveNimbleRowReader::initReadRange() {
  const auto& tablet = readerBase_->tablet();
  startStripe_ = tablet.stripeCount();
  endStripe_ = 0;
  int64_t numRows = 0;
  stripeRowOffsets_.resize(tablet.stripeCount());
  const auto low = options_.offset();
  const auto high = options_.limit();
  for (int i = 0; i < tablet.stripeCount(); ++i) {
    stripeRowOffsets_[i] = numRows;
    if (low <= tablet.stripeOffset(i) && tablet.stripeOffset(i) < high) {
      startStripe_ = std::min(startStripe_, i);
      endStripe_ = std::max(endStripe_, i + 1);
    }
    numRows += tablet.stripeRowCount(i);
  }
  // Initialize actual read boundaries to split boundaries.
  currentStripe_ = startStripe_;
  rowInCurrentStripe_ = 0;
}

// Computes the set of top-level columns eligible for lazy I/O. A column
// qualifies if it has no pushdown filter, is not referenced by the remaining
// filter, and is projected in the output. Returns empty set when lazy I/O is
// disabled.
folly::F14FastSet<std::string> SelectiveNimbleRowReader::computeLazyIoColumns(
    const dwio::common::RowReaderOptions& options) {
  folly::F14FastSet<std::string> lazyIoColumns;
  if (!options.lazyColumnIo()) {
    return lazyIoColumns;
  }
  auto* scanSpec = options.scanSpec().get();
  VELOX_CHECK_NOT_NULL(scanSpec);
  const auto& remainingFilterColumns = options.remainingFilterColumns();
  const auto stableChildren = scanSpec->stableChildren();
  for (const auto& childSpec : *stableChildren) {
    if (childSpec->isConstant() || !childSpec->readFromFile()) {
      continue;
    }
    const auto& name = childSpec->fieldName();
    NIMBLE_DCHECK(
        childSpec->hasFilter() || childSpec->projectOut(),
        "Column with no filter should be projected");
    // Being lazy only defers a column's stream I/O; it does not change which
    // loader decodes the column, so transform and delta columns are eligible
    // too, not just plain projected ones.
    if (!childSpec->hasFilter() && remainingFilterColumns.count(name) == 0 &&
        childSpec->projectOut()) {
      lazyIoColumns.insert(name);
    }
  }
  return lazyIoColumns;
}

void SelectiveNimbleRowReader::loadCurrentStripe() {
  addThreadLocalRuntimeStat(kNumStripeLoads, velox::RuntimeCounter(1));

  streams_.setStripe(currentStripe_);
  NimbleParams params(
      *readerBase_->pool(),
      splitStats_,
      readerBase_->nimbleSchema(),
      streams_,
      options_.trackRowSize() ? rowSizeTracker_.get() : nullptr,
      *encodingFactory_,
      options_.stringDecoderZeroCopy(),
      options_.preserveFlatMapsInMemory(),
      options_.nimblePreserveDictionaryEncoding(),
      lazyIoColumns_.empty() ? nullptr : &lazyIoColumns_);

  columnReader_ = buildColumnReader(
      options_.requestedType() ? options_.requestedType()
                               : readerBase_->fileSchema(),
      readerBase_->fileSchemaWithId(),
      params,
      *options_.scanSpec(),
      true);
  rowSizeTracker_->finalizeProjection();
  columnReader_->setIsTopLevel();
  streams_.load();
  setStripeRowRange();
  const int64_t numStripeRows =
      readerBase_->tablet().stripeRowCount(currentStripe_);
  const int64_t stripeEnd = endRowInCurrentStripe_.value_or(numStripeRows);
  NIMBLE_CHECK_LE(rowInCurrentStripe_, stripeEnd);
  currentStripeRows_ = stripeEnd - rowInCurrentStripe_;
}

bool SelectiveNimbleRowReader::hasIndexBounds() const {
  return indexRowRange_.has_value();
}

void SelectiveNimbleRowReader::initIndexBounds() {
  NIMBLE_CHECK_NULL(clusterIndex_);
  if (currentStripe_ >= endStripe_) {
    return;
  }

  if (!options_.indexEnabled()) {
    return;
  }

  if (!readerBase_->tablet().hasClusterIndex()) {
    return;
  }

  auto* clusterIndex = readerBase_->tablet().clusterIndex();
  NIMBLE_CHECK_NOT_NULL(clusterIndex);

  const auto indexColumns = convertIndexColumnsToFileSchema(
      clusterIndex->indexColumns(),
      readerBase_->nimbleSchema(),
      readerBase_->fileSchema());

  const auto& sortOrders = clusterIndex->sortOrders();
  auto result = convertFilterToIndexBounds(
      indexColumns,
      sortOrders,
      readerBase_->fileSchema(),
      *options_.scanSpec(),
      readerBase_->pool());

  if (!result.has_value()) {
    return;
  }

  filtersToRestore_ = std::move(result->removedFilters);
  options_.scanSpec()->resetCachedValues(/*doReorder=*/false);

  clusterIndex_ = clusterIndex;

  addThreadLocalRuntimeStat(
      kNumIndexFilterConversions,
      velox::RuntimeCounter(result->indexBounds.indexColumns.size()));

  auto keyEncoder = velox::serializer::KeyEncoder::create(
      result->indexBounds.indexColumns,
      asRowType(result->indexBounds.type()),
      toVeloxSortOrders(sortOrders, result->indexBounds.indexColumns.size()),
      readerBase_->pool());
  auto encodedBounds = keyEncoder->encodeIndexBounds(result->indexBounds);
  NIMBLE_CHECK_EQ(
      encodedBounds.size(),
      1,
      "Expected single encoded bounds, got {}",
      encodedBounds.size());

  // Single lookup to get the exact file-level row range.
  const auto lookupResult = clusterIndex_->lookup(
      index::IndexLookup::LookupRequest::rangeScan({encodedBounds}));
  const auto rowRanges = lookupResult[0];
  if (rowRanges.empty()) {
    int64_t rowsSkipped = 0;
    for (int stripe = startStripe_; stripe < endStripe_; ++stripe) {
      rowsSkipped += readerBase_->tablet().stripeRowCount(stripe);
    }
    maybeUpdateRandomSkip(rowsSkipped);
    startStripe_ = endStripe_;
    currentStripe_ = startStripe_;
    return;
  }

  NIMBLE_CHECK_EQ(rowRanges.size(), 1, "Expected single row range per lookup");
  indexRowRange_ = rowRanges[0];
  const auto& rowRange = indexRowRange_.value();

  const auto& tablet = readerBase_->tablet();
  const auto originalStartStripe = startStripe_;
  const auto originalEndStripe = endStripe_;

  // Derive stripe range from the file-level row range.
  startStripe_ = std::max(
      startStripe_, static_cast<int>(tablet.rowToStripe(rowRange.startRow)));
  endStripe_ = std::min(
      endStripe_,
      static_cast<int>(tablet.rowToStripe(rowRange.endRow - 1) + 1));

  // Update random skip tracker for leading skipped stripes.
  if (startStripe_ > originalStartStripe) {
    int64_t rowsSkipped = 0;
    for (int stripe = originalStartStripe; stripe < startStripe_; ++stripe) {
      rowsSkipped += tablet.stripeRowCount(stripe);
    }
    maybeUpdateRandomSkip(rowsSkipped);
  }

  // Track trailing skipped stripes.
  if (endStripe_ < originalEndStripe) {
    for (int stripe = endStripe_; stripe < originalEndStripe; ++stripe) {
      trailingSkippedRows_ += tablet.stripeRowCount(stripe);
    }
  }

  currentStripe_ = startStripe_;
}

void SelectiveNimbleRowReader::maybeUpdateRandomSkip(int64_t rowsSkipped) {
  if (readerBase_->randomSkip() && rowsSkipped > 0) {
    const auto skip = readerBase_->randomSkip()->nextSkip();
    readerBase_->randomSkip()->consume(
        std::min(static_cast<int64_t>(skip), rowsSkipped));
  }
}

void SelectiveNimbleRowReader::restoreFilters() {
  // Restore filters that were removed during index bound conversion.
  // This is needed because the scan spec may be shared across multiple
  // split readers in multiple split execution modes.
  NIMBLE_CHECK(
      filtersToRestore_.empty() || clusterIndex_ != nullptr,
      "filtersToRestore_ should only be set when index bounds are used");
  for (auto& [columnName, filter] : filtersToRestore_) {
    auto* childSpec = options_.scanSpec()->childByName(columnName);
    if (childSpec != nullptr) {
      childSpec->setFilter(std::move(filter));
    }
  }
  filtersToRestore_.clear();
  options_.scanSpec()->resetCachedValues(/*doReorder=*/false);
}

void SelectiveNimbleRowReader::setAtEnd() {
  restoreFilters();
  nextRowNumber_ = kAtEnd;
}

void SelectiveNimbleRowReader::setStripeRowRange() {
  if (!hasIndexBounds()) {
    return;
  }
  // Only the first and last stripes need row-level narrowing.
  if (currentStripe_ != startStripe_ && currentStripe_ != endStripe_ - 1) {
    return;
  }

  const auto& rowRange = indexRowRange_.value();
  const auto stripeStart = stripeRowOffsets_[currentStripe_];
  const auto numStripeRows =
      readerBase_->tablet().stripeRowCount(currentStripe_);

  // For the first stripe, narrow the start row.
  if (currentStripe_ == startStripe_ && rowRange.startRow > stripeStart) {
    rowInCurrentStripe_ = rowRange.startRow - stripeStart;
    maybeUpdateRandomSkip(rowInCurrentStripe_);
  }

  // For the last stripe, narrow the end row.
  if (currentStripe_ == endStripe_ - 1) {
    const auto stripeEnd = stripeStart + numStripeRows;
    if (rowRange.endRow < stripeEnd) {
      endRowInCurrentStripe_ = rowRange.endRow - stripeStart;
      trailingSkippedRows_ += stripeEnd - rowRange.endRow;
    }
  }

  // Skip to the starting row position within the stripe.
  if (rowInCurrentStripe_ > 0 &&
      (!endRowInCurrentStripe_.has_value() ||
       endRowInCurrentStripe_.value() > rowInCurrentStripe_)) {
    columnReader_->seekTo(rowInCurrentStripe_, /*readsNullsOnly=*/false);
  }
}

class SelectiveNimbleReader : public dwio::common::Reader {
 public:
  SelectiveNimbleReader(
      std::unique_ptr<dwio::common::BufferedInput> input,
      const dwio::common::ReaderOptions& options)
      : readerBase_(ReaderBase::create(std::move(input), options)),
        options_(options) {
    detail::initHook();
  }

  std::optional<uint64_t> numberOfRows() const override;

  std::unique_ptr<dwio::common::ColumnStatistics> columnStatistics(
      uint32_t index) const override;

  const RowTypePtr& rowType() const override;

  const std::shared_ptr<const dwio::common::TypeWithId>& typeWithId()
      const override;

  std::unique_ptr<dwio::common::RowReader> createRowReader(
      const dwio::common::RowReaderOptions& options) const override;

  std::unique_ptr<dwio::common::IndexReader> createIndexReader(
      const dwio::common::RowReaderOptions& options) const override;

 private:
  void validateOmittedKeyColumnStorageAccess(
      const dwio::common::RowReaderOptions& options) const;

  const std::shared_ptr<ReaderBase> readerBase_;
  const dwio::common::ReaderOptions options_;
};

std::optional<uint64_t> SelectiveNimbleReader::numberOfRows() const {
  return readerBase_->tablet().tabletRowCount();
}

std::unique_ptr<dwio::common::ColumnStatistics>
SelectiveNimbleReader::columnStatistics(uint32_t index) const {
  const auto& stats = readerBase_->fileColumnStats();
  // Return nullptr if the index is out of range (e.g., no vectorized stats
  // section) or if the column has no statistics (e.g., unsupported type).
  if (index >= stats.size() || !stats[index]) {
    return nullptr;
  }
  return stats[index]->toCommonStatistics();
}

const RowTypePtr& SelectiveNimbleReader::rowType() const {
  return readerBase_->fileSchema();
}

const std::shared_ptr<const dwio::common::TypeWithId>&
SelectiveNimbleReader::typeWithId() const {
  return readerBase_->fileSchemaWithId();
}

std::unique_ptr<dwio::common::RowReader> SelectiveNimbleReader::createRowReader(
    const dwio::common::RowReaderOptions& options) const {
  validateOmittedKeyColumnStorageAccess(options);
  return std::make_unique<SelectiveNimbleRowReader>(readerBase_, options);
}

std::unique_ptr<dwio::common::IndexReader>
SelectiveNimbleReader::createIndexReader(
    const dwio::common::RowReaderOptions& options) const {
  validateOmittedKeyColumnStorageAccess(options);
  // Empty files (no stripes) have no cluster index even when index config
  // is set. Return nullptr so the caller handles it as no-index.
  if (readerBase_->tablet().clusterIndex() == nullptr) {
    return nullptr;
  }
  return std::make_unique<SelectiveNimbleIndexReader>(readerBase_, options);
}

void SelectiveNimbleReader::validateOmittedKeyColumnStorageAccess(
    const dwio::common::RowReaderOptions& options) const {
  const auto& properties = readerBase_->tablet().properties();
  if (!properties.clusterIndexKeyColumnStorageOmitted()) {
    return;
  }

  const auto outputType = options.requestedType() ? options.requestedType()
                                                  : readerBase_->fileSchema();
  const auto* scanSpec = options.scanSpec().get();
  for (const auto& column :
       properties.clusterIndexKeyColumnsWithOmittedStorage()) {
    NIMBLE_USER_CHECK(
        !outputType->containsChild(column),
        "Cluster index key column '{}' cannot be projected because this file stores it only in the cluster index key stream",
        column);
    if (scanSpec != nullptr) {
      const auto* childSpec = scanSpec->childByName(column);
      NIMBLE_USER_CHECK(
          childSpec == nullptr || !childSpec->hasFilter(),
          "Cluster index key column '{}' cannot be used as a remaining scan filter because this file stores it only in the cluster index key stream",
          column);
    }
  }
}

} // namespace

std::unique_ptr<dwio::common::Reader>
SelectiveNimbleReaderFactory::createReader(
    std::unique_ptr<dwio::common::BufferedInput> input,
    const dwio::common::ReaderOptions& options) {
  return std::make_unique<SelectiveNimbleReader>(std::move(input), options);
}

} // namespace facebook::nimble
