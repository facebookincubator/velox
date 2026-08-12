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

#include "velox/dwio/parquet/reader/ParquetData.h"

#include "velox/dwio/common/BufferedInput.h"
#include "velox/dwio/parquet/reader/ParquetStatsContext.h"

#include <cmath>
#include <limits>

namespace facebook::velox::parquet {
namespace {

bool filterAcceptsNaN(const common::Filter* filter) {
  if (!filter) {
    return false;
  }
  switch (filter->kind()) {
    case common::FilterKind::kDoubleRange:
    case common::FilterKind::kFloatRange:
    case common::FilterKind::kMultiRange:
      return filter->testDouble(std::numeric_limits<double>::quiet_NaN()) ||
          filter->testFloat(std::numeric_limits<float>::quiet_NaN());
    default:
      return false;
  }
}

PageBoundsCapability pageBoundsCapability(
    const ParquetTypeWithId& type,
    const common::Filter* filter,
    const std::vector<std::unique_ptr<common::Filter>>& metadataFilters,
    bool hasTypeDefinedColumnOrder) {
  if (!hasTypeDefinedColumnOrder || type.isRepeated_) {
    return PageBoundsCapability::kNullabilityOnly;
  }
  if (type.parquetType_ == thrift::Type::INT96 ||
      type.parquetType_ == thrift::Type::FIXED_LEN_BYTE_ARRAY) {
    return PageBoundsCapability::kNullabilityOnly;
  }
  if (type.logicalType_ &&
      (type.logicalType_->getType() == thrift::LogicalType::Type::DATE ||
       type.logicalType_->getType() == thrift::LogicalType::Type::DECIMAL ||
       type.logicalType_->getType() == thrift::LogicalType::Type::TIME ||
       type.logicalType_->getType() == thrift::LogicalType::Type::TIMESTAMP ||
       type.logicalType_->getType() == thrift::LogicalType::Type::JSON ||
       type.logicalType_->getType() == thrift::LogicalType::Type::BSON ||
       type.logicalType_->getType() == thrift::LogicalType::Type::UUID)) {
    return PageBoundsCapability::kNullabilityOnly;
  }
  if (type.convertedType_ &&
      (*type.convertedType_ == thrift::ConvertedType::INT_8 ||
       *type.convertedType_ == thrift::ConvertedType::INT_16 ||
       *type.convertedType_ == thrift::ConvertedType::UINT_8 ||
       *type.convertedType_ == thrift::ConvertedType::UINT_16 ||
       *type.convertedType_ == thrift::ConvertedType::UINT_32 ||
       *type.convertedType_ == thrift::ConvertedType::UINT_64 ||
       *type.convertedType_ == thrift::ConvertedType::TIME_MILLIS ||
       *type.convertedType_ == thrift::ConvertedType::TIME_MICROS ||
       *type.convertedType_ == thrift::ConvertedType::TIMESTAMP_MILLIS ||
       *type.convertedType_ == thrift::ConvertedType::TIMESTAMP_MICROS ||
       *type.convertedType_ == thrift::ConvertedType::DECIMAL ||
       *type.convertedType_ == thrift::ConvertedType::DATE)) {
    return PageBoundsCapability::kNullabilityOnly;
  }
  if (type.logicalType_ &&
      type.logicalType_->getType() == thrift::LogicalType::Type::INTEGER &&
      (!*type.logicalType_->get_INTEGER().isSigned() ||
       *type.logicalType_->get_INTEGER().bitWidth() < 32)) {
    return PageBoundsCapability::kNullabilityOnly;
  }
  if (type.type()->kind() == TypeKind::VARBINARY) {
    return PageBoundsCapability::kNullabilityOnly;
  }
  if ((type.type()->kind() == TypeKind::REAL ||
       type.type()->kind() == TypeKind::DOUBLE) &&
      (filterAcceptsNaN(filter) || [&]() {
        for (const auto& metadataFilter : metadataFilters) {
          if (filterAcceptsNaN(metadataFilter.get())) {
            return true;
          }
        }
        return false;
      }())) {
    return PageBoundsCapability::kNullabilityOnly;
  }
  if (type.parquetType_ != thrift::Type::BOOLEAN &&
      type.parquetType_ != thrift::Type::INT32 &&
      type.parquetType_ != thrift::Type::INT64 &&
      type.parquetType_ != thrift::Type::FLOAT &&
      type.parquetType_ != thrift::Type::DOUBLE &&
      type.parquetType_ != thrift::Type::BYTE_ARRAY) {
    return PageBoundsCapability::kNullabilityOnly;
  }
  return PageBoundsCapability::kOrderedBounds;
}

} // namespace

std::unique_ptr<dwio::common::FormatData> ParquetParams::toFormatData(
    const std::shared_ptr<const dwio::common::TypeWithId>& type,
    const common::ScanSpec& scanSpec) {
  return std::make_unique<ParquetData>(
      type, metaData_, pool(), runtimeStatistics(), sessionTimezone_, scanSpec);
}

ParquetData::ParquetData(
    const std::shared_ptr<const dwio::common::TypeWithId>& type,
    const FileMetaDataPtr fileMetadataPtr,
    memory::MemoryPool& pool,
    dwio::common::ColumnReaderStatistics& stats,
    const tz::TimeZone* sessionTimezone,
    const velox::common::ScanSpec& scanSpec)
    : pool_(pool),
      type_(std::static_pointer_cast<const ParquetTypeWithId>(type)),
      fileMetaDataPtr_(fileMetadataPtr),
      maxDefine_(type_->maxDefine_),
      maxRepeat_(type_->maxRepeat_),
      rowsInRowGroup_(-1),
      stats_(stats),
      sessionTimezone_(sessionTimezone),
      scanSpec_(scanSpec),
      pageIndexFilter_(
          scanSpec.filter() ? scanSpec.filter()->clone() : nullptr) {
  pageIndexMetadataFilters_.reserve(scanSpec.numMetadataFilters());
  for (int i = 0; i < scanSpec.numMetadataFilters(); ++i) {
    auto* filter = scanSpec.metadataFilterAt(i);
    pageIndexMetadataFilters_.push_back(filter ? filter->clone() : nullptr);
  }
}

void ParquetData::filterRowGroups(
    const common::ScanSpec& scanSpec,
    uint64_t /*rowsPerRowGroup*/,
    const dwio::common::StatsContext& writerContext,
    FilterRowGroupsResult& result) {
  auto parquetStatsContext =
      reinterpret_cast<const ParquetStatsContext*>(&writerContext);
  if (type_->parquetType_.has_value() &&
      parquetStatsContext->shouldIgnoreStatistics(
          type_->parquetType_.value())) {
    return;
  }
  result.totalCount =
      std::max<int>(result.totalCount, fileMetaDataPtr_.numRowGroups());
  auto numWords = bits::nwords(result.totalCount);
  if (result.filterResult.size() < numWords) {
    result.filterResult.resize(numWords);
  }
  const auto metadataFiltersStartIndex = result.metadataFilterResults.size();
  for (int i = 0; i < scanSpec.numMetadataFilters(); ++i) {
    result.metadataFilterResults.emplace_back(
        scanSpec.metadataFilterNodeAt(i), std::vector<uint64_t>(numWords));
  }
  if (!scanSpec.filter() && scanSpec.numMetadataFilters() == 0) {
    return;
  }

  for (auto rowGroup = 0; rowGroup < fileMetaDataPtr_.numRowGroups();
       ++rowGroup) {
    if (bits::isBitSet(result.filterResult.data(), rowGroup)) {
      continue;
    }
    if (scanSpec.filter() && !rowGroupMatches(rowGroup, scanSpec.filter())) {
      bits::setBit(result.filterResult.data(), rowGroup);
      continue;
    }
    for (int filter = 0; filter < scanSpec.numMetadataFilters(); ++filter) {
      auto* metadataFilter = scanSpec.metadataFilterAt(filter);
      if (!rowGroupMatches(rowGroup, metadataFilter)) {
        bits::setBit(
            result.metadataFilterResults[metadataFiltersStartIndex + filter]
                .second.data(),
            rowGroup);
      }
    }
  }
}

bool ParquetData::rowGroupMatches(
    uint32_t rowGroupId,
    const common::Filter* filter) {
  if (!filter) {
    return true;
  }
  const auto rowGroup = fileMetaDataPtr_.rowGroup(rowGroupId);
  VELOX_CHECK_GT(rowGroup.numColumns(), 0);
  auto columnChunk = rowGroup.columnChunk(type_->column());
  if (!columnChunk.hasStatistics()) {
    return true;
  }
  auto columnStats = columnChunk.getColumnStatistics(
      type_->type(),
      rowGroup.numRows(),
      type_->convertedType_,
      type_->logicalType_);
  return testFilter(
      filter, columnStats.get(), rowGroup.numRows(), type_->type());
}

bool ParquetData::collectIndexPageInfoMap(
    uint32_t index,
    PageIndexInfoMap& map) {
  const bool hasFilter =
      pageIndexFilter_ != nullptr || !pageIndexMetadataFilters_.empty();
  bool canApplyPagePruning = hasFilter;
  if (canApplyPagePruning) {
    for (auto* parent = type_.get(); parent != nullptr;
         parent = parent->parquetParent()) {
      if (parent->parquetParent() &&
          (parent->type()->kind() == TypeKind::ARRAY ||
           parent->type()->kind() == TypeKind::MAP ||
           parent->type()->kind() == TypeKind::ROW)) {
        canApplyPagePruning = false;
        break;
      }
    }
  }

  const auto chunk =
      fileMetaDataPtr_.rowGroup(index).columnChunk(type_->column());
  if (chunk.hasOffsetIndex() && chunk.offsetIndexLength() > 0) {
    const bool readColumnIndex = canApplyPagePruning &&
        chunk.hasColumnIndex() && chunk.columnIndexLength() > 0;
    map[type_->column()] = {
        chunk.offsetIndexOffset(),
        chunk.offsetIndexLength(),
        chunk.hasColumnIndex() ? chunk.columnIndexOffset() : 0,
        chunk.hasColumnIndex() ? chunk.columnIndexLength() : 0,
        readColumnIndex,
        true,
        readColumnIndex
            ? pageBoundsCapability(
                  *type_,
                  pageIndexFilter_.get(),
                  pageIndexMetadataFilters_,
                  fileMetaDataPtr_.hasTypeDefinedColumnOrder(type_->column()))
            : PageBoundsCapability::kNone,
    };
  }
  return canApplyPagePruning;
}

void ParquetData::evaluatePageIndex(
    const ValidatedPageIndexes& pageIndexes,
    dwio::common::RowIntervalSet& rejectedRows,
    std::vector<std::pair<
        const velox::common::MetadataFilter::LeafNode*,
        dwio::common::RowIntervalSet>>& metadataResults) const {
  const auto acceptsNaN = [&](const common::Filter* filter) {
    if (!filter ||
        (type_->type()->kind() != TypeKind::REAL &&
         type_->type()->kind() != TypeKind::DOUBLE)) {
      return false;
    }
    switch (filter->kind()) {
      case common::FilterKind::kDoubleRange:
      case common::FilterKind::kFloatRange:
      case common::FilterKind::kMultiRange:
        return filter->testDouble(std::numeric_limits<double>::quiet_NaN()) ||
            filter->testFloat(std::numeric_limits<float>::quiet_NaN());
      default:
        return false;
    }
  };
  if (!pageIndexes.column.has_value()) {
    return;
  }
  const auto& pageIndex = *pageIndexes.column;
  const bool hasOrderedBounds =
      pageIndex.capability() == PageBoundsCapability::kOrderedBounds;
  const bool filterMayAcceptNaN = acceptsNaN(pageIndexFilter_.get());
  std::vector<bool> metadataAcceptsNaN;
  metadataAcceptsNaN.reserve(pageIndexMetadataFilters_.size());
  for (const auto& filter : pageIndexMetadataFilters_) {
    metadataAcceptsNaN.push_back(acceptsNaN(filter.get()));
  }

  const auto metadataResultsStart = metadataResults.size();
  for (size_t i = 0; i < pageIndexMetadataFilters_.size(); ++i) {
    metadataResults.emplace_back(
        scanSpec_.metadataFilterNodeAt(i), dwio::common::RowIntervalSet());
  }

  for (size_t page = 0; page < pageIndex.numPages(); ++page) {
    const auto& location = pageIndex.bounds(page);
    const auto& pageLocation = pageIndexes.offset.pages.at(page);
    const auto pageEnd = checkedPlus(
        pageLocation.firstRow, pageLocation.numRows, "page row end");
    const dwio::common::RowInterval pageRows{pageLocation.firstRow, pageEnd};
    const auto nullCount = pageIndex.nullCount(page);
    const bool allNull = pageIndex.isNullPage(page) ||
        (nullCount.has_value() && nullCount.value() == pageLocation.numRows);
    const bool hasNull = pageIndex.isNullPage(page) || !nullCount.has_value() ||
        nullCount.value() > 0;

    const auto canReject = [&](const common::Filter* filter,
                               bool acceptsNan) -> bool {
      if (!filter || (!hasOrderedBounds && !allNull)) {
        return false;
      }
      if (allNull) {
        return !filter->testNull();
      }
      if (acceptsNan &&
          (type_->type()->kind() == TypeKind::REAL ||
           type_->type()->kind() == TypeKind::DOUBLE)) {
        return false;
      }
      if (!location.minimum || !location.maximum) {
        return false;
      }
      return std::visit(
          [&](const auto& minimum) {
            using Value = std::decay_t<decltype(minimum)>;
            if (!std::holds_alternative<Value>(*location.maximum)) {
              return false;
            }
            const auto& maximum = std::get<Value>(*location.maximum);
            if constexpr (std::is_same_v<Value, bool>) {
              return !filter->testBool(minimum) && !filter->testBool(maximum) &&
                  !hasNull;
            } else if constexpr (
                std::is_same_v<Value, int32_t> ||
                std::is_same_v<Value, int64_t>) {
              return !filter->testInt64Range(
                  static_cast<int64_t>(minimum),
                  static_cast<int64_t>(maximum),
                  hasNull);
            } else if constexpr (std::is_same_v<Value, float>) {
              return !filter->testDoubleRange(
                  static_cast<double>(minimum),
                  static_cast<double>(maximum),
                  hasNull);
            } else if constexpr (std::is_same_v<Value, double>) {
              return !filter->testDoubleRange(minimum, maximum, hasNull);
            } else if constexpr (std::is_same_v<Value, std::string>) {
              return !filter->testBytesRange(minimum, maximum, hasNull);
            } else {
              return false;
            }
          },
          *location.minimum);
    };

    if (canReject(pageIndexFilter_.get(), filterMayAcceptNaN)) {
      rejectedRows.add(pageRows);
    }
    for (size_t filter = 0; filter < pageIndexMetadataFilters_.size();
         ++filter) {
      if (canReject(
              pageIndexMetadataFilters_[filter].get(),
              metadataAcceptsNaN[filter])) {
        metadataResults[metadataResultsStart + filter].second.add(pageRows);
      }
    }
  }
}

void ParquetData::enqueueRowGroup(
    uint32_t index,
    dwio::common::BufferedInput& input,
    const RowGroupPagePruningPlanPtr& pagePlan) {
  const auto chunk =
      fileMetaDataPtr_.rowGroup(index).columnChunk(type_->column());
  streams_.resize(fileMetaDataPtr_.numRowGroups());
  plannedStreams_.resize(fileMetaDataPtr_.numRowGroups());
  pagePlans_.resize(fileMetaDataPtr_.numRowGroups());
  pagePlans_[index] = pagePlan;
  VELOX_CHECK(
      chunk.hasMetadata(),
      "ColumnMetaData does not exist for schema Id ",
      type_->column());

  uint64_t chunkReadOffset = chunk.dataPageOffset();
  if (chunk.hasDictionaryPageOffset() && chunk.dictionaryPageOffset() >= 4) {
    chunkReadOffset = chunk.dictionaryPageOffset();
  }
  const auto readSize =
      chunk.compression() == common::CompressionKind::CompressionKind_NONE
      ? chunk.totalUncompressedSize()
      : chunk.totalCompressedSize();
  VELOX_CHECK_GE(readSize, 0, "Negative column chunk size");

  const ColumnPageReadPlan* columnPlan{nullptr};
  if (pagePlan) {
    auto it = pagePlan->columns.find(type_->column());
    if (it != pagePlan->columns.end()) {
      columnPlan = &it->second;
    }
  }
  const auto streamId = dwio::common::StreamIdentifier(type_->column());
  if (!columnPlan || columnPlan->useWholeChunkStream) {
    streams_[index] = input.enqueue(
        {chunkReadOffset, static_cast<uint64_t>(readSize)}, &streamId);
    plannedStreams_[index].reset();
    return;
  }

  if (columnPlan->allPagesSkipped) {
    plannedStreams_[index] = PlannedStreams{};
    streams_[index].reset();
    return;
  }

  PlannedStreams planned;
  if (columnPlan->prefixRegion) {
    planned.prefix = input.enqueue(*columnPlan->prefixRegion, &streamId);
  }
  planned.runs.reserve(columnPlan->retainedRuns.size());
  for (const auto& run : columnPlan->retainedRuns) {
    planned.runs.push_back(input.enqueue(run.region, &streamId));
  }
  plannedStreams_[index] = std::move(planned);
  streams_[index].reset();
}

dwio::common::PositionProvider ParquetData::seekToRowGroup(int64_t index) {
  static std::vector<uint64_t> empty;
  VELOX_CHECK_LT(index, streams_.size());
  if (index > 0) {
    for (int64_t oldIndex = 0; oldIndex < index; ++oldIndex) {
      pagePlans_[oldIndex].reset();
    }
  }
  const auto metadata =
      fileMetaDataPtr_.rowGroup(index).columnChunk(type_->column());
  if (streams_[index] != nullptr) {
    reader_ = std::make_unique<PageReader>(
        std::move(streams_[index]),
        pool_,
        type_,
        metadata.compression(),
        metadata.totalCompressedSize(),
        stats_,
        sessionTimezone_);
    return dwio::common::PositionProvider(empty);
  }

  VELOX_CHECK(
      plannedStreams_[index].has_value(),
      "No planned streams for row group {} column {}",
      index,
      type_->column());
  const auto& pagePlan = pagePlans_.at(index);
  VELOX_CHECK_NOT_NULL(pagePlan);
  const auto planColumn = pagePlan->columns.find(type_->column());
  VELOX_CHECK(
      planColumn != pagePlan->columns.end(),
      "Missing immutable page plan for column {}",
      type_->column());
  auto planned = std::move(*plannedStreams_[index]);
  plannedStreams_[index].reset();
  reader_ = std::make_unique<PageReader>(
      std::move(planned.runs),
      std::move(planned.prefix),
      pool_,
      type_,
      metadata.compression(),
      metadata.totalCompressedSize(),
      stats_,
      sessionTimezone_,
      &planColumn->second);
  return dwio::common::PositionProvider(empty);
}

std::pair<int64_t, int64_t> ParquetData::getRowGroupRegion(
    uint32_t index) const {
  const auto rowGroup = fileMetaDataPtr_.rowGroup(index);
  VELOX_CHECK_GT(rowGroup.numColumns(), 0);
  const auto fileOffset = rowGroup.hasFileOffset() && rowGroup.fileOffset() != 0
      ? rowGroup.fileOffset()
      : rowGroup.columnChunk(0).hasDictionaryPageOffset()
      ? rowGroup.columnChunk(0).dictionaryPageOffset()
      : rowGroup.columnChunk(0).dataPageOffset();
  VELOX_CHECK_GT(fileOffset, 0);
  const auto length = rowGroup.hasTotalCompressedSize()
      ? rowGroup.totalCompressedSize()
      : rowGroup.totalByteSize();
  return {fileOffset, length};
}

} // namespace facebook::velox::parquet
