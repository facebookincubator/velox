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
#include "velox/dwio/parquet/common/ParquetBloomFilter.h"
#include "velox/dwio/parquet/reader/ParquetStatsContext.h"

namespace facebook::velox::parquet {

using thrift::RowGroup;

namespace {
bool isFilterRangeCoversStatsRange(
    common::Filter* filter,
    dwio::common::ColumnStatistics* stats,
    const TypePtr& type) {
  switch (type->kind()) {
    case TypeKind::BIGINT:
    case TypeKind::INTEGER:
    case TypeKind::SMALLINT:
    case TypeKind::TINYINT: {
      auto intStats =
          dynamic_cast<dwio::common::IntegerColumnStatistics*>(stats);
      if (!intStats)
        return false;

      int64_t min =
          intStats->getMinimum().value_or(std::numeric_limits<int64_t>::min());
      int64_t max =
          intStats->getMaximum().value_or(std::numeric_limits<int64_t>::max());

      switch (filter->kind()) {
        case common::FilterKind::kBigintRange:
          return static_cast<common::BigintRange*>(filter)->lower() <= min &&
              max <= static_cast<common::BigintRange*>(filter)->upper();
        case common::FilterKind::kBigintMultiRange: {
          common::BigintMultiRange* multiRangeFilter =
              static_cast<common::BigintMultiRange*>(filter);
          auto numRanges = multiRangeFilter->ranges().size();
          if (numRanges > 0) {
            return multiRangeFilter->ranges()[0]->lower() <= min &&
                max <= multiRangeFilter->ranges()[numRanges - 1]->upper();
          }
        } break;
        default:
          return false;
      }
    } break;
    default:
      return false;
  }
  return false;
}
} // namespace

std::unique_ptr<dwio::common::FormatData> ParquetParams::toFormatData(
    const std::shared_ptr<const dwio::common::TypeWithId>& type,
    const common::ScanSpec& /*scanSpec*/) {
  return std::make_unique<ParquetData>(
      type, metaData_, pool(), sessionTimezone_, parquetReadBloomFilter_);
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
  auto nwords = bits::nwords(result.totalCount);
  if (result.filterResult.size() < nwords) {
    result.filterResult.resize(nwords);
  }
  auto metadataFiltersStartIndex = result.metadataFilterResults.size();

  for (int i = 0; i < scanSpec.numMetadataFilters(); ++i) {
    result.metadataFilterResults.emplace_back(
        scanSpec.metadataFilterNodeAt(i), std::vector<uint64_t>(nwords));
  }
  if (scanSpec.filter() || scanSpec.numMetadataFilters() > 0) {
    for (auto i = 0; i < fileMetaDataPtr_.numRowGroups(); ++i) {
      if (scanSpec.filter() && !rowGroupMatches(i, scanSpec.filter())) {
        bits::setBit(result.filterResult.data(), i);
        continue;
      }
      for (int j = 0; j < scanSpec.numMetadataFilters(); ++j) {
        auto* metadataFilter = scanSpec.metadataFilterAt(j);
        if (!rowGroupMatches(i, metadataFilter)) {
          bits::setBit(
              result.metadataFilterResults[metadataFiltersStartIndex + j]
                  .second.data(),
              i);
        }
      }
    }
  }
}

bool ParquetData::rowGroupMatches(uint32_t rowGroupId, common::Filter* filter) {
  auto column = type_->column();
  auto type = type_->type();
  auto rowGroup = fileMetaDataPtr_.rowGroup(rowGroupId);
  assert(rowGroup.numColumns() != 0);

  if (!filter) {
    return true;
  }

  bool needsToCheckBloomFilter = true;
  auto columnChunk = rowGroup.columnChunk(column);
  if (columnChunk.hasStatistics()) {
    auto columnStats =
        columnChunk.getColumnStatistics(type, rowGroup.numRows());
    if (!testFilter(filter, columnStats.get(), rowGroup.numRows(), type)) {
      return false;
    }

    // We can avoid testing bloom filter unnecessarily if we know that the
    // filter (min,max) range is a superset of the stats (min,max) range. For
    // example, if the filter is "COL between 1 and 20" and the column stats
    // range is (5,10), then we have to read the whole row group and hence avoid
    // bloom filter test.
    needsToCheckBloomFilter = parquetReadBloomFilter_ &&
        !isFilterRangeCoversStatsRange(filter, columnStats.get(), type);
  }

  if (needsToCheckBloomFilter &&
      rowGroup.columnChunk(column).hasBloomFilterOffset()) {
    std::unique_ptr<common::AbstractBloomFilter> parquetBloomFilter =
        std::make_unique<ParquetBloomFilter>(getBloomFilter(rowGroupId));
    return filter->testBloomFilter(*parquetBloomFilter, *type);
  }

  return true;
}

void ParquetData::enqueueRowGroup(
    uint32_t index,
    dwio::common::BufferedInput& input) {
  auto chunk = fileMetaDataPtr_.rowGroup(index).columnChunk(type_->column());
  streams_.resize(fileMetaDataPtr_.numRowGroups());
  VELOX_CHECK(
      chunk.hasMetadata(),
      "ColumnMetaData does not exist for schema Id ",
      type_->column());
  ;

  uint64_t chunkReadOffset = chunk.dataPageOffset();
  if (chunk.hasDictionaryPageOffset() && chunk.dictionaryPageOffset() >= 4) {
    // this assumes the data pages follow the dict pages directly.
    chunkReadOffset = chunk.dictionaryPageOffset();
  }

  uint64_t readSize =
      (chunk.compression() == common::CompressionKind::CompressionKind_NONE)
      ? chunk.totalUncompressedSize()
      : chunk.totalCompressedSize();

  auto id = dwio::common::StreamIdentifier(type_->column());
  streams_[index] = input.enqueue({chunkReadOffset, readSize}, &id);
}

dwio::common::PositionProvider ParquetData::seekToRowGroup(uint32_t index) {
  static std::vector<uint64_t> empty;
  VELOX_CHECK_LT(index, streams_.size());
  VELOX_CHECK(streams_[index], "Stream not enqueued for column");
  auto metadata = fileMetaDataPtr_.rowGroup(index).columnChunk(type_->column());
  reader_ = std::make_unique<PageReader>(
      std::move(streams_[index]),
      pool_,
      type_,
      metadata.compression(),
      metadata.totalCompressedSize(),
      sessionTimezone_);
  return dwio::common::PositionProvider(empty);
}

std::pair<int64_t, int64_t> ParquetData::getRowGroupRegion(
    uint32_t index) const {
  auto rowGroup = fileMetaDataPtr_.rowGroup(index);

  VELOX_CHECK_GT(rowGroup.numColumns(), 0);
  auto fileOffset = rowGroup.hasFileOffset() ? rowGroup.fileOffset()
      : rowGroup.columnChunk(0).hasDictionaryPageOffset()
      ? rowGroup.columnChunk(0).dictionaryPageOffset()
      : rowGroup.columnChunk(0).dataPageOffset();
  VELOX_CHECK_GT(fileOffset, 0);

  auto length = rowGroup.hasTotalCompressedSize()
      ? rowGroup.totalCompressedSize()
      : rowGroup.totalByteSize();

  return {fileOffset, length};
}

void ParquetData::setBloomFilterInputStream(
    uint32_t rowGroupId,
    dwio::common::BufferedInput& bufferedInput) {
  bloomFilterInputStreams_.resize(fileMetaDataPtr_.numRowGroups());
  if (bloomFilterInputStreams_[rowGroupId] != nullptr) {
    return;
  }
  auto rowGroup = fileMetaDataPtr_.rowGroup(rowGroupId);
  auto colChunk = rowGroup.columnChunk(type_->column());

  if (!colChunk.hasBloomFilterOffset()) {
    return;
  }

  VELOX_CHECK(
      !colChunk.hasCryptoMetadata(), "Cannot read encrypted bloom filter yet");

  auto bloomFilterOffset = colChunk.bloomFilterOffset();
  auto fileSize = bufferedInput.getInputStream()->getLength();
  VELOX_CHECK_GT(
      fileSize,
      bloomFilterOffset,
      "file size {} less or equal than bloom offset {}",
      fileSize,
      bloomFilterOffset);

  //    bloomFilterInputStream_ = bufferedInput.read(
  //        bloomFilterOffset,
  //        fileSize - bloomFilterOffset,
  //        dwio::common::LogType::FOOTER);

  auto id = dwio::common::StreamIdentifier(type_->column());
  bloomFilterInputStreams_[rowGroupId] = bufferedInput.enqueue(
      {static_cast<uint64_t>(bloomFilterOffset), fileSize - bloomFilterOffset},
      &id);
}

std::shared_ptr<BloomFilter> ParquetData::getBloomFilter(
    const uint32_t rowGroupId) {
  auto columnBloomFilterIter = columnBloomFilterMap_.find(rowGroupId);
  if (columnBloomFilterIter != columnBloomFilterMap_.end()) {
    return columnBloomFilterIter->second;
  }

  VELOX_CHECK_LT(
      rowGroupId,
      fileMetaDataPtr_.numRowGroups(),
      "Invalid row group ordinal: {}",
      rowGroupId);

  if (bloomFilterInputStreams_[rowGroupId] == nullptr) {
    return nullptr;
  }

  auto bloomFilter = BlockSplitBloomFilter::deserialize(
      bloomFilterInputStreams_[rowGroupId].get(), pool_);

  auto blockSplitBloomFilter =
      std::make_shared<BlockSplitBloomFilter>(std::move(bloomFilter));
  columnBloomFilterMap_[rowGroupId] = blockSplitBloomFilter;
  return blockSplitBloomFilter;
}

} // namespace facebook::velox::parquet
