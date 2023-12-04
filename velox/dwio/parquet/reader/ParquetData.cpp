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

#include "velox/dwio/parquet/common/ParquetBloomFilter.h"
#include "velox/dwio/parquet/reader/Statistics.h"

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
      type, metaData_.row_groups, pool(), isUseParquetBloomFilter_);
}

void ParquetData::filterRowGroups(
    const common::ScanSpec& scanSpec,
    dwio::common::BufferedInput& bufferedInput,
    uint64_t /*rowsPerRowGroup*/,
    const dwio::common::StatsContext& /*writerContext*/,
    FilterRowGroupsResult& result) {
  result.totalCount = std::max<int>(result.totalCount, rowGroups_.size());
  auto nwords = bits::nwords(result.totalCount);
  if (result.filterResult.size() < nwords) {
    result.filterResult.resize(nwords);
  }
  auto metadataFiltersStartIndex = result.metadataFilterResults.size();

  for (int i = 0; i < scanSpec.numMetadataFilters(); ++i) {
    result.metadataFilterResults.emplace_back(
        scanSpec.metadataFilterNodeAt(i), std::vector<uint64_t>(nwords));
  }

  for (auto i = 0; i < rowGroups_.size(); ++i) {
    if (scanSpec.filter() &&
        !rowGroupMatches(i, scanSpec.filter(), bufferedInput)) {
      bits::setBit(result.filterResult.data(), i);
      continue;
    }
    for (int j = 0; j < scanSpec.numMetadataFilters(); ++j) {
      auto* metadataFilter = scanSpec.metadataFilterAt(j);
      if (!rowGroupMatches(i, metadataFilter, bufferedInput)) {
        bits::setBit(
            result.metadataFilterResults[metadataFiltersStartIndex + j]
                .second.data(),
            i);
      }
    }
  }
}

bool ParquetData::rowGroupMatches(
    uint32_t rowGroupId,
    common::Filter* FOLLY_NULLABLE filter,
    dwio::common::BufferedInput& bufferedInput) {
  auto column = type_->column();
  auto type = type_->type();
  auto rowGroup = rowGroups_[rowGroupId];
  assert(!rowGroup.columns.empty());

  if (!filter) {
    return true;
  }

  bool needsToCheckBloomFilter = true;
  if (rowGroup.columns[column].__isset.meta_data &&
      rowGroup.columns[column].meta_data.__isset.statistics) {
    auto columnStats = buildColumnStatisticsFromThrift(
        rowGroup.columns[column].meta_data.statistics,
        *type,
        rowGroup.num_rows);
    if (!testFilter(filter, columnStats.get(), rowGroup.num_rows, type)) {
      return false;
    }

    // We can avoid testing bloom filter unnecessarily if we know that the
    // filter (min,max) range is a superset of the stats (min,max) range. For
    // example, if the filter is "COL between 1 and 20" and the column stats
    // range is (5,10), then we have to read the whole row group and hence avoid
    // bloom filter test.
    needsToCheckBloomFilter = isParquetBloomFilterEnabled_ &&
        !isFilterRangeCoversStatsRange(filter, columnStats.get(), type);
  }

  if (needsToCheckBloomFilter && rowGroup.columns[column].__isset.meta_data &&
      rowGroup.columns[column].meta_data.__isset.bloom_filter_offset) {
    std::unique_ptr<AbstractBloomFilter> parquetBloomFilter =
        std::make_unique<ParquetBloomFilter>(
            getBloomFilter(bufferedInput, rowGroupId));
    return filter->testBloomFilter(*parquetBloomFilter, *type);
  }

  return true;
}

void ParquetData::enqueueRowGroup(
    uint32_t index,
    dwio::common::BufferedInput& input) {
  auto& chunk = rowGroups_[index].columns[type_->column()];
  streams_.resize(rowGroups_.size());
  VELOX_CHECK(
      chunk.__isset.meta_data,
      "ColumnMetaData does not exist for schema Id ",
      type_->column());
  auto& metaData = chunk.meta_data;

  uint64_t chunkReadOffset = metaData.data_page_offset;
  if (metaData.__isset.dictionary_page_offset &&
      metaData.dictionary_page_offset >= 4) {
    // this assumes the data pages follow the dict pages directly.
    chunkReadOffset = metaData.dictionary_page_offset;
  }
  VELOX_CHECK_GE(chunkReadOffset, 0);

  uint64_t readSize = (metaData.codec == thrift::CompressionCodec::UNCOMPRESSED)
      ? metaData.total_uncompressed_size
      : metaData.total_compressed_size;

  auto id = dwio::common::StreamIdentifier(type_->column());
  streams_[index] = input.enqueue({chunkReadOffset, readSize}, &id);
}

dwio::common::PositionProvider ParquetData::seekToRowGroup(uint32_t index) {
  static std::vector<uint64_t> empty;
  VELOX_CHECK_LT(index, streams_.size());
  VELOX_CHECK(streams_[index], "Stream not enqueued for column");
  auto& metadata = rowGroups_[index].columns[type_->column()].meta_data;
  reader_ = std::make_unique<PageReader>(
      std::move(streams_[index]),
      pool_,
      type_,
      metadata.codec,
      metadata.total_compressed_size);
  return dwio::common::PositionProvider(empty);
}

std::pair<int64_t, int64_t> ParquetData::getRowGroupRegion(
    uint32_t index) const {
  auto& rowGroup = rowGroups_[index];

  VELOX_CHECK_GT(rowGroup.columns.size(), 0);
  auto fileOffset = rowGroup.__isset.file_offset ? rowGroup.file_offset
      : rowGroup.columns[0].meta_data.__isset.dictionary_page_offset
      ? rowGroup.columns[0].meta_data.dictionary_page_offset
      : rowGroup.columns[0].meta_data.data_page_offset;
  VELOX_CHECK_GT(fileOffset, 0);

  auto length = rowGroup.__isset.total_compressed_size
      ? rowGroup.total_compressed_size
      : rowGroup.total_byte_size;

  return {fileOffset, length};
}

std::shared_ptr<BloomFilter> ParquetData::getBloomFilter(
    dwio::common::BufferedInput& bufferedInput,
    const uint32_t rowGroupId) {
  auto columnBloomFilterIter = columnBloomFilterMap_.find(rowGroupId);
  if (columnBloomFilterIter != columnBloomFilterMap_.end()) {
    return columnBloomFilterIter->second;
  }

  VELOX_CHECK_LT(
      rowGroupId,
      rowGroups_.size(),
      "Invalid row group ordinal: {}",
      rowGroupId);

  auto rowGroup = rowGroups_[rowGroupId];
  auto colChunk = rowGroup.columns[type_->column()];
  VELOX_CHECK(
      !colChunk.__isset.crypto_metadata,
      "Cannot read encrypted bloom filter yet");

  if (!colChunk.meta_data.__isset.bloom_filter_offset) {
    return nullptr;
  }
  auto bloomFilterOffset = colChunk.meta_data.bloom_filter_offset;
  auto fileSize = bufferedInput.getInputStream()->getLength();
  VELOX_CHECK_GT(
      fileSize,
      bloomFilterOffset,
      "file size {} less or equal than bloom offset {}",
      fileSize,
      bloomFilterOffset);

  auto inputStream = bufferedInput.read(
      bloomFilterOffset,
      fileSize - bloomFilterOffset,
      dwio::common::LogType::FOOTER);
  auto bloomFilter =
      BlockSplitBloomFilter::deserialize(inputStream.get(), pool_);

  auto blockSplitBloomFilter =
      std::make_shared<BlockSplitBloomFilter>(std::move(bloomFilter));
  columnBloomFilterMap_[rowGroupId] = blockSplitBloomFilter;
  return blockSplitBloomFilter;
}

} // namespace facebook::velox::parquet
