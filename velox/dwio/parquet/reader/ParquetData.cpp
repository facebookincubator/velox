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
#include "velox/dwio/parquet/reader/Statistics.h"

namespace facebook::velox::parquet {

using thrift::RowGroup;

std::unique_ptr<dwio::common::FormatData> ParquetParams::toFormatData(
    const std::shared_ptr<const dwio::common::TypeWithId>& type,
    const common::ScanSpec& /*scanSpec*/) {
  return std::make_unique<ParquetData>(
      type, metaData_.get_row_groups(), pool());
}

void ParquetData::filterRowGroups(
    const common::ScanSpec& scanSpec,
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

bool ParquetData::rowGroupMatches(
    uint32_t rowGroupId,
    common::Filter* FOLLY_NULLABLE filter) {
  auto column = type_->column();
  auto type = type_->type();
  auto rowGroup = rowGroups_[rowGroupId];
  const auto& columns = rowGroup.get_columns();
  VELOX_CHECK(!columns.empty());

  if (!filter) {
    return true;
  }

  if (columns[column].meta_data().has_value() &&
      columns[column].meta_data()->statistics().has_value()) {
    auto columnStats = buildColumnStatisticsFromThrift(
        *columns[column].meta_data()->statistics(),
        *type,
        rowGroup.get_num_rows());
    return testFilter(filter, columnStats.get(), rowGroup.get_num_rows(), type);
  }
  return true;
}

void ParquetData::enqueueRowGroup(
    uint32_t index,
    dwio::common::BufferedInput& input) {
  const auto& chunk = rowGroups_[index].get_columns()[type_->column()];
  streams_.resize(rowGroups_.size());
  VELOX_CHECK(
      chunk.meta_data().has_value(),
      "ColumnMetaData does not exist for schema Id ",
      type_->column());
  const auto& metaData = *chunk.meta_data();

  uint64_t chunkReadOffset = metaData.get_data_page_offset();
  if (metaData.dictionary_page_offset().has_value() &&
      *metaData.dictionary_page_offset() >= 4) {
    // this assumes the data pages follow the dict pages directly.
    chunkReadOffset = *metaData.dictionary_page_offset();
  }
  VELOX_CHECK_GE(chunkReadOffset, 0);

  uint64_t readSize =
      (metaData.get_codec() == thrift::CompressionCodec::UNCOMPRESSED)
      ? metaData.get_total_uncompressed_size()
      : metaData.get_total_compressed_size();

  auto id = dwio::common::StreamIdentifier(type_->column());
  streams_[index] = input.enqueue({chunkReadOffset, readSize}, &id);
}

dwio::common::PositionProvider ParquetData::seekToRowGroup(uint32_t index) {
  static std::vector<uint64_t> empty;
  VELOX_CHECK_LT(index, streams_.size());
  VELOX_CHECK(streams_[index], "Stream not enqueued for column");
  auto& metadata =
      *rowGroups_[index].get_columns()[type_->column()].meta_data();
  reader_ = std::make_unique<PageReader>(
      std::move(streams_[index]),
      pool_,
      type_,
      metadata.get_codec(),
      metadata.get_total_compressed_size());
  return dwio::common::PositionProvider(empty);
}

std::pair<int64_t, int64_t> ParquetData::getRowGroupRegion(
    uint32_t index) const {
  auto& rowGroup = rowGroups_[index];

  VELOX_CHECK_GT(rowGroup.get_columns().size(), 0);
  auto fileOffset = rowGroup.file_offset().has_value() ? *rowGroup.file_offset()
      : rowGroup.get_columns()[0]
            .meta_data()
            ->dictionary_page_offset()
            .has_value()
      ? *rowGroup.get_columns()[0].meta_data()->dictionary_page_offset()
      : rowGroup.get_columns()[0].meta_data()->get_data_page_offset();
  VELOX_CHECK_GT(fileOffset, 0);

  auto length = rowGroup.total_compressed_size().has_value()
      ? *rowGroup.total_compressed_size()
      : rowGroup.get_total_byte_size();

  return {fileOffset, length};
}

} // namespace facebook::velox::parquet
