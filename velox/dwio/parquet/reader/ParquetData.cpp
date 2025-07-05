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

namespace facebook::velox::parquet {

std::unique_ptr<dwio::common::FormatData> ParquetParams::toFormatData(
    const std::shared_ptr<const dwio::common::TypeWithId>& type,
    const common::ScanSpec& scanSpec) {
  return std::make_unique<ParquetData>(
      type,
      metaData_,
      pool(),
      runtimeStatistics(),
      sessionTimezone_,
      scanSpec);
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
        // std::cout << "row group " << i << " is filtered" << std::endl;
        bits::setBit(result.filterResult.data(), i);
        continue;
      }
      for (int j = 0; j < scanSpec.numMetadataFilters(); ++j) {
        auto* metadataFilter = scanSpec.metadataFilterAt(j);
        if (!rowGroupMatches(i, metadataFilter)) {
          // std::cout << "row group " << i << " is filtered" << std::endl;
          bits::setBit(
              result.metadataFilterResults[metadataFiltersStartIndex + j]
                  .second.data(),
              i);
        }
      }
    }
  }
}

bool ParquetData::rowGroupMatches(
    uint32_t rowGroupId,
    const common::Filter* filter) {
  auto column = type_->column();
  auto type = type_->type();
  auto rowGroup = fileMetaDataPtr_.rowGroup(rowGroupId);
  assert(rowGroup.numColumns() != 0);

  if (!filter) {
    return true;
  }

  auto columnChunk = rowGroup.columnChunk(column);
  if (columnChunk.hasStatistics()) {
    auto columnStats =
        columnChunk.getColumnStatistics(type, rowGroup.numRows());
    return testFilter(filter, columnStats.get(), rowGroup.numRows(), type);
  }
  return true;
}

void ParquetData::collectIndexPageInfoMap(
    uint32_t index,
    PageIndexInfoMap& map) {
  auto chunk = fileMetaDataPtr_.rowGroup(index).columnChunk(type_->column());
  if (chunk.hasColumnAndOffsetIndexOffset()) {
    map[type_->column()] = {
        chunk.offsetIndexOffset(),
        chunk.offsetIndexLength(),
        chunk.columnIndexOffset(),
        chunk.columnIndexLength()};
  }
}

void ParquetData::filterDataPages(
    uint32_t index,
    folly::F14FastMap<uint32_t, std::unique_ptr<ColumnPageIndex>>& pageIndices,
    RowRanges& range) {
  pageIndices_.resize(fileMetaDataPtr_.numRowGroups());
  auto it = pageIndices.find(type_->column());
  if (it != pageIndices.end()) {
    pageIndices_[index] = std::move(it->second);
  } else {
    pageIndices_[index].reset();
  }

  if (pageIndices_[index] != nullptr) {
    RowRanges temp;
    auto numPages = pageIndices_[index]->numPages();
    auto type = type_->type();
    // std::cout <<"row group " << index << " column" << type_->column() <<
    // "page index filter:" << numPages << std::endl;
    for (auto i = 0; i < numPages; ++i) {
      auto numRows = pageIndices_[index]->pageRowCount(i);
      auto firstRowIndex = pageIndices_[index]->pageFirstRowIndex(i);
      // std::cout <<"row group " << index << " column" << type_->column();
      auto stats = pageIndices_[index]->buildColumnStatisticsForPage(i, *type);

      if (scanSpec_.filter() &&
          !testFilter(scanSpec_.filter(), stats.get(), numRows, type)) {
        // std::cout <<"row group " << index << " column" << type_->column() <<
        // " page " << i<< "filter done" << std::endl;
        continue;
      }
      bool shouldSkip = false;
      for (int j = 0; j < scanSpec_.numMetadataFilters(); ++j) {
        auto* metadataFilter = scanSpec_.metadataFilterAt(j);
        if (metadataFilter &&
            !testFilter(metadataFilter, stats.get(), numRows, type)) {
          // std::cout <<"row group " << index << " column" << type_->column()
          // << " page " << i << "metadata filter done" << std::endl;
          shouldSkip = true;
          continue;
        }
      }
      if (shouldSkip) {
        continue;
      }
      RowRange r(firstRowIndex, firstRowIndex + numRows - 1);
      // std::cout <<"row group " << index << " column" << type_->column() <<
      // "page " << i <<  " add " << firstRowIndex << " " << firstRowIndex +
      // numRows - 1 << std::endl;
      temp.add(r);
    }
    // std::cout <<"row group " << index << " column" << type_->column() <<
    // "intersection:" << range.toString() << " " << temp.toString() <<
    // std::endl;
    range = RowRanges::intersection(range, temp);
    // std::cout <<"row group " << index << " column" << type_->column() <<
    // "end filterDataPages:" << range.toString() << std::endl;
  }
}
void ParquetData::enqueueRowGroup(
    uint32_t index,
    dwio::common::BufferedInput& input,
    const RowRanges& rowRanges) {
  if (pageIndices_.size() > index && pageIndices_[index] != nullptr) {
    // std::cout <<"row group " << index << " column" << type_->column() <<
    // "update skip pages:" << rowRanges.toString() << std::endl;
    pageIndices_[index]->updateSkippedPages(rowRanges);
    for (int i = 0; i < pageIndices_[index]->numPages(); ++i) {
      if (pageIndices_[index]->isPageSkipped(i)) {
      }
      // std::cout <<"row group " << index << " page " << i << " is skipped"<<
      // std::endl;
    }
  }
  auto chunk = fileMetaDataPtr_.rowGroup(index).columnChunk(type_->column());
  streams_.resize(fileMetaDataPtr_.numRowGroups());
  VELOX_CHECK(
      chunk.hasMetadata(),
      "ColumnMetaData does not exist for schema Id ",
      type_->column());
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

dwio::common::PositionProvider ParquetData::seekToRowGroup(int64_t index) {
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
      stats_,
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

} // namespace facebook::velox::parquet
