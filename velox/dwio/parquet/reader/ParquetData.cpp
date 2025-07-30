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

bool ParquetData::collectIndexPageInfoMap(
    uint32_t index,
    PageIndexInfoMap& map) {
  bool applyPageSkipping =
      scanSpec_.filter() || scanSpec_.numMetadataFilters() > 0;
  if (applyPageSkipping) {
    for (auto* parent = type_.get(); parent != nullptr;
         parent = parent->parquetParent()) {
      if (parent->parquetParent() &&
          (parent->type()->kind() == TypeKind::ARRAY ||
           parent->type()->kind() == TypeKind::MAP ||
           parent->type()->kind() == TypeKind::ROW)) {
        applyPageSkipping = false;
        break;
      }
    }
  }
  const auto& chunk =
      fileMetaDataPtr_.rowGroup(index).columnChunk(type_->column());
  if (chunk.hasColumnAndOffsetIndexOffset()) {
    map[type_->column()] = {
        chunk.offsetIndexOffset(),
        chunk.offsetIndexLength(),
        chunk.columnIndexOffset(),
        chunk.columnIndexLength()};
  }
  return applyPageSkipping;
}

void ParquetData::filterDataPages(
    uint32_t index,
    folly::F14FastMap<uint32_t, std::unique_ptr<ColumnPageIndex>>& pageIndices,
    dwio::common::RowRanges& range,
    std::vector<std::pair<
        const velox::common::MetadataFilter::LeafNode*,
        dwio::common::RowRanges>>& metadataFilterResults) {
  pageIndices_.resize(fileMetaDataPtr_.numRowGroups());
  auto it = pageIndices.find(type_->column());
  if (it != pageIndices.end()) {
    pageIndices_[index] = std::move(it->second);
  } else {
    pageIndices_[index].reset();
    return;
  }

  dwio::common::RowRanges filteredPages;
  auto* pageIndex = pageIndices_[index].get();
  auto numPages = pageIndex->numPages();
  auto type = type_->type();

  auto metadataFiltersStartIndex = metadataFilterResults.size();
  for (int i = 0; i < scanSpec_.numMetadataFilters(); ++i) {
    metadataFilterResults.emplace_back(
        scanSpec_.metadataFilterNodeAt(i), dwio::common::RowRanges());
  }

  for (auto i = 0; i < numPages; ++i) {
    auto numRows = pageIndex->pageRowCount(i);
    auto firstRowIndex = pageIndex->pageFirstRowIndex(i);
    auto lastRowIndex = firstRowIndex + numRows - 1;
    auto stats = pageIndex->buildColumnStatisticsForPage(i, *type);

    // Skip page if main filter does not match.
    if (scanSpec_.filter() &&
        !testFilter(scanSpec_.filter(), stats.get(), numRows, type)) {
      filteredPages.add(dwio::common::RowRange(firstRowIndex, lastRowIndex));
      continue;
    }

    for (int j = 0; j < scanSpec_.numMetadataFilters(); ++j) {
      auto* metadataFilter = scanSpec_.metadataFilterAt(j);
      if (metadataFilter &&
          !testFilter(metadataFilter, stats.get(), numRows, type)) {
        metadataFilterResults[metadataFiltersStartIndex + j].second.add(
            dwio::common::RowRange(firstRowIndex, lastRowIndex));
      }
    }
  }

  range = dwio::common::RowRanges::unionWith(range, filteredPages);
}

void ParquetData::enqueueRowGroup(
    uint32_t index,
    dwio::common::BufferedInput& input,
    dwio::common::RowRanges& rowRanges) {
  auto chunk = fileMetaDataPtr_.rowGroup(index).columnChunk(type_->column());
  streams_.resize(fileMetaDataPtr_.numRowGroups());
  pagesStreams_.resize(fileMetaDataPtr_.numRowGroups());
  pageIndices_.resize(fileMetaDataPtr_.numRowGroups());
  VELOX_CHECK(
      chunk.hasMetadata(),
      "ColumnMetaData does not exist for schema Id ",
      type_->column());

  uint64_t chunkReadOffset = chunk.dataPageOffset();

  // Parquet file header is 4 bytes, so dictionaryPageOffset() < 4 is invalid.
  if (chunk.hasDictionaryPageOffset() && chunk.dictionaryPageOffset() >= 4) {
    // This assumes the data pages follow the dict pages directly.
    chunkReadOffset = chunk.dictionaryPageOffset();
  }

  uint64_t readSize =
      (chunk.compression() == common::CompressionKind::CompressionKind_NONE)
      ? chunk.totalUncompressedSize()
      : chunk.totalCompressedSize();

  if (handlePageSkipping(index, input, rowRanges, chunkReadOffset, readSize) >
      0) {
    return;
  }

  auto id = dwio::common::StreamIdentifier(type_->column());
  streams_[index] = input.enqueue({chunkReadOffset, readSize}, &id);
}

int64_t ParquetData::handlePageSkipping(
    uint32_t index,
    dwio::common::BufferedInput& input,
    dwio::common::RowRanges& rowRanges,
    uint64_t chunkReadOffset,
    uint64_t chunkReadSize) {
  int64_t skippedPages = 0;
  // Check if the required chunk is already buffered.
  if (!input.isBuffered(chunkReadOffset, chunkReadSize)) {
    // Update skipped pages if applicable.
    if (pageIndices_[index]) {
      skippedPages = pageIndices_[index]->updateSkippedPages(rowRanges);
      rowRanges.updateAffectedPages(skippedPages);
      rowRanges.updateCoveredPages(
          pageIndices_[index]->numPages() - skippedPages);
    }

    // If there are skipped pages, enqueue only the required pages.
    if (pageIndices_[index] && pageIndices_[index]->hasSkippedPages()) {
      auto numPages = pageIndices_[index]->numPages();
      pagesStreams_[index].clear();

      // Combine contiguous pages into runs to reduce the number of streams.
      // This is done to optimize the number of streams and reduce overhead.
      // Each run will be a contiguous set of pages that can be read together.
      int32_t runStart = -1;
      uint64_t runOffset = 0;
      uint64_t runLength = 0;
      std::vector<int32_t> runPages;

      // Helper to flush a run of contiguous pages.
      auto flushRun = [&]() {
        if (runStart < 0) {
          return;
        }
        auto id = dwio::common::StreamIdentifier(
            type_->column() * (numPages + 1) + pagesStreams_[index].size());
        auto streamPtr = input.enqueue({runOffset, runLength}, &id);
        pagesStreams_[index].push_back(std::move(streamPtr));
        int32_t streamIdx = pagesStreams_[index].size() - 1;
        for (int32_t page : runPages) {
          pageIndices_[index]->setPageStreamIndex(page, streamIdx);
        }
        runStart = -1;
        runPages.clear();
      };

      auto addPageToRun = [&](int32_t page, uint64_t offset, uint64_t length) {
        if (runStart < 0) {
          runStart = page;
          runOffset = offset;
          runLength = length;
          runPages = {page};
        } else {
          runLength += length;
          runPages.push_back(page);
        }
      };

      // Dictionary page.
      auto chunk =
          fileMetaDataPtr_.rowGroup(index).columnChunk(type_->column());
      if (chunk.hasDictionaryPageOffset() &&
          chunk.dictionaryPageOffset() >= 4) {
        uint64_t dictOffset = chunk.dictionaryPageOffset();
        uint64_t dictLen = chunk.dataPageOffset() - dictOffset;
        addPageToRun(0, dictOffset, dictLen);
      }

      // Data pages.
      for (int32_t i = 0; i < numPages; ++i) {
        if (!pageIndices_[index]->isPageSkipped(i)) {
          uint64_t off = pageIndices_[index]->pageOffset(i);
          uint64_t len = pageIndices_[index]->compressedPageSize(i);
          addPageToRun(i + 1, off, len);
        } else {
          flushRun();
        }
      }
      // Ensure the last run is flushed if it exists.
      flushRun();
      streams_[index] = nullptr;
    }
  }
  return skippedPages;
}

dwio::common::PositionProvider ParquetData::seekToRowGroup(int64_t index) {
  static std::vector<uint64_t> empty;
  VELOX_CHECK_LT(index, streams_.size());
  auto metadata = fileMetaDataPtr_.rowGroup(index).columnChunk(type_->column());
  if (streams_[index] == nullptr) {
    VELOX_CHECK_NOT_NULL(pageIndices_[index]);
    reader_ = std::make_unique<PageReader>(
        std::move(pagesStreams_[index]),
        pool_,
        type_,
        metadata.compression(),
        metadata.totalCompressedSize(),
        stats_,
        sessionTimezone_,
        std::move(pageIndices_[index]));
  } else {
    VELOX_CHECK(streams_[index], "Stream not enqueued for column");
    reader_ = std::make_unique<PageReader>(
        std::move(streams_[index]),
        pool_,
        type_,
        metadata.compression(),
        metadata.totalCompressedSize(),
        stats_,
        sessionTimezone_);
  }
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
