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

#include <thrift/protocol/TCompactProtocol.h>
#include <iomanip>
#include "velox/dwio/parquet/thrift/ThriftTransport.h"

namespace facebook::velox::parquet {

std::unique_ptr<dwio::common::FormatData> ParquetParams::toFormatData(
    const std::shared_ptr<const dwio::common::TypeWithId>& type,
    const common::ScanSpec& /*scanSpec*/) {
  auto parquetData = std::make_unique<ParquetData>(
      type, metaData_, pool(), sessionTimezone_);
  // Set the BufferedInput if available
  if (bufferedInput_) {
    parquetData->setBufferedInput(bufferedInput_);
  }
  return parquetData;
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
    bool statisticsResult = testFilter(filter, columnStats.get(), rowGroup.numRows(), type);
    if (!statisticsResult) {
      return false;
    }
  }


    bool canUseDictionaryFiltering = isOnlyDictionaryEncodingPagesImpl(columnChunk);
    if (canUseDictionaryFiltering) {
      bool dictionaryResult = testFilterAgainstDictionary(rowGroupId, filter, columnChunk);
      if (!dictionaryResult) {
        return false;
      }
    }
    return true;
}

void ParquetData::enqueueRowGroup(
    uint32_t index,
    dwio::common::BufferedInput& input) {
  // Store the BufferedInput reference for creating streams on demand
  bufferedInput_ = &input;

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

// Presto's exact isOnlyDictionaryEncodingPages function from PR #4779
bool ParquetData::isOnlyDictionaryEncodingPagesImpl(const ColumnChunkMetaDataPtr& columnChunk) {
  // Files written with newer versions of Parquet libraries (e.g. parquet-mr 1.9.0) will have EncodingStats available
  // Otherwise, fallback to v1 logic

  // Check for EncodingStats when available (newer Parquet files)
  if (columnChunk.hasEncodingStats()) {
    const auto& stats = columnChunk.getEncodingStats();
    return hasDictionaryPages(stats) && !hasNonDictionaryEncodedPages(stats);
  }

  // Fallback to v1 logic
  auto encodings = columnChunk.getEncoding();
  std::set<thrift::Encoding::type> encodingSet(encodings.begin(), encodings.end());

  if (encodingSet.count(thrift::Encoding::PLAIN_DICTIONARY)) {
    // PLAIN_DICTIONARY was present, which means at least one page was
    // dictionary-encoded and 1.0 encodings are used
    // The only other allowed encodings are RLE and BIT_PACKED which are used for repetition or definition levels
    std::set<thrift::Encoding::type> allowedEncodings = {
      thrift::Encoding::PLAIN_DICTIONARY,
      thrift::Encoding::RLE,           // For repetition/definition levels
      thrift::Encoding::BIT_PACKED     // For repetition/definition levels
    };

    // Check if there are any disallowed encodings (equivalent to Sets.difference in Java)
    for (const auto& encoding : encodings) {
      if (allowedEncodings.find(encoding) == allowedEncodings.end()) {
        return false;
      }
    }
    return true;
  }

  return false;
}

// Helper methods for EncodingStats analysis (like Java Presto)
bool ParquetData::hasDictionaryPages(const std::vector<thrift::PageEncodingStats>& stats) {
  for (const auto& pageStats : stats) {
    if (pageStats.page_type == thrift::PageType::DICTIONARY_PAGE) {
      return true;
    }
  }
  return false;
}

bool ParquetData::hasNonDictionaryEncodedPages(const std::vector<thrift::PageEncodingStats>& stats) {
  for (const auto& pageStats : stats) {
    if (pageStats.page_type == thrift::PageType::DATA_PAGE ||
        pageStats.page_type == thrift::PageType::DATA_PAGE_V2) {
      // Check if this data page uses non-dictionary encoding
      if (pageStats.encoding != thrift::Encoding::PLAIN_DICTIONARY &&
          pageStats.encoding != thrift::Encoding::RLE_DICTIONARY) {
        return true;
      }
    }
  }
  return false;
}

bool ParquetData::testFilterAgainstDictionary(
    uint32_t rowGroupId,
    const common::Filter* filter,
    const ColumnChunkMetaDataPtr& columnChunk) {
    if (!filter) {
      VLOG(3) << "    [DICT-TEST] No filter provided, dictionary test passes";
    }

  // Log file path for debugging
  std::string filePath = "unknown";
  if (bufferedInput_ && bufferedInput_->getReadFile()) {
    filePath = bufferedInput_->getReadFile()->getName();
  }

  // Special handling for IsNull filters - conservative include before dictionary parsing
  if (filter->kind() == common::FilterKind::kIsNull) {
    return true; // Conservative include for IsNull filters
  }

  const dwio::common::DictionaryValues* finalDictionary = nullptr;
  auto dictionaryValues = readDictionaryPageForFiltering(rowGroupId, columnChunk);
  if (!dictionaryValues.values || dictionaryValues.numValues == 0) {
    return true;
  }

  tempDictionary_ = std::move(dictionaryValues);
  finalDictionary = &tempDictionary_;

  // Step 2: Test filter directly against dictionary values using dict() pattern
  auto numValues = finalDictionary->numValues;

  // Test if any dictionary value passes the filter
  auto testDict = [&]<typename T>() {
    const T* dict = reinterpret_cast<const T*>(finalDictionary->values->as<void>());

    // For larger dictionaries, we could use SIMD testValues() for better performance
    // For now, use simple scalar approach which is sufficient for typical dictionary sizes
    for (int32_t i = 0; i < numValues; ++i) {
      if (common::applyFilter(*filter, dict[i])) return true;
    }
    return false;
  };

  bool anyValuePasses = [&] {
    switch (type_->type()->kind()) {
      case TypeKind::BIGINT:    return testDict.operator()<int64_t>();
      case TypeKind::INTEGER:   return testDict.operator()<int32_t>();
      case TypeKind::VARCHAR:
      case TypeKind::VARBINARY: return testDict.operator()<StringView>();
      case TypeKind::REAL:      return testDict.operator()<float>();
      case TypeKind::DOUBLE:    return testDict.operator()<double>();
      case TypeKind::BOOLEAN:   return testDict.operator()<bool>();
      default: return true;  // Conservative fallback
    }
  }();

  // If no dictionary values pass the filter, but the filter accepts NULLs,
  // we must conservatively include because the row group might contain NULLs that would match
  if (!anyValuePasses && filter->testNull()) {
    anyValuePasses = true;
  }
  return anyValuePasses;
}

// Read dictionary page directly for row group filtering (like Presto's dictionaryPredicatesMatch)
dwio::common::DictionaryValues ParquetData::readDictionaryPageForFiltering(
    uint32_t rowGroupId,
    const ColumnChunkMetaDataPtr& columnChunk) {

  // Create input stream for the column chunk
  auto inputStream = getInputStream(rowGroupId, columnChunk);
  if (!inputStream) {
    return dwio::common::DictionaryValues{};
  }

  // Create PageReader - it will automatically handle dictionary loading
  auto pageReader = std::make_unique<PageReader>(
      std::move(inputStream),
      pool_,
      type_,
      columnChunk.compression(),
      columnChunk.totalCompressedSize(),
      sessionTimezone_);


  // Read the first page header to trigger dictionary loading
  auto pageHeader = pageReader->readPageHeader();

  // If it's a dictionary page, prepare it
  if (pageHeader.type == thrift::PageType::DICTIONARY_PAGE) {
    pageReader->prepareDictionary(pageHeader);
  } else {
    return dwio::common::DictionaryValues{};
  }

  // Get the dictionary data directly from PageReader
  const auto& dict = pageReader->dictionary();
  return dict;
}
// Helper method to get input stream for column chunk
std::unique_ptr<dwio::common::SeekableInputStream> ParquetData::getInputStream(
    uint32_t rowGroupId, const ColumnChunkMetaDataPtr& columnChunk) {
  // Use existing stream if available
  if (rowGroupId < streams_.size() && streams_[rowGroupId]) {
    return std::move(streams_[rowGroupId]);
  }

  // Create new stream using the same logic as enqueueRowGroup
  if (!bufferedInput_) {
    return nullptr;
  }

  // Calculate read parameters (same as enqueueRowGroup)
  uint64_t chunkReadOffset = columnChunk.dataPageOffset();
  if (columnChunk.hasDictionaryPageOffset() && columnChunk.dictionaryPageOffset() >= 4) {
    chunkReadOffset = columnChunk.dictionaryPageOffset();
  }

  uint64_t readSize =
      (columnChunk.compression() == common::CompressionKind::CompressionKind_NONE)
      ? columnChunk.totalUncompressedSize()
      : columnChunk.totalCompressedSize();

  auto id = dwio::common::StreamIdentifier(type_->column());
  auto stream = bufferedInput_->enqueue({chunkReadOffset, readSize}, &id);

  bufferedInput_->load(dwio::common::LogType::STRIPE);
  return stream;
}

} // namespace facebook::velox::parquet
