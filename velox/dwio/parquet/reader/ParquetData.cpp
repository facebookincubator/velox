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
#include "velox/type/Filter.h"

namespace facebook::velox::parquet {

namespace {

// True if the column chunk's encoding statistics record a dictionary page.
bool hasDictionaryPages(const std::vector<thrift::PageEncodingStats>& stats) {
  for (const auto& pageStats : stats) {
    if (*pageStats.page_type() == thrift::PageType::DICTIONARY_PAGE) {
      return true;
    }
  }
  return false;
}

// True if any data page in the column chunk uses a non-dictionary encoding.
bool hasNonDictionaryEncodedPages(
    const std::vector<thrift::PageEncodingStats>& stats) {
  for (const auto& pageStats : stats) {
    const auto pageType = *pageStats.page_type();
    if (pageType == thrift::PageType::DATA_PAGE ||
        pageType == thrift::PageType::DATA_PAGE_V2) {
      const auto encoding = *pageStats.encoding();
      if (encoding != thrift::Encoding::PLAIN_DICTIONARY &&
          encoding != thrift::Encoding::RLE_DICTIONARY) {
        return true;
      }
    }
  }
  return false;
}

} // namespace

std::unique_ptr<dwio::common::FormatData> ParquetParams::toFormatData(
    const std::shared_ptr<const dwio::common::TypeWithId>& type,
    const common::ScanSpec& /*scanSpec*/) {
  return std::make_unique<ParquetData>(
      type,
      metaData_,
      pool(),
      runtimeStatistics(),
      sessionTimezone_,
      bufferedInput_,
      dictionaryRowGroupSkippingEnabled_);
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
      // Already excluded by another column or by the caller (e.g. row group
      // outside the split range, empty row group). Skip statistics build and
      // testFilter. The MetadataFilter::eval call ORs into filterResult, so
      // leaving the per-leaf metadata bits at 0 here is harmless: filterResult
      // already has the bit set.
      if (bits::isBitSet(result.filterResult.data(), i)) {
        continue;
      }
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
    auto columnStats = columnChunk.getColumnStatistics(
        type_->type(),
        rowGroup.numRows(),
        type_->convertedType_,
        type_->logicalType_);
    if (!testFilter(filter, columnStats.get(), rowGroup.numRows(), type)) {
      return false;
    }
  }
  if (dictionaryRowGroupSkippingEnabled_ &&
      isOnlyDictionaryEncodingPages(columnChunk)) {
    if (!testFilterAgainstDictionary(rowGroupId, filter, columnChunk)) {
      return false;
    }
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

bool ParquetData::isOnlyDictionaryEncodingPages(
    const ColumnChunkMetaDataPtr& columnChunk) {
  // Files written by newer Parquet writers (parquet-mr 1.9.0+) record per-page
  // encoding statistics, which tell us directly whether every data page is
  // dictionary encoded.
  if (columnChunk.hasEncodingStats()) {
    const auto stats = columnChunk.encodingStats();
    return hasDictionaryPages(stats) && !hasNonDictionaryEncodedPages(stats);
  }

  // Older files lack encoding statistics, so fall back to the chunk's encoding
  // list. A dictionary was used only if PLAIN_DICTIONARY appears and every
  // other encoding is one used for repetition/definition levels (RLE,
  // BIT_PACKED); any other encoding means some data page was not dictionary
  // encoded.
  const auto encodings = columnChunk.encodings();
  const std::set<thrift::Encoding> encodingSet(
      encodings.begin(), encodings.end());
  if (encodingSet.count(thrift::Encoding::PLAIN_DICTIONARY)) {
    static const std::set<thrift::Encoding> kAllowedEncodings = {
        thrift::Encoding::PLAIN_DICTIONARY,
        thrift::Encoding::RLE,
        thrift::Encoding::BIT_PACKED,
    };
    for (const auto& encoding : encodings) {
      if (kAllowedEncodings.find(encoding) == kAllowedEncodings.end()) {
        return false;
      }
    }
    return true;
  }
  return false;
}

bool ParquetData::testFilterAgainstDictionary(
    uint32_t rowGroupId,
    const common::Filter* filter,
    const ColumnChunkMetaDataPtr& columnChunk) {
  if (filter->kind() == common::FilterKind::kIsNull) {
    return true; // Conservative include for IsNull filters.
  }

  auto dictionaryPtr = readDictionaryPageForFiltering(rowGroupId, columnChunk);

  auto numValues = dictionaryPtr->numValues;
  if (numValues == 0) {
    return true; // Conservative: no dict values means we cannot prune.
  }
  const void* dictPtr = dictionaryPtr->values->as<void>();

  // Test if any dictionary value passes the filter.
  auto testDict = [&]<typename T>() {
    const T* dict = reinterpret_cast<const T*>(dictPtr);
    for (int32_t i = 0; i < numValues; ++i) {
      if (common::applyFilter(*filter, dict[i])) {
        return true;
      }
    }
    return false;
  };

  bool anyValuePasses = [&] {
    switch (type_->type()->kind()) {
      case TypeKind::BIGINT:
        // A Velox BIGINT can be backed by a parquet INT32 (e.g. TIME with
        // millisecond precision, or DATE). prepareDictionary leaves such a
        // dictionary in its physical 4-byte layout, so reading 8-byte values
        // from it would conflate adjacent entries. Short decimals are already
        // expanded to int64 by prepareDictionary and take the default path.
        if (type_->parquetType_.has_value() &&
            type_->parquetType_.value() == thrift::Type::INT32 &&
            !type_->type()->isShortDecimal()) {
          // Widen with the column's signedness. An unsigned INT32 value above
          // 2^31-1 read as int32 would become negative and cause incorrect
          // pruning, so unsigned columns widen through uint32_t.
          const bool isUnsigned = type_->logicalType_.has_value() &&
              type_->logicalType_->getType() ==
                  thrift::LogicalType::Type::INTEGER &&
              !*type_->logicalType_->get_INTEGER().isSigned();
          const int32_t* dict = reinterpret_cast<const int32_t*>(dictPtr);
          for (int32_t i = 0; i < numValues; ++i) {
            const int64_t value = isUnsigned
                ? static_cast<int64_t>(static_cast<uint32_t>(dict[i]))
                : static_cast<int64_t>(dict[i]);
            if (common::applyFilter(*filter, value)) {
              return true;
            }
          }
          return false;
        }
        return testDict.operator()<int64_t>();
      case TypeKind::INTEGER:
        return testDict.operator()<int32_t>();
      case TypeKind::VARCHAR:
      case TypeKind::VARBINARY:
        return testDict.operator()<StringView>();
      case TypeKind::REAL:
        return testDict.operator()<float>();
      case TypeKind::DOUBLE:
        return testDict.operator()<double>();
      case TypeKind::BOOLEAN:
        return testDict.operator()<bool>();
      default:
        return true; // Conservative fallback.
    }
  }();
  if (!anyValuePasses && filter->testNull()) {
    anyValuePasses = true;
  }
  return anyValuePasses;
}

std::unique_ptr<dwio::common::DictionaryValues>
ParquetData::readDictionaryPageForFiltering(
    uint32_t rowGroupId,
    const ColumnChunkMetaDataPtr& columnChunk) {
  auto inputStream = getInputStream(rowGroupId, columnChunk);
  if (!inputStream) {
    return std::make_unique<dwio::common::DictionaryValues>();
  }

  // Pass the dictionary page size as chunkSize: PageReader only reads the
  // dictionary page header and body here, never seeks into data pages.
  // getInputStream has already validated dataPageOffset > dictionaryPageOffset.
  const uint64_t dictPageSize =
      columnChunk.dataPageOffset() - columnChunk.dictionaryPageOffset();
  auto pageReader = std::make_unique<PageReader>(
      std::move(inputStream),
      pool_,
      type_,
      columnChunk.compression(),
      dictPageSize,
      stats_,
      sessionTimezone_);
  // The first page of a dictionary-encoded column chunk is expected to be a
  // dictionary page. If it is not, skip dictionary filtering for this row group
  // rather than failing the query.
  auto pageHeader = pageReader->readPageHeader();
  if (*pageHeader.type() != thrift::PageType::DICTIONARY_PAGE) {
    return std::make_unique<dwio::common::DictionaryValues>();
  }

  pageReader->prepareDictionary(pageHeader);
  const auto& dict = pageReader->dictionary();
  return std::make_unique<dwio::common::DictionaryValues>(dict);
}

std::unique_ptr<dwio::common::SeekableInputStream> ParquetData::getInputStream(
    uint32_t rowGroupId,
    const ColumnChunkMetaDataPtr& columnChunk) {
  // Row group pruning only needs the dictionary page, so request just that
  // region instead of the full column chunk. The dictionary page lies in
  // [dictionaryPageOffset, dataPageOffset).
  if (!bufferedInput_ || !columnChunk.hasDictionaryPageOffset() ||
      columnChunk.dictionaryPageOffset() < 4) {
    return nullptr;
  }
  const uint64_t dictPageOffset = columnChunk.dictionaryPageOffset();
  // The dictionary page occupies [dictionaryPageOffset, dataPageOffset). If the
  // offsets are inconsistent, skip dictionary filtering (conservative include)
  // instead of computing a bogus size.
  if (columnChunk.dataPageOffset() <= dictPageOffset) {
    return nullptr;
  }
  const uint64_t dictPageSize = columnChunk.dataPageOffset() - dictPageOffset;

  // Use BufferedInput::read() rather than enqueue()+load(): load() clears all
  // previously enqueued/loaded regions on the shared BufferedInput, which would
  // corrupt subsequent reads from other readers using the same input. read()
  // either returns the data from the cache if already buffered, or reads
  // directly from the file without mutating BufferedInput state.
  return bufferedInput_->read(
      dictPageOffset, dictPageSize, dwio::common::LogType::STRIPE);
}

std::pair<int64_t, int64_t> ParquetData::getRowGroupRegion(
    uint32_t index) const {
  auto rowGroup = fileMetaDataPtr_.rowGroup(index);

  VELOX_CHECK_GT(rowGroup.numColumns(), 0);
  auto fileOffset = (rowGroup.hasFileOffset() && rowGroup.fileOffset() != 0)
      ? rowGroup.fileOffset()
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
