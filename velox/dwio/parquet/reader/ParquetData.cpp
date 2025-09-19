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
<<<<<<< HEAD
=======
#include "velox/dwio/parquet/reader/PageReader.h"
#include "velox/dwio/parquet/crypto/CryptoFactory.h"
#include "velox/common/base/Counters.h"
#include "velox/common/base/StatsReporter.h"
>>>>>>> 20c7dd067 (feat: support parquet dictionary filter based rowgroup skipping)

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
  // Start row group filtering process
  LOG(INFO) << "🔍 Starting row group filtering for column '" << type_->name_ << "'";
  LOG(INFO) << "   Total row groups to evaluate: " << fileMetaDataPtr_.numRowGroups();
  LOG(INFO) << "   ScanSpec has filter: " << (scanSpec.hasFilter() ? "YES" : "NO");
  LOG(INFO) << "   ScanSpec metadata filters: " << scanSpec.numMetadataFilters();
  if (scanSpec.filter()) {
    LOG(INFO) << "   Primary filter: " << scanSpec.filter()->toString();
  }

  auto parquetStatsContext =
      reinterpret_cast<const ParquetStatsContext*>(&writerContext);
  if (type_->parquetType_.has_value() &&
      parquetStatsContext->shouldIgnoreStatistics(
          type_->parquetType_.value())) {
    LOG(INFO) << "  Ignoring statistics for parquet type";
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

  // Begin row group evaluation
  LOG(INFO) << "";
  LOG(INFO) << "📊 Evaluating " << fileMetaDataPtr_.numRowGroups() << " row groups for filtering...";

  if (scanSpec.filter() || scanSpec.numMetadataFilters() > 0) {
    int includedCount = 0;
    int excludedCount = 0;

    for (auto i = 0; i < fileMetaDataPtr_.numRowGroups(); ++i) {
      bool rowGroupFiltered = false;

      // Apply primary ScanSpec filter
      if (scanSpec.filter() && !rowGroupMatches(i, scanSpec.filter())) {
        LOG(INFO) << "❌ RowGroup " << i << " EXCLUDED by primary filter";
        bits::setBit(result.filterResult.data(), i);
        rowGroupFiltered = true;
        excludedCount++;
        continue;
      }

      // Apply metadata filters
      for (int j = 0; j < scanSpec.numMetadataFilters(); ++j) {
        auto* metadataFilter = scanSpec.metadataFilterAt(j);
        if (!rowGroupMatches(i, metadataFilter)) {
          LOG(INFO) << "❌ RowGroup " << i << " EXCLUDED by metadata filter " << j;
          bits::setBit(
              result.metadataFilterResults[metadataFiltersStartIndex + j]
                  .second.data(),
              i);
        }
      }

      if (!rowGroupFiltered) {
        LOG(INFO) << "✅ RowGroup " << i << " INCLUDED (passed all filters)";
        includedCount++;
      }
    }

    LOG(INFO) << "";
    LOG(INFO) << "📈 Row group filtering summary:";
    LOG(INFO) << "   Total evaluated: " << fileMetaDataPtr_.numRowGroups();
    LOG(INFO) << "   Included: " << includedCount;
    LOG(INFO) << "   Excluded: " << excludedCount;
    LOG(INFO) << "   Filter efficiency: " << (excludedCount * 100.0 / fileMetaDataPtr_.numRowGroups()) << "% filtered out";
  } else {
    LOG(INFO) << "ℹ️  No filters configured - all " << fileMetaDataPtr_.numRowGroups() << " row groups will be processed";
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

  // Start row group filtering evaluation
  LOG(INFO) << "=== Evaluating RowGroup " << rowGroupId << " ===";
  LOG(INFO) << "  Column: " << column << " ('" << type_->name_ << "', type: " << type->toString() << ")";
  LOG(INFO) << "  Filter: " << filter->toString();

  auto columnChunk = rowGroup.columnChunk(column);

  // Step 1: Check for bloom filter availability (most precise filtering)
  if (columnChunk.hasBloomFilter()) {
    LOG(INFO) << "  [BLOOM FILTER] Available at offset: " << columnChunk.bloom_filter_offset().value();
    LOG(INFO) << "  [BLOOM FILTER] TODO: Bloom filter evaluation not yet implemented, falling back to statistics";
  } else {
    LOG(INFO) << "  [BLOOM FILTER] Not available for this column";
  }

  if (columnChunk.hasMetadata() && columnChunk.hasStatistics()) {
    LOG(INFO) << "  [STATISTICS] Column metadata and statistics available";

    auto columnStats = columnChunk.getColumnStatistics(type, rowGroup.numRows());

    // Log statistics information for debugging
    auto stringStats = dynamic_cast<dwio::common::StringColumnStatistics*>(columnStats.get());
    if (stringStats) {
      if (stringStats->getMinimum().has_value() && stringStats->getMaximum().has_value()) {
        LOG(INFO) << "  [STATISTICS] String range: ['" << stringStats->getMinimum().value()
                  << "' ... '" << stringStats->getMaximum().value() << "']";
      } else {
        LOG(INFO) << "  [STATISTICS] String min/max values not available";
      }
    } else {
      LOG(INFO) << "  [STATISTICS] Non-string column or statistics not available";
    }

    // Step 2: Apply basic statistics-based filtering
    LOG(INFO) << "  [STATISTICS] Evaluating filter against column statistics...";
    bool statisticsResult = testFilter(filter, columnStats.get(), rowGroup.numRows(), type);
    if (!statisticsResult) {
      LOG(INFO) << "  [RESULT] ❌ Row group " << rowGroupId << " EXCLUDED by statistics filtering";
      return false;
    }
    LOG(INFO) << "  [STATISTICS] ✅ Passed statistics filtering, proceeding to dictionary check...";

    // Step 3: Analyze dictionary encoding for enhanced filtering
    LOG(INFO) << "  [DICTIONARY] Analyzing encoding information for dictionary filtering...";
    auto encodings = columnChunk.getEncoding();

    // Check if dictionary page is available (from metadata or encodings)
    bool hasDictionaryPage = columnChunk.hasDictionaryPageOffset();
    if (!hasDictionaryPage) {
      // Fallback: detect dictionary from encoding types
      for (const auto& encoding : encodings) {
        if (encoding == thrift::Encoding::PLAIN_DICTIONARY ||
            encoding == thrift::Encoding::RLE_DICTIONARY) {
          hasDictionaryPage = true;
          LOG(INFO) << "  [DICTIONARY] Dictionary page detected from encodings (offset not in metadata)";
          break;
        }
      }
    }

    // Apply Presto's logic to determine if dictionary filtering is safe
    bool isOnlyDictionaryEncodingPages = isOnlyDictionaryEncodingPagesImpl(encodings);
    bool canUseDictionaryFiltering = hasDictionaryPage && isOnlyDictionaryEncodingPages;

    // Log encoding analysis for debugging
    LOG(INFO) << "  [DICTIONARY] Encoding analysis:";
    LOG(INFO) << "    - Total encodings found: " << encodings.size();
    for (size_t i = 0; i < encodings.size(); ++i) {
      LOG(INFO) << "    - Encoding[" << i << "]: " << static_cast<int>(encodings[i]);
    }
    LOG(INFO) << "    - Has dictionary page: " << (hasDictionaryPage ? "YES" : "NO");
    LOG(INFO) << "    - Only dictionary encoding pages: " << (isOnlyDictionaryEncodingPages ? "YES" : "NO");
    LOG(INFO) << "    - Can use dictionary filtering: " << (canUseDictionaryFiltering ? "YES" : "NO");

    // Step 4: Apply dictionary filtering if applicable
    if (canUseDictionaryFiltering) {
      if (columnChunk.hasDictionaryPageOffset()) {
        LOG(INFO) << "  [DICTIONARY] Dictionary page available at offset: " << columnChunk.dictionaryPageOffset();
      } else {
        LOG(INFO) << "  [DICTIONARY] Dictionary page detected from encodings (offset not in metadata)";
      }

      LOG(INFO) << "  [DICTIONARY] Evaluating filter against dictionary values...";
      bool dictionaryResult = testFilterAgainstDictionary(rowGroupId, filter, columnChunk);
      if (!dictionaryResult) {
        LOG(INFO) << "  [RESULT] ❌ Row group " << rowGroupId << " EXCLUDED by dictionary filtering";
        return false;
      }
      LOG(INFO) << "  [DICTIONARY] ✅ Passed dictionary filtering";
    } else {
      // Log why dictionary filtering cannot be used
      if (!hasDictionaryPage) {
        LOG(INFO) << "  [DICTIONARY] ⚠️  Dictionary filtering skipped: No dictionary page available";
      } else if (!isOnlyDictionaryEncodingPages) {
        LOG(INFO) << "  [DICTIONARY] ⚠️  Dictionary filtering skipped: Mixed encodings detected (Presto safety logic)";
      }
    }

    LOG(INFO) << "  [RESULT] ✅ Row group " << rowGroupId << " INCLUDED (passed all applicable filters)";
    return true;
  }

  // Fallback case: no metadata or statistics available
  LOG(INFO) << "  [FALLBACK] ⚠️  No metadata or statistics available for column";
  LOG(INFO) << "  [RESULT] ✅ Row group " << rowGroupId << " INCLUDED (no filtering data available - conservative approach)";
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
bool ParquetData::isOnlyDictionaryEncodingPagesImpl(const std::vector<thrift::Encoding::type>& encodings) {
  // TODO: update to use EncodingStats in ColumnChunkMetaData when available

  // Check if PLAIN_DICTIONARY is present
  bool hasPlainDictionary = false;
  for (const auto& encoding : encodings) {
    if (encoding == thrift::Encoding::PLAIN_DICTIONARY) {
      hasPlainDictionary = true;
      break;
    }
  }

  if (hasPlainDictionary) {
    // PLAIN_DICTIONARY was present, which means at least one page was
    // dictionary-encoded and 1.0 encodings are used
    int expectedEncodings = 1; // for PLAIN_DICTIONARY

    // RLE and BIT_PACKED are only used for repetition or definition levels
    bool hasRLE = false;
    bool hasBitPacked = false;
    for (const auto& encoding : encodings) {
      if (encoding == thrift::Encoding::RLE) {
        hasRLE = true;
      } else if (encoding == thrift::Encoding::BIT_PACKED) {
        hasBitPacked = true;
      }
    }

    if (hasRLE) {
      expectedEncodings += 1;
    }
    if (hasBitPacked) {
      expectedEncodings += 1;
    }

    if (encodings.size() > expectedEncodings) {
      return false; // no encodings other than dictionary or rep/def levels
    }

    return true;
  } else {
    // if PLAIN_DICTIONARY wasn't present, then either the column is not
    // dictionary-encoded, or the 2.0 encoding, RLE_DICTIONARY, was used.
    // for 2.0, this cannot determine whether a page fell back without
    // page encoding stats
    return false;
  }
}

bool ParquetData::testFilterAgainstDictionary(
    uint32_t rowGroupId,
    common::Filter* filter,
    const ColumnChunkMetaDataPtr& columnChunk) {

  if (!filter) {
    LOG(INFO) << "    [DICT-TEST] No filter provided, dictionary test passes";
    return true;
  }

  LOG(INFO) << "    [DICT-TEST] 🔍 Starting dictionary filtering for row group " << rowGroupId;
  LOG(INFO) << "    [DICT-TEST] Filter type: " << filter->toString();

  try {
    // Step 1: Load dictionary data from Parquet file
    LOG(INFO) << "    [DICT-TEST] Step 1: Loading dictionary data...";

    const dwio::common::DictionaryValues* finalDictionary = nullptr;
    try {
      auto dictionaryValues = readDictionaryPageForFiltering(rowGroupId, columnChunk);
      if (!dictionaryValues.values || dictionaryValues.numValues == 0) {
        LOG(INFO) << "    [DICT-TEST] ⚠️  Dictionary is empty or unavailable, falling back (conservative include)";
        return true;
      }

      // Store the dictionary for this filtering operation
      tempDictionary_ = std::move(dictionaryValues);
      finalDictionary = &tempDictionary_;
      LOG(INFO) << "    [DICT-TEST] ✅ Dictionary loaded: " << finalDictionary->numValues << " values";

    } catch (const std::exception& e) {
      LOG(WARNING) << "    [DICT-TEST] ❌ Failed to load dictionary: " << e.what() << " (falling back to include)";
      return true;
    }

    // Step 2: Test filter against dictionary values
    auto parquetType = type_->parquetType_.value();
    auto numValues = finalDictionary->numValues;

    LOG(INFO) << "    [DICT-TEST] Step 2: Testing " << numValues << " dictionary values against filter";
    LOG(INFO) << "    [DICT-TEST] Parquet type: " << static_cast<int>(parquetType);
    bool anyValuePasses = false;

    switch (parquetType) {
      case thrift::Type::INT32: {
        LOG(INFO) << "    [DICT-TEST] Testing INT32 dictionary values...";
        auto* values = reinterpret_cast<const int32_t*>(finalDictionary->values->as<void>());
        for (int32_t i = 0; i < numValues; ++i) {
          if (filter->testInt64(values[i])) {
            anyValuePasses = true;
            LOG(INFO) << "    [DICT-TEST] ✅ Match found: value=" << values[i] << " at index=" << i << " (early exit)";
            break; // Early exit - we only need one match
          }
        }
        if (!anyValuePasses) {
          LOG(INFO) << "    [DICT-TEST] ❌ No matches found in " << numValues << " INT32 values";
        }
        break;
      }
      case thrift::Type::INT64: {
        LOG(INFO) << "    [DICT-TEST] Testing INT64 dictionary values...";
        auto* values = reinterpret_cast<const int64_t*>(finalDictionary->values->as<void>());
        for (int32_t i = 0; i < numValues; ++i) {
          if (filter->testInt64(values[i])) {
            anyValuePasses = true;
            LOG(INFO) << "    [DICT-TEST] ✅ Match found: value=" << values[i] << " at index=" << i << " (early exit)";
            break; // Early exit - we only need one match
          }
        }
        if (!anyValuePasses) {
          LOG(INFO) << "    [DICT-TEST] ❌ No matches found in " << numValues << " INT64 values";
        }
        break;
      }
      case thrift::Type::BYTE_ARRAY: {
        LOG(INFO) << "    [DICT-TEST] Testing BYTE_ARRAY (string) dictionary values...";
        auto* stringViews = reinterpret_cast<const StringView*>(finalDictionary->values->as<void>());
        for (int32_t i = 0; i < numValues; ++i) {
          const auto& sv = stringViews[i];
          if (filter->testBytes(sv.data(), sv.size())) {
            anyValuePasses = true;
            LOG(INFO) << "    [DICT-TEST] ✅ Match found: value='" << sv.getString() << "' at index=" << i << " (early exit)";
            break; // Early exit - we only need one match
          }
        }
        if (!anyValuePasses) {
          LOG(INFO) << "    [DICT-TEST] ❌ No matches found in " << numValues << " BYTE_ARRAY values";
        }
        break;
      }
      case thrift::Type::FIXED_LEN_BYTE_ARRAY: {
        LOG(INFO) << "    [DICT-TEST] Testing FIXED_LEN_BYTE_ARRAY dictionary values...";
        if (type_->type()->isVarchar() || type_->type()->isVarbinary()) {
          // Treat as strings with StringViews
          auto* stringViews = reinterpret_cast<const StringView*>(finalDictionary->values->as<void>());
          for (int32_t i = 0; i < numValues; ++i) {
            const auto& sv = stringViews[i];
            if (filter->testBytes(sv.data(), sv.size())) {
              anyValuePasses = true;
              LOG(INFO) << "    [DICT-TEST] ✅ Match found: value='" << sv.getString() << "' at index=" << i << " (early exit)";
              break; // Early exit - we only need one match
            }
          }
          if (!anyValuePasses) {
            LOG(INFO) << "    [DICT-TEST] ❌ No matches found in " << numValues << " FIXED_LEN_BYTE_ARRAY values";
          }
        } else {
          LOG(INFO) << "    [DICT-TEST] ⚠️  FIXED_LEN_BYTE_ARRAY for non-string type, falling back (conservative include)";
          return true;
        }
        break;
      }
      default:
        LOG(INFO) << "    [DICT-TEST] ⚠️  Unsupported parquet type " << static_cast<int>(parquetType) << ", falling back (conservative include)";
        return true;
    }

    // Step 3: Evaluate results and make filtering decision
    LOG(INFO) << "    [DICT-TEST] Step 3: Dictionary filtering results:";
    LOG(INFO) << "    [DICT-TEST]   - Total dictionary values scanned: " << numValues;
    LOG(INFO) << "    [DICT-TEST]   - Early exit optimization: " << (anyValuePasses ? "Used (found match)" : "Not applicable (no matches)");

    if (anyValuePasses) {
      LOG(INFO) << "    [DICT-TEST] ✅ RESULT: Row group INCLUDED (dictionary contains at least one matching value)";
    } else {
      LOG(INFO) << "    [DICT-TEST] ❌ RESULT: Row group EXCLUDED (no dictionary values match filter)";
      LOG(INFO) << "    [DICT-TEST] 📝 Safe to exclude because ALL data pages use dictionary encoding";
    }

    return anyValuePasses;

  } catch (const std::exception& e) {
    LOG(WARNING) << "    [DICT-TEST] ❌ EXCEPTION during dictionary filtering: " << e.what();
    LOG(WARNING) << "    [DICT-TEST] 🛡️  Falling back to conservative include for safety";
    return true;
  }
}

// Read dictionary page directly for row group filtering (like Presto's dictionaryPredicatesMatch)
dwio::common::DictionaryValues ParquetData::readDictionaryPageForFiltering(
    uint32_t rowGroupId,
    const ColumnChunkMetaDataPtr& columnChunk) {

  LOG(INFO) << "      [DICT-READ] 📖 Loading dictionary page for row group " << rowGroupId;

  dwio::common::DictionaryValues emptyDict{};

  // Step 1: Detect dictionary page availability
  LOG(INFO) << "      [DICT-READ] Step 1: Detecting dictionary page availability...";
  bool hasDictionaryPage = columnChunk.hasDictionaryPageOffset();

  if (!hasDictionaryPage) {
    LOG(INFO) << "      [DICT-READ] No dictionary offset in metadata, checking encodings...";
    auto encodings = columnChunk.getEncoding();
    for (const auto& encoding : encodings) {
      if (encoding == thrift::Encoding::PLAIN_DICTIONARY ||
          encoding == thrift::Encoding::RLE_DICTIONARY) {
        hasDictionaryPage = true;
        LOG(INFO) << "      [DICT-READ] ✅ Dictionary page detected from encodings";
        break;
      }
    }
  } else {
    LOG(INFO) << "      [DICT-READ] ✅ Dictionary page offset found in metadata";
  }

  if (!hasDictionaryPage) {
    LOG(INFO) << "      [DICT-READ] ❌ No dictionary page available (checked metadata + encodings)";
    return emptyDict;
  }

  // Step 2: Determine dictionary page offset
  LOG(INFO) << "      [DICT-READ] Step 2: Determining dictionary page offset...";
  int64_t dictionaryPageOffset = -1;

  if (columnChunk.hasDictionaryPageOffset()) {
    dictionaryPageOffset = columnChunk.dictionaryPageOffset();
    LOG(INFO) << "      [DICT-READ] ✅ Using metadata offset: " << dictionaryPageOffset;
  } else {
    LOG(INFO) << "      [DICT-READ] ⚠️  No offset in metadata, attempting to locate dictionary page...";

    // Scan from column start to find dictionary page (like Presto does)
    dictionaryPageOffset = findDictionaryPageOffset(columnChunk);

    if (dictionaryPageOffset < 0) {
      LOG(INFO) << "      [DICT-READ] ❌ Failed to locate dictionary page by scanning";
      return emptyDict;
    }

    LOG(INFO) << "      [DICT-READ] ✅ Located dictionary page at offset: " << dictionaryPageOffset;
  }

  // Step 3: Read and parse dictionary page
  LOG(INFO) << "      [DICT-READ] Step 3: Reading dictionary page data...";

  try {
    // Create input stream for the column chunk
    LOG(INFO) << "      [DICT-READ] Creating input stream for column chunk...";
    auto stream = getInputStream(rowGroupId, columnChunk);
    if (!stream) {
      LOG(WARNING) << "      [DICT-READ] ❌ Failed to create input stream for column";
      return emptyDict;
    }

    // TODO: Implement proper seeking to dictionary page offset
    // stream->seekToPosition requires PositionProvider implementation
    LOG(INFO) << "      [DICT-READ] ⚠️  TODO: Seeking to dictionary page offset " << dictionaryPageOffset << " (not yet implemented)";

    // Read stream data (header + dictionary data)
    LOG(INFO) << "      [DICT-READ] Reading stream data...";
    const void* buffer;
    int32_t bufferSize;
    if (!stream->Next(&buffer, &bufferSize)) {
      LOG(WARNING) << "      [DICT-READ] ❌ Failed to read data from stream";
      return emptyDict;
    }

    LOG(INFO) << "      [DICT-READ] ✅ Read " << bufferSize << " bytes from stream";

    const char* bufferStart = reinterpret_cast<const char*>(buffer);
    const char* bufferEnd = bufferStart + bufferSize;

    // Create thrift transport and protocol to read header
    std::shared_ptr<thrift::ThriftTransport> transport =
        std::make_shared<thrift::ThriftStreamingTransport>(
            stream.get(), bufferStart, bufferEnd);
    apache::thrift::protocol::TCompactProtocolT<thrift::ThriftTransport> protocol(
        transport);

    // Read the page header
    thrift::PageHeader pageHeader;
    uint64_t headerBytes = pageHeader.read(&protocol);

    LOG(INFO) << "        Successfully read page header, type: " << static_cast<int>(pageHeader.type)
              << ", header bytes: " << headerBytes;

    // Verify it's a dictionary page
    if (pageHeader.type != thrift::PageType::DICTIONARY_PAGE) {
      LOG(INFO) << "        First page is not a dictionary page, type: " << static_cast<int>(pageHeader.type);
      return emptyDict;
    }

    LOG(INFO) << "        Confirmed dictionary page with " << pageHeader.dictionary_page_header.num_values << " values";

    // Check compression info
    auto compressionType = columnChunk.compression();
    LOG(INFO) << "        Column chunk compression: " << static_cast<int>(compressionType);
    LOG(INFO) << "        Page compressed size: " << pageHeader.compressed_page_size;
    LOG(INFO) << "        Page uncompressed size: " << pageHeader.uncompressed_page_size;

    // Log compression type name for clarity
    std::string compressionName = "UNKNOWN";
    switch (compressionType) {
      case common::CompressionKind::CompressionKind_NONE: compressionName = "NONE"; break;
      case common::CompressionKind::CompressionKind_SNAPPY: compressionName = "SNAPPY"; break;
      case common::CompressionKind::CompressionKind_GZIP: compressionName = "GZIP"; break;
      case common::CompressionKind::CompressionKind_LZO: compressionName = "LZO"; break;
      case common::CompressionKind::CompressionKind_LZ4: compressionName = "LZ4"; break;
      case common::CompressionKind::CompressionKind_ZSTD: compressionName = "ZSTD"; break;
      default: compressionName = "UNKNOWN(" + std::to_string(static_cast<int>(compressionType)) + ")"; break;
    }
    LOG(INFO) << "        Compression type: " << compressionName;

    // The issue might be that the ThriftTransport consumed the exact header bytes
    // Let's try to get the remaining data from the transport itself
    // After reading the header, the transport should be positioned at the start of data

    // Try to get the current buffer position from transport
    const char* remainingData = nullptr;
    int32_t remainingSize = 0;

    // The transport should have the current position after reading the header
    // Let's use the original calculation but verify the data looks like ZSTD
    const char* dictDataStart = bufferStart + headerBytes;
    auto dictDataSize = pageHeader.compressed_page_size;

    LOG(INFO) << "        Using header-based offset: " << headerBytes;

    LOG(INFO) << "        Dictionary data: start offset=" << headerBytes
              << ", compressed size=" << dictDataSize
              << ", available=" << (bufferEnd - dictDataStart);

    if (dictDataStart + dictDataSize > bufferEnd) {
      LOG(INFO) << "        Insufficient data in buffer for dictionary";
      return emptyDict;
    }

    // Check if we need decompression
    if (pageHeader.compressed_page_size != pageHeader.uncompressed_page_size) {
      LOG(INFO) << "        Dictionary data is compressed, decompressing...";

      // Debug: show first 16 bytes of compressed data and check for ZSTD magic
      LOG(INFO) << "        First 16 bytes of compressed data:";
      for (int i = 0; i < std::min(16, (int)pageHeader.compressed_page_size); i++) {
        LOG(INFO) << "          [" << i << "] = " << (int)(unsigned char)dictDataStart[i]
                  << " (0x" << std::hex << (unsigned char)dictDataStart[i] << std::dec << ")";
      }

      // Check if this looks like ZSTD magic bytes (0x28 0xB5 0x2F 0xFD)
      bool hasZstdMagic = false;
      if (pageHeader.compressed_page_size >= 4) {
        uint32_t magic = *reinterpret_cast<const uint32_t*>(dictDataStart);
        hasZstdMagic = (magic == 0xFD2FB528); // ZSTD magic in little-endian
        LOG(INFO) << "        ZSTD magic check: " << (hasZstdMagic ? "FOUND" : "NOT FOUND")
                  << " (got 0x" << std::hex << magic << std::dec << ", expected 0xFD2FB528)";
      }

      // If no ZSTD magic, try different offsets
      if (!hasZstdMagic) {
        LOG(INFO) << "        Searching for ZSTD magic at different offsets...";
        for (int offset = 0; offset < std::min(32, (int)bufferSize - 4); offset++) {
          uint32_t magic = *reinterpret_cast<const uint32_t*>(bufferStart + offset);
          if (magic == 0xFD2FB528) {
            LOG(INFO) << "        FOUND ZSTD magic at offset " << offset << "!";
            dictDataStart = bufferStart + offset;
            dictDataSize = pageHeader.compressed_page_size;
            break;
          }
        }
      }

      // Decompress using the same logic as PageReader::decompressData
      std::unique_ptr<dwio::common::SeekableInputStream> inputStream =
          std::make_unique<dwio::common::SeekableArrayInputStream>(
              dictDataStart, pageHeader.compressed_page_size, 0);

      auto streamDebugInfo = fmt::format("Dictionary decompression for column {}", type_->column());
      std::unique_ptr<dwio::common::SeekableInputStream> decompressedStream =
          dwio::common::compression::createDecompressor(
              columnChunk.compression(),
              std::move(inputStream),
              pageHeader.uncompressed_page_size,
              pool_,
              getParquetDecompressionOptions(columnChunk.compression()),
              streamDebugInfo,
              nullptr,
              true,
              pageHeader.compressed_page_size);

      // Read decompressed data into a buffer
      auto decompressedBuffer = AlignedBuffer::allocate<char>(pageHeader.uncompressed_page_size, &pool_);
      decompressedStream->readFully(
          decompressedBuffer->asMutable<char>(), pageHeader.uncompressed_page_size);

      LOG(INFO) << "        Successfully decompressed " << pageHeader.compressed_page_size
                << " -> " << pageHeader.uncompressed_page_size << " bytes";

      // Parse the decompressed data
      return parseDictionaryFromBuffer(pageHeader, decompressedBuffer->as<char>(), pageHeader.uncompressed_page_size);
    } else {
      LOG(INFO) << "        Dictionary data is uncompressed, parsing directly";
      // Parse dictionary data directly from buffer
      return parseDictionaryFromBuffer(pageHeader, dictDataStart, dictDataSize);
    }

  } catch (const std::exception& e) {
    LOG(WARNING) << "      [DICT-READ] ❌ Exception reading dictionary page: " << e.what();
    return emptyDict;
  }
}

// Find dictionary page offset by checking if first page is a dictionary page (like Presto)
int64_t ParquetData::findDictionaryPageOffset(const ColumnChunkMetaDataPtr& columnChunk) {
  LOG(INFO) << "        [DICT-LOCATE] 🔍 Attempting to locate dictionary page (Presto-style scanning)...";

  // Get column chunk start position - dictionary pages come first if they exist
  auto columnStartOffset = columnChunk.dataPageOffset();

  // Sanity check: if we have dictionary offset in metadata, use it
  // (This shouldn't happen as caller already checked, but defensive programming)
  if (columnChunk.hasDictionaryPageOffset()) {
    auto dictOffset = columnChunk.dictionaryPageOffset();
    LOG(INFO) << "        [DICT-LOCATE] ✅ Found dictionary offset in metadata: " << dictOffset;
    return dictOffset;
  }

  LOG(INFO) << "        [DICT-LOCATE] Column chunk starts at offset: " << columnStartOffset;
  LOG(INFO) << "        [DICT-LOCATE] Using Presto assumption: first page is dictionary page";

  // TODO: Implement proper page header verification
  LOG(INFO) << "        [DICT-LOCATE] ⚠️  TODO: Should verify first page is DICTIONARY_PAGE by reading header";
  LOG(INFO) << "        [DICT-LOCATE] 📝 Current implementation: assume first page = dictionary (Presto approach)";

  // Conservative approach: assume the first page is the dictionary page (like Presto)
  LOG(INFO) << "        [DICT-LOCATE] ✅ Returning assumed dictionary offset: " << columnStartOffset;
  return columnStartOffset;
}

// Helper method to get input stream for column chunk
std::unique_ptr<dwio::common::SeekableInputStream> ParquetData::getInputStream(
    uint32_t rowGroupId, const ColumnChunkMetaDataPtr& columnChunk) {
  LOG(INFO) << "        Creating input stream for row group " << rowGroupId;

  try {
    // Use the existing streams_ infrastructure that ParquetData already has
    if (rowGroupId < streams_.size() && streams_[rowGroupId]) {
      LOG(INFO) << "        Using existing stream for row group " << rowGroupId;
      // Move the existing stream (transfer ownership)
      return std::move(streams_[rowGroupId]);
    }

    // Create new stream if not available using BufferedInput
    if (!bufferedInput_) {
      LOG(INFO) << "        No BufferedInput available to create stream";
      return nullptr;
    }

    LOG(INFO) << "        No existing stream available for row group " << rowGroupId;
    LOG(INFO) << "        Creating new stream for column chunk at offset " << columnChunk.dataPageOffset();

    // Calculate the read parameters similar to enqueueRowGroup
    uint64_t chunkReadOffset = columnChunk.dataPageOffset();
    if (columnChunk.hasDictionaryPageOffset() && columnChunk.dictionaryPageOffset() >= 4) {
      // Include dictionary page in the read
      chunkReadOffset = columnChunk.dictionaryPageOffset();
      LOG(INFO) << "        Using dictionary page offset: " << chunkReadOffset;
    }

    // Get column path for encryption check (same logic as enqueueRowGroup)
    ColumnPath columnPath = columnChunk.getColumnPath();
    std::string path = columnPath.toDotString();

    // Use the same logic as enqueueRowGroup for determining read size
    uint64_t readSize =
        (columnChunk.compression() == common::CompressionKind::CompressionKind_NONE &&
         (!fileDecryptor_ || !fileDecryptor_->getColumnCryptoMetadata(path)->isEncrypted()))
        ? columnChunk.totalUncompressedSize()
        : columnChunk.totalCompressedSize();

    // Add validation and debugging
    LOG(INFO) << "        Calculated read parameters: offset=" << chunkReadOffset
              << " size=" << readSize
              << " compression=" << static_cast<int>(columnChunk.compression())
              << " totalCompressed=" << columnChunk.totalCompressedSize()
              << " totalUncompressed=" << columnChunk.totalUncompressedSize();

    // Create stream identifier
    auto id = dwio::common::StreamIdentifier(type_->column());

    // Validate the read parameters before enqueueing
    if (readSize == 0) {
      LOG(INFO) << "        Invalid read size (0), cannot create stream";
      return nullptr;
    }

    // Try to enqueue and get the stream
    try {
      auto stream = bufferedInput_->enqueue({chunkReadOffset, readSize}, &id);
      LOG(INFO) << "        Successfully enqueued stream for offset=" << chunkReadOffset
                << " size=" << readSize;

      // Load the data into memory
      bufferedInput_->load(dwio::common::LogType::STRIPE);
      LOG(INFO) << "        Successfully loaded data into BufferedInput";

      return stream;
    } catch (const std::exception& e) {
      LOG(INFO) << "        Failed to enqueue stream: " << e.what();
      LOG(INFO) << "        This might indicate the row group data is not available in BufferedInput";
      return nullptr;
    }

  } catch (const std::exception& e) {
    LOG(INFO) << "        Exception creating input stream: " << e.what();
    return nullptr;
  }
}

// Helper method to read Parquet page header (thrift format)
std::optional<thrift::PageHeader> ParquetData::readPageHeader(dwio::common::SeekableInputStream* stream) {
  LOG(INFO) << "        Reading page header from stream";

  try {
    // Get some data from the stream to work with
    const void* buffer;
    int32_t size;
    if (!stream->Next(&buffer, &size)) {
      LOG(INFO) << "        Failed to get data from stream";
      return std::nullopt;
    }

    const char* bufferStart = reinterpret_cast<const char*>(buffer);
    const char* bufferEnd = bufferStart + size;

    // Create thrift transport and protocol (same as PageReader)
    std::shared_ptr<thrift::ThriftTransport> transport =
        std::make_shared<thrift::ThriftStreamingTransport>(
            stream, bufferStart, bufferEnd);
    apache::thrift::protocol::TCompactProtocolT<thrift::ThriftTransport> protocol(
        transport);

    // Read the page header
    thrift::PageHeader pageHeader;
    uint64_t readBytes = pageHeader.read(&protocol);

    LOG(INFO) << "        Successfully read page header, type: " << static_cast<int>(pageHeader.type)
              << ", bytes read: " << readBytes;

    return pageHeader;

  } catch (const std::exception& e) {
    LOG(INFO) << "        Exception reading page header: " << e.what();
    return std::nullopt;
  }
}

// Helper method to read and decode dictionary data
dwio::common::DictionaryValues ParquetData::parseDictionaryFromBuffer(
    const thrift::PageHeader& pageHeader,
    const char* dictData,
    size_t dictDataSize) {

  LOG(INFO) << "        Parsing dictionary data with " << pageHeader.dictionary_page_header.num_values << " values";

  dwio::common::DictionaryValues dict{};
  dict.numValues = pageHeader.dictionary_page_header.num_values;

  try {
    // For now, let's handle the most common case: BYTE_ARRAY (strings)
    auto parquetType = type_->parquetType_.value();
    if (parquetType == thrift::Type::BYTE_ARRAY) {
      // Allocate buffer for string views
      dict.values = AlignedBuffer::allocate<StringView>(dict.numValues, &pool_);
      auto values = dict.values->asMutable<StringView>();

      // Allocate buffer for string data and copy from input
      dict.strings = AlignedBuffer::allocate<char>(dictDataSize, &pool_);
      auto strings = dict.strings->asMutable<char>();
      memcpy(strings, dictData, dictDataSize);

      LOG(INFO) << "        Parsing " << dictDataSize << " bytes of BYTE_ARRAY dictionary data";

      // Check if we need to decompress the data first
      LOG(INFO) << "        Page header info:";
      LOG(INFO) << "          compressed_page_size: " << pageHeader.compressed_page_size;
      LOG(INFO) << "          uncompressed_page_size: " << pageHeader.uncompressed_page_size;

      // Debug: show first 16 bytes as decimal values (avoid hex formatting issues)
      LOG(INFO) << "        First 16 bytes (decimal): ";
      for (int i = 0; i < std::min(16, (int)dictDataSize); i++) {
        LOG(INFO) << "          [" << i << "] = " << (int)(unsigned char)strings[i];
      }

      // Parse the string data (length-prefixed format)
      // Format: [4-byte length][string data][4-byte length][string data]...
      auto header = strings;
      auto headerEnd = strings + dictDataSize;
      for (auto i = 0; i < dict.numValues; ++i) {
        if (header + sizeof(int32_t) > headerEnd) {
          LOG(WARNING) << "        Invalid dictionary data format at value " << i
                       << ": not enough data for length field";
          return dwio::common::DictionaryValues{};
        }

        // Read length as little-endian (Parquet uses little-endian)
        auto length = *reinterpret_cast<const int32_t*>(header);

        // Debug first few values
        if (i < 3) {
          LOG(INFO) << "        Value " << i << " raw bytes: "
                    << "0x" << std::hex << (unsigned char)header[0] << " "
                    << "0x" << std::hex << (unsigned char)header[1] << " "
                    << "0x" << std::hex << (unsigned char)header[2] << " "
                    << "0x" << std::hex << (unsigned char)header[3] << std::dec
                    << " -> length=" << length;
        }

        if (length < 0 || length > 1000000 || header + sizeof(int32_t) + length > headerEnd) {
          LOG(WARNING) << "        Invalid dictionary data format at value " << i
                       << ": invalid length " << length << " (0x" << std::hex << length << std::dec << ")";
          return dwio::common::DictionaryValues{};
        }

        values[i] = StringView(header + sizeof(int32_t), length);
        header += length + sizeof(int32_t);

        // Log first few values for debugging
        if (i < 5) {
          LOG(INFO) << "        Dict value " << i << ": length=" << length
                    << " value='" << std::string(values[i].data(), std::min(length, 50)) << "'";
        }
      }

      if (header != headerEnd) {
        LOG(WARNING) << "        Dictionary parsing ended at wrong position: expected "
                     << dictDataSize << ", got " << (header - strings);
      }

      LOG(INFO) << "        Successfully loaded " << dict.numValues << " string dictionary values";
      return dict;
    } else {
      LOG(INFO) << "        Dictionary type " << static_cast<int>(parquetType) << " not yet supported";
      return dwio::common::DictionaryValues{};
    }
  } catch (const std::exception& e) {
    LOG(INFO) << "        Exception parsing dictionary data: " << e.what();
    return dwio::common::DictionaryValues{};
  }
}

} // namespace facebook::velox::parquet
