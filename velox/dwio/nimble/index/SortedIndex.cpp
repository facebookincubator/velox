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
#include "velox/dwio/nimble/index/SortedIndex.h"

#include <algorithm>

#include <fmt/ranges.h>

#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/tablet/SortedIndexGenerated.h"

namespace facebook::nimble::index {

namespace {

bool columnsMatch(
    const std::vector<std::string>& indexColumns,
    const std::vector<std::string>& queryColumns) {
  if (indexColumns.size() != queryColumns.size()) {
    return false;
  }
  for (size_t i = 0; i < indexColumns.size(); ++i) {
    if (indexColumns[i] != queryColumns[i]) {
      return false;
    }
  }
  return true;
}

const serialization::SortedIndex* getSortedIndexRoot(
    const MetadataBuffer& metadata) {
  const auto* entry = flatbuffers::GetRoot<serialization::SortedIndex>(
      metadata.content().data());
  NIMBLE_CHECK_NOT_NULL(entry);
  return entry;
}

uint8_t getRowIdWidth(const MetadataBuffer& metadata) {
  const auto width = getSortedIndexRoot(metadata)->row_id_width();
  NIMBLE_CHECK(
      width >= 1 && width <= 4,
      "Invalid row ID width: {}",
      static_cast<int>(width));
  return width;
}

} // namespace

// ---------------------------------------------------------------------------
// SortedIndex
// ---------------------------------------------------------------------------

SortedIndex::SortedIndex(
    std::vector<std::string> columns,
    std::unique_ptr<MetadataBuffer> indexMetadata,
    std::shared_ptr<velox::dwio::common::BufferedInput> dataInput,
    velox::memory::MemoryPool* pool)
    : IndexLookup{IndexType::Sorted},
      columns_{std::move(columns)},
      indexMetadata_{std::move(indexMetadata)},
      pool_{pool},
      rowIdWidth_{getRowIdWidth(*indexMetadata_)},
      dataInput_{std::move(dataInput)},
      sortedIndex_{getSortedIndexRoot(*indexMetadata_)},
      numChunks_{sortedIndex_->chunk_keys()->size()} {}

std::unique_ptr<SortedIndex> SortedIndex::create(
    std::vector<std::string> columns,
    std::unique_ptr<MetadataBuffer> indexMetadata,
    std::shared_ptr<velox::dwio::common::BufferedInput> dataInput,
    velox::memory::MemoryPool* pool) {
  return std::unique_ptr<SortedIndex>(new SortedIndex(
      std::move(columns),
      std::move(indexMetadata),
      std::move(dataInput),
      pool));
}

std::string_view SortedIndex::chunkKey(uint32_t chunkIdx) const {
  return sortedIndex_->chunk_keys()->Get(chunkIdx)->string_view();
}

uint32_t SortedIndex::chunkOffset(uint32_t chunkIdx) const {
  return sortedIndex_->chunk_offsets()->Get(chunkIdx);
}

uint32_t SortedIndex::chunkSize(uint32_t chunkIdx) const {
  const uint32_t offset = chunkOffset(chunkIdx);
  const uint32_t nextOffset = chunkIdx + 1 < numChunks_
      ? chunkOffset(chunkIdx + 1)
      : sortedIndex_->key_stream_size();
  return nextOffset - offset;
}

uint32_t SortedIndex::rowOffset(uint32_t chunkIdx) const {
  return chunkIdx == 0 ? 0 : sortedIndex_->chunk_rows()->Get(chunkIdx - 1);
}

std::optional<uint32_t> SortedIndex::findChunkIndex(
    std::string_view key) const {
  const auto* chunkKeys = sortedIndex_->chunk_keys();
  const auto it = std::lower_bound(
      chunkKeys->begin(),
      chunkKeys->end(),
      key,
      [](const flatbuffers::String* chunkKey, std::string_view searchKey) {
        return chunkKey->string_view() < searchKey;
      });
  if (it == chunkKeys->end()) {
    return std::nullopt;
  }
  return static_cast<uint32_t>(std::distance(chunkKeys->begin(), it));
}

std::shared_ptr<DecodedKeyChunk> SortedIndex::loadChunk(
    uint32_t chunkIdx) const {
  const uint64_t streamOffset = sortedIndex_->key_stream_offset();
  const uint32_t chunkOffset = this->chunkOffset(chunkIdx);
  const uint32_t chunkSize = this->chunkSize(chunkIdx);

  return decodeKeyChunk(
      dataInput_->read(
          streamOffset + chunkOffset,
          chunkSize,
          velox::dwio::common::LogType::FOOTER),
      *pool_,
      chunkBuffer_);
}

std::string_view SortedIndex::extractKey(
    std::string_view compositeEntry) const {
  NIMBLE_DCHECK_GE(compositeEntry.size(), rowIdWidth_);
  return compositeEntry.substr(0, compositeEntry.size() - rowIdWidth_);
}

uint32_t SortedIndex::extractRowId(std::string_view compositeEntry) const {
  NIMBLE_DCHECK_GE(compositeEntry.size(), rowIdWidth_);
  const auto* data = reinterpret_cast<const uint8_t*>(
      compositeEntry.data() + compositeEntry.size() - rowIdWidth_);
  uint32_t rowId = 0;
  for (uint8_t i = 0; i < rowIdWidth_; ++i) {
    rowId = (rowId << 8) | data[i];
  }
  return rowId;
}

std::string SortedIndex::firstSeekKey(std::string_view key) const {
  std::string result(key);
  result.append(rowIdWidth_, '\0');
  return result;
}

std::string SortedIndex::lastSeekKey(std::string_view key) const {
  std::string result(key);
  result.append(rowIdWidth_, '\xFF');
  return result;
}

void SortedIndex::scanEntries(
    uint32_t startChunk,
    std::string_view startSeekKey,
    std::string_view endSeekKey,
    std::string_view boundKey,
    bool isPointLookup,
    const std::optional<RowRange>& filterRange,
    std::vector<RowRange>& rowRanges) const {
  // Returns true if entries matching the scan range may continue past this
  // chunk boundary. For point lookup, this means the chunk's last key equals
  // the lookup key (duplicates span). For range scan, this means the chunk's
  // last key is below the upper bound.
  auto hasMoreEntries = [&](uint32_t chunkIdx) {
    const auto ck = chunkKey(chunkIdx);
    return isPointLookup ? (ck == boundKey) : (ck < boundKey);
  };

  std::vector<uint32_t> matchedRows;
  for (uint32_t ci = startChunk; ci < numChunks_; ++ci) {
    if (ci > startChunk && !hasMoreEntries(ci - 1)) {
      break;
    }

    auto chunk = loadChunk(ci);
    const uint32_t rowCount = chunk->encoding->rowCount();

    // Seek only in the first chunk. For continuation chunks, the loop guard
    // ensures entries are in range, so start at 0.
    const uint32_t startPos = (ci != startChunk)
        ? 0
        : chunk->encoding->seek(startSeekKey, /*inclusive=*/true)
              .value_or(rowCount);
    if (startPos >= rowCount) {
      break;
    }

    // If all remaining entries in this chunk match, skip the end seek.
    const uint32_t endPos = hasMoreEntries(ci)
        ? rowCount
        : chunk->encoding->seek(endSeekKey, /*inclusive=*/!isPointLookup)
              .value_or(rowCount);
    if (startPos >= endPos) {
      break;
    }

    auto entries = chunk->encoding->materialize(startPos, endPos - startPos);
    for (const auto& entry : entries) {
      matchedRows.emplace_back(extractRowId(entry));
    }
  }

  if (filterRange.has_value()) {
    for (const auto row : matchedRows) {
      if (filterRange->contains(row)) {
        rowRanges.emplace_back(row, row + 1);
      }
    }
  } else {
    for (const auto row : matchedRows) {
      rowRanges.emplace_back(row, row + 1);
    }
  }
}

void SortedIndex::pointLookup(
    std::string_view key,
    const std::optional<RowRange>& filterRange,
    std::vector<RowRange>& rowRanges) const {
  if (key < minKey() || key > maxKey()) {
    ++numMinMaxSkips_;
    return;
  }
  const auto startChunkIdx = findChunkIndex(key);
  if (!startChunkIdx.has_value()) {
    return;
  }
  const auto firstKey = firstSeekKey(key);
  const auto lastKey = lastSeekKey(key);
  scanEntries(
      startChunkIdx.value(),
      firstKey,
      lastKey,
      key,
      /*isPointLookup=*/true,
      filterRange,
      rowRanges);
}

void SortedIndex::rangeScan(
    const velox::serializer::EncodedKeyBounds& bounds,
    const std::optional<RowRange>& filterRange,
    std::vector<RowRange>& rowRanges) const {
  NIMBLE_CHECK(
      bounds.lowerKey.has_value() && bounds.upperKey.has_value(),
      "Range scan requires both lower and upper bounds");
  const auto& lowerKey = bounds.lowerKey.value();
  const auto& upperKey = bounds.upperKey.value();
  if (lowerKey > maxKey() || upperKey <= minKey()) {
    ++numMinMaxSkips_;
    return;
  }
  const auto startChunkIdx = findChunkIndex(lowerKey);
  if (!startChunkIdx.has_value()) {
    return;
  }
  const auto lowerSeekKey = firstSeekKey(lowerKey);
  const auto upperSeekKey = firstSeekKey(upperKey);
  scanEntries(
      startChunkIdx.value(),
      lowerSeekKey,
      upperSeekKey,
      upperKey,
      /*isPointLookup=*/false,
      filterRange,
      rowRanges);
}

std::string_view SortedIndex::minKey() const {
  return sortedIndex_->min_key()->string_view();
}

std::string_view SortedIndex::maxKey() const {
  return chunkKey(numChunks_ - 1);
}

IndexLookup::LookupResult SortedIndex::lookup(
    const LookupRequest& request) const {
  const auto& options = request.options();

  std::vector<RowRange> rowRanges;
  std::vector<uint32_t> resultOffsets;
  resultOffsets.reserve(request.size() + 1);
  resultOffsets.push_back(0);

  for (uint32_t i = 0; i < request.size(); ++i) {
    if (request.mode() == LookupRequest::Mode::PointLookup) {
      ++numPointLookups_;
      pointLookup(request.pointKey(i), options.rowRange, rowRanges);
    } else {
      ++numRangeScans_;
      rangeScan(request.rangeBound(i), options.rowRange, rowRanges);
    }

    numMatchedRows_ += rowRanges.size() - resultOffsets.back();
    resultOffsets.push_back(rowRanges.size());
  }

  return LookupResult{std::move(rowRanges), std::move(resultOffsets)};
}

folly::F14FastMap<std::string, velox::RuntimeMetric> SortedIndex::stats()
    const {
  folly::F14FastMap<std::string, velox::RuntimeMetric> result;
  const auto numPointLookups = numPointLookups_.load();
  if (numPointLookups > 0) {
    result.emplace(kNumPointLookups, velox::RuntimeMetric(numPointLookups));
  }
  const auto numRangeScans = numRangeScans_.load();
  if (numRangeScans > 0) {
    result.emplace(kNumRangeScans, velox::RuntimeMetric(numRangeScans));
  }
  const auto numMinMaxSkips = numMinMaxSkips_.load();
  if (numMinMaxSkips > 0) {
    result.emplace(kNumMinMaxSkips, velox::RuntimeMetric(numMinMaxSkips));
  }
  const auto numMatchedRows = numMatchedRows_.load();
  if (numMatchedRows > 0) {
    result.emplace(kNumMatchedRows, velox::RuntimeMetric(numMatchedRows));
  }
  return result;
}

} // namespace facebook::nimble::index
